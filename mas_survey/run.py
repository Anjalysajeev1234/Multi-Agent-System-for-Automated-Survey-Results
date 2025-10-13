# mas_survey/run.py
"""
Run the MAS pipeline and write a Kaggle-ready CSV.

CLI:
    python -m mas_survey.run --config config.yaml --api_key <YOUR_KEY or DUMMY>

This variant supports:
- Retrieval via mas_survey.retrieval: BM25 / TF-IDF / + optional FAISS fusion
- Labeling provider: "cosine" (fast, no LLM)
- Guaranteed 100 unique supports per question
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import yaml
from scipy import sparse
from sklearn.preprocessing import normalize

from mas_survey.retrieval import (
    RetrievalResources,
    retrieve_candidates,
    _tfidf_transform_short,  # reuse the tiny transformer
)

# ---------------- utils ----------------

def _load_cfg(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _load_documents(jsonl_path: str) -> Dict[str, Dict]:
    out = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            out[str(obj["id"])] = obj
    return out

def _concat_text(doc: Dict) -> str:
    title = doc.get("title") or ""
    description = doc.get("description") or ""
    post_content = doc.get("post_content") or ""
    content = doc.get("content") or ""
    return "\n".join([p for p in [title, description, post_content, content] if p]).strip()

def _load_questions(path: str) -> Dict[str, Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _load_tfidf(art_dir: Path):
    X = sparse.load_npz(art_dir / "tfidf_index.npz").tocsr().astype(np.float32)
    with open(art_dir / "tfidf_vocab.json", "r", encoding="utf-8") as f:
        vocab_payload = json.load(f)
    with open(art_dir / "tfidf_id_map.json", "r", encoding="utf-8") as f:
        id_list = json.load(f)
    vocab = vocab_payload.get("vocabulary", {})
    idf = np.array(vocab_payload.get("idf") or [], dtype=np.float32)
    # X already L2-normalized in builder; keep consistent just in case
    X = normalize(X, norm="l2", axis=1, copy=False)
    return X, vocab, idf, id_list

def _cosine(a: sparse.csr_matrix, b: sparse.csr_matrix) -> np.ndarray:
    # assumes rows are L2-normalized
    return (a @ b.T).toarray()

def _softmax(scores: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    s = np.array(scores, dtype=np.float32)
    m = float(np.max(s)) if s.size else 0.0
    z = np.exp((s - m) / max(temperature, 1e-6))
    zsum = float(z.sum())
    return z / zsum if zsum > 0 else np.ones_like(s) / max(1, len(s))

def aggregate_distribution(option_scores: Dict[str, float], temperature: float, alpha: float) -> Dict[str, float]:
    opts = list(option_scores.keys())
    vec = np.array([option_scores[o] for o in opts], dtype=np.float32)
    prob = _softmax(vec, temperature=temperature)
    prob = (prob + float(alpha))
    prob = prob / prob.sum()
    return {o: float(p) for o, p in zip(opts, prob)}

# -------------- CLI --------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run MAS pipeline (retrieval + cosine aggregation).")
    p.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    p.add_argument("--api_key", type=str, default=None, help="Accepted but unused in cosine mode.")
    return p.parse_args()

# -------------- main -------------------

def main() -> None:
    args = parse_args()
    cfg = _load_cfg(args.config)

    # --- paths & settings ---
    docs_path = cfg["data"]["documents_path"]
    q_path = cfg["data"]["questions_path"]
    artifacts_dir = Path(cfg.get("index", {}).get("output_dir", "./artifacts/index"))
    output_csv = cfg.get("run", {}).get("output_csv", "./artifacts/submission_js.csv")
    audit_log = cfg.get("run", {}).get("audit_log", "./artifacts/audit.jsonl")

    temperature = float(cfg.get("labeling", {}).get("temperature", 0.8))
    alpha = float(cfg.get("labeling", {}).get("dirichlet_alpha", 0.2))
    top_docs_for_label = int(cfg.get("labeling", {}).get("top_docs_for_label", 160))
    supports_k = int(cfg.get("supports", {}).get("size", 100))

    # --- load resources ---
    docs = _load_documents(docs_path)
    X, vocab, idf, id_map = _load_tfidf(artifacts_dir)
    id_to_row = {did: i for i, did in enumerate(id_map)}
    questions = _load_questions(q_path)

    # optional dense resources (only used if you later set use_dense:true)
    dense_emb_path = Path(cfg["data"].get("embeddings_path", "")) if cfg["data"].get("embeddings_path") else None
    dense_embeddings = None
    dense_id_map = None
    if dense_emb_path and dense_emb_path.exists():
        data = np.load(dense_emb_path, allow_pickle=True)
        dense_id_map = list(map(str, data["ids"].tolist()))
        dense_embeddings = data["embeddings"].astype(np.float32)
        norms = np.linalg.norm(dense_embeddings, axis=1, keepdims=True) + 1e-12
        dense_embeddings = dense_embeddings / norms

    res = RetrievalResources(
        X_tfidf=X,
        vocab=vocab,
        idf=idf,
        id_map=id_map,
        id_to_row=id_to_row,
        whoosh_dir=Path(cfg.get("whoosh", {}).get("index_dir", "./artifacts/whoosh_index")),
        faiss_index_path=Path(cfg.get("index", {}).get("output_dir", "./artifacts/index")) / "faiss.index",
        dense_id_map=dense_id_map,
        dense_embeddings=dense_embeddings,
    )

    # --- outputs ---
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(audit_log).parent.mkdir(parents=True, exist_ok=True)
    audit_f = open(audit_log, "w", encoding="utf-8")

    with open(output_csv, "w", encoding="utf-8", newline="") as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=["question", "distribution", "supports"])
        writer.writeheader()

        for q_text, q_obj in questions.items():
            options: List[str] = list(q_obj["distribution"].keys())

            # 1) retrieval (BM25 / TF-IDF / +optional FAISS via config)
            cand_ids, cand_rows, X_rows, dbg = retrieve_candidates(cfg, q_text, options, res)

            # 2) shortlist for pseudo-labeling
            L = min(top_docs_for_label, len(cand_ids))
            cand_ids_L = cand_ids[:L]
            X_rows_L = X_rows[:L] if L else X_rows

            # 3) cosine pseudo-labeling: doc rows vs option texts
            if options:
                opt_vec = _tfidf_transform_short(options, vocab, idf)
                sims = _cosine(X_rows_L, opt_vec)  # shape: [L, |options|]
                option_scores = {opt: float(sims[:, j].sum()) for j, opt in enumerate(options)}
            else:
                option_scores = {}

            # 4) aggregate + calibrate → valid probability distribution
            dist = aggregate_distribution(option_scores, temperature=temperature, alpha=alpha)

            # 5) supports: first K after dedup; pad to exactly 100 unique IDs
            supports = cand_ids[:supports_k]
            if len(set(supports)) < supports_k:
                seen = set(supports)
                # first pad from remaining candidates
                pad = [did for did in cand_ids if did not in seen]
                # then pad from global id_map if still short
                if len(seen) + len(pad) < supports_k:
                    pad += [did for did in id_map if did not in seen]
                supports = list(seen) + [did for did in pad if did not in seen][: supports_k - len(seen)]
            # final trim (safety)
            supports = supports[:supports_k]

            # 6) audit
            audit_rec = {
                "question": q_text,
                "options": options,
                "option_scores": option_scores,
                "distribution": dist,
                "supports_count": len(supports),
                "retrieval_dbg": dbg,
            }
            audit_f.write(json.dumps(audit_rec, ensure_ascii=True) + "\n")

            # 7) write CSV row
            writer.writerow({
                "question": q_text,
                "distribution": json.dumps(dist, ensure_ascii=True),
                "supports": json.dumps(supports, ensure_ascii=True),
            })

    audit_f.close()
    print(f"[run] Wrote {output_csv} and {audit_log}")


if __name__ == "__main__":
    main()
