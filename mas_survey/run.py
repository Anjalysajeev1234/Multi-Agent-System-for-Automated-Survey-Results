# mas_survey/run.py
"""
Hybrid retrieval + LLM stance labeling + LLM-informed supports ranking.

- Sparse TF-IDF (always) + optional FAISS (if present) with RRF fusion
- Per-option sparse retrieval for better coverage
- TogetherAI JSON-labeled stances (cached), or cosine fallback
- Dirichlet + temperature aggregation (single normalisation step)
- Supports ranking:
    * per-option top-ups (quota)
    * global sort by LLM confidence then fused rank
    * subreddit diversity cap
Produces: artifacts/submission_js.csv and artifacts/audit.jsonl
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import yaml
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from mas_survey.llm_stance import label_with_llm

FAISS_OK = False
try:
    import faiss  # type: ignore
    FAISS_OK = True
except Exception:
    FAISS_OK = False


# --------------------------- helpers: IO ---------------------------

def _load_cfg(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_questions(path: str) -> Dict[str, Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_documents(jsonl_path: str) -> Dict[str, Dict]:
    out = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            out[obj["id"]] = obj
    return out


def _concat_text(doc: Dict) -> str:
    title = doc.get("title") or ""
    description = doc.get("description") or ""
    post_content = doc.get("post_content") or ""
    content = doc.get("content") or ""
    return "\n".join([p for p in [title, description, post_content, content] if p]).strip()


# ---------------------- load TF-IDF artefacts ----------------------

def _load_tfidf(artifacts_dir: Path):
    tfidf_index_path = artifacts_dir / "tfidf_index.npz"
    tfidf_vocab_path = artifacts_dir / "tfidf_vocab.json"
    id_map_path = artifacts_dir / "tfidf_id_map.json"

    if not tfidf_index_path.exists() or not tfidf_vocab_path.exists() or not id_map_path.exists():
        print("[mas_survey.run] Missing TF-IDF artifacts. Run: python -m index.build --config config.yaml", file=sys.stderr)
        sys.exit(1)

    X = sparse.load_npz(tfidf_index_path)
    vocab_payload = json.loads(tfidf_vocab_path.read_text(encoding="utf-8"))
    id_map = json.loads(id_map_path.read_text(encoding="utf-8"))

    vec = TfidfVectorizer(
        lowercase=vocab_payload["lowercase"],
        ngram_range=tuple(vocab_payload["ngram_range"]),
        token_pattern=vocab_payload["token_pattern"],
        max_features=vocab_payload["max_features"],
        dtype=np.float32,
    )
    vec.vocabulary_ = {k: int(v) for k, v in vocab_payload["vocabulary"].items()}
    if vocab_payload["idf"] is not None:
        vec.idf_ = np.array(vocab_payload["idf"], dtype=np.float32)
    vec._tfidf._idf_diag = sparse.spdiags(vec.idf_, diags=0, m=len(vec.idf_), n=len(vec.idf_))

    return vec, X, id_map


# --------------------------- retrieval -----------------------------

def _cosine_query_sparse(vec: TfidfVectorizer, X: sparse.csr_matrix, query_text: str) -> np.ndarray:
    q = vec.transform([query_text])
    q = normalize(q, norm="l2", axis=1, copy=False)
    sims = (X @ q.T).toarray().ravel().astype(np.float32)  # L2-normalised so dot == cosine
    return sims


def _dedup_by_cosine(X: sparse.csr_matrix, order: np.ndarray, thresh: float = 0.98, cap: int = 300) -> List[int]:
    picked: List[int] = []
    kept_rows: List[int] = []
    for idx in order:
        if len(picked) >= cap:
            break
        xi = X[idx]
        keep = True
        for j in kept_rows:
            xj = X[j]
            sim = float(xi.multiply(xj).sum())  # cosine
            if sim >= thresh:
                keep = False
                break
        if keep:
            picked.append(int(idx))
            kept_rows.append(int(idx))
    return picked


def _rrf_fuse(scores_list: List[np.ndarray], k: int = 60) -> np.ndarray:
    fused = np.zeros_like(scores_list[0], dtype=np.float64)
    for scores in scores_list:
        order = np.argsort(-scores)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(1, len(scores) + 1)
        fused += 1.0 / (k + ranks)
    fused = (fused - fused.min()) / (fused.max() - fused.min() + 1e-12)
    return fused.astype(np.float32)


# --------- dense lookup using centroid of top-TF-IDF embeddings ---------

def _maybe_load_faiss(artifacts_dir: Path):
    faiss_path = artifacts_dir / "faiss.index"
    dense_map_path = artifacts_dir / "dense_id_map.json"
    if not (FAISS_OK and faiss_path.exists() and dense_map_path.exists()):
        return None, None
    index = faiss.read_index(str(faiss_path))
    dense_id_map = json.loads(dense_map_path.read_text(encoding="utf-8"))
    return index, dense_id_map


def _dense_scores_from_centroid(index, dense_id_map: List[str],
                                id_to_emb: Dict[str, np.ndarray],
                                tfidf_top_doc_ids: List[str]) -> np.ndarray:
    # Build centroid of embeddings of top TF-IDF docs (PRF-style)
    vecs = []
    for did in tfidf_top_doc_ids:
        v = id_to_emb.get(did)
        if v is not None:
            vecs.append(v)
    if not vecs:
        return np.zeros(len(dense_id_map), dtype=np.float32)
    centroid = np.mean(np.stack(vecs, axis=0), axis=0).astype(np.float32)
    # L2 normalise
    norm = np.linalg.norm(centroid) + 1e-12
    centroid = centroid / norm
    D, I = index.search(centroid.reshape(1, -1), len(dense_id_map))  # distances, indices
    scores = -D.ravel().astype(np.float32)  # monotonic transform OK for ranking
    # Expand to full vector aligned to dense_id_map
    full = np.zeros(len(dense_id_map), dtype=np.float32)
    full[I.ravel()] = scores
    return full


# ---------------------- aggregation / outputs ----------------------

def _aggregate(option_scores: Dict[str, float], alpha: float, temperature: float) -> Dict[str, float]:
    """Temperature -> softmax; add alpha once; normalise once."""
    opts = list(option_scores.keys())
    raw = np.array([option_scores[o] for o in opts], dtype=np.float64)
    ex = np.exp((raw - raw.max()) / max(temperature, 1e-6))
    ex = ex + float(alpha)                    # Dirichlet-style smoothing
    probs = ex / (ex.sum() + 1e-12)           # single normalisation
    probs = np.clip(probs, 0.0, 1.0)
    probs = probs / (probs.sum() + 1e-12)
    return {o: float(p) for o, p in zip(opts, probs)}


def _write_csv(rows: List[Tuple[str, Dict[str, float], List[str]]], out_path: str) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["question", "distribution", "supports"])
        for q, dist, supports in rows:
            dist_json = json.dumps(dist, ensure_ascii=True)
            supports_json = json.dumps(supports, ensure_ascii=True)
            w.writerow([q, dist_json, supports_json])


# ------------------------------ CLI --------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run MAS pipeline (hybrid + LLM stance + improved supports)")
    p.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    p.add_argument("--api_key", type=str, default=None, help="TogetherAI API key (only if provider=together).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = _load_cfg(args.config)
    np.random.seed(cfg.get("seed", 42))

    docs_path = cfg["data"]["documents_path"]
    q_path = cfg["data"]["questions_path"]
    artifacts_dir = Path(cfg.get("index", {}).get("output_dir", "./artifacts/index"))
    embeddings_npz = cfg["data"].get("embeddings_path", "")

    print(f"[run] Documents: {docs_path}")
    print(f"[run] Questions:  {q_path}")
    print(f"[run] Index dir:  {artifacts_dir}")

    # Load artefacts & data
    vec, X, tfidf_id_map = _load_tfidf(artifacts_dir)
    documents = _load_documents(docs_path)
    questions = _load_questions(q_path)

    # Optional dense
    faiss_index, dense_id_map = _maybe_load_faiss(artifacts_dir)
    id_to_emb: Dict[str, np.ndarray] = {}
    if faiss_index is not None and embeddings_npz and Path(embeddings_npz).exists():
        data = np.load(embeddings_npz, allow_pickle=True)
        dense_ids = list(map(str, data["ids"].tolist()))
        emb = data["embeddings"].astype(np.float32)
        emb /= (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
        id_to_emb = {did: emb[i] for i, did in enumerate(dense_ids)}
        print(f"[run] Dense retrieval enabled.")
    else:
        print(f"[run] Dense retrieval disabled (TF-IDF only).")

    # Config knobs
    retrieval_cfg = cfg.get("retrieval", {})
    filtering_cfg = cfg.get("filtering", {})
    labeling_cfg = cfg.get("labeling", {})
    supports_cfg = cfg.get("supports", {})
    run_cfg = cfg.get("run", {})

    top_k_sparse = int(retrieval_cfg.get("top_k_sparse", 800))
    use_dense = bool(retrieval_cfg.get("use_dense", True))
    rrf_k = int(retrieval_cfg.get("rrf_k", 60))

    dedup_thresh = float(filtering_cfg.get("dedup_sim_thresh", 0.98))
    label_cap = int(labeling_cfg.get("top_docs_for_label", 160))
    alpha = float(labeling_cfg.get("dirichlet_alpha", 0.2))
    temperature = float(labeling_cfg.get("temperature", 0.8))

    support_k = int(supports_cfg.get("size", 100))
    div_cap = int(supports_cfg.get("subreddit_cap", 5))  # optional; default 5

    # Pre-materialise doc texts in tfidf order for fast lookup
    tfidf_texts = [_concat_text(documents[doc_id]) for doc_id in tfidf_id_map]

    rows: List[Tuple[str, Dict[str, float], List[str]]] = []
    audit_lines: List[str] = []

    for q_text, q_payload in questions.items():
        options = list(q_payload["distribution"].keys())
        base_query = q_text + " " + " ".join(options)

        # ---- sparse retrieval (base) ----
        sparse_scores = _cosine_query_sparse(vec, X, base_query)

        # ---- per-option sparse scores (for later mixing/supports) ----
        opt_sparse_scores = []
        for opt in options:
            qopt = q_text + " " + opt
            opt_sparse_scores.append(_cosine_query_sparse(vec, X, qopt))
        opt_sparse_scores = np.stack(opt_sparse_scores, axis=1)  # [num_docs, num_opts]

        # ---- optional dense retrieval using centroid of top sparse ----
        fused_base = sparse_scores.copy()
        if use_dense and faiss_index is not None and id_to_emb:
            sparse_order = np.argsort(-sparse_scores)[:top_k_sparse]
            seed_ids = [tfidf_id_map[i] for i in sparse_order[:50]]
            dense_scores_full = _dense_scores_from_centroid(
                faiss_index, dense_id_map, id_to_emb, seed_ids
            )
            # Map dense scores onto TF-IDF doc universe by ID (missing IDs get 0)
            dense_on_tfidf = np.zeros_like(fused_base, dtype=np.float32)
            dense_pos = {did: pos for pos, did in enumerate(dense_id_map)}
            for i, did in enumerate(tfidf_id_map):
                j = dense_pos.get(did)
                if j is not None:
                    dense_on_tfidf[i] = dense_scores_full[j]
            fused_base = _rrf_fuse([sparse_scores, dense_on_tfidf], k=rrf_k)

        # shortlist + dedup
        order = np.argsort(-fused_base)[:top_k_sparse]
        picked_rows = _dedup_by_cosine(X, order, thresh=dedup_thresh, cap=300)

        # ---- label set ----
        consider_rows = picked_rows[:label_cap]
        consider_ids = [tfidf_id_map[i] for i in consider_rows]
        consider_texts = [tfidf_texts[i] for i in consider_rows]

        # ---- labeling: LLM (Together) or cosine fallback ----
        option_scores = {opt: 0.0 for opt in options}
        doc2lab: Dict[str, Tuple[str, float]] = {}

        provider = str(labeling_cfg.get("provider", "together")).lower()
        if provider == "together":
            model = labeling_cfg.get("model", "Qwen/Qwen2.5-7B-Instruct-Turbo")
            api_key = args.api_key or ""
            if not api_key:
                print("[mas_survey.run] ERROR: provider=together but no API key provided.", file=sys.stderr)
                sys.exit(1)

            print(f"[run] LLM labeling: {len(consider_ids)} docs, model={model} …", flush=True)
            doc2lab = label_with_llm(
                question=q_text,
                options=options,
                doc_ids=consider_ids,
                doc_texts=consider_texts,
                model=model,
                api_key=api_key,
                batch_size=int(labeling_cfg.get("batch_size", 8)),
            )
            for _, (opt, conf) in doc2lab.items():
                if opt in option_scores:
                    option_scores[opt] += float(conf)
            print(f"[run] LLM done, votes={len(doc2lab)}", flush=True)
        else:
            # Cosine fallback: treat option with highest doc cosine as label; confidence = softmax doc-wise
            option_query_texts = [q_text + " " + opt for opt in options]
            option_vecs = vec.transform(option_query_texts)
            option_vecs = normalize(option_vecs, norm="l2", axis=1, copy=False)

            doc_mat = vec.transform(consider_texts)
            doc_mat = normalize(doc_mat, norm="l2", axis=1, copy=False)

            cos = (doc_mat @ option_vecs.T).toarray()  # docs x options
            # doc-wise softmax to create pseudo-confidence
            for did, row in zip(consider_ids, cos):
                row = row - row.max()
                ex = np.exp(row / max(temperature, 1e-6))
                probs = ex / (ex.sum() + 1e-12)
                j = int(np.argmax(probs))
                conf = float(probs[j])
                chosen = options[j]
                doc2lab[did] = (chosen, conf)
                option_scores[chosen] += conf

        # ---- aggregate to distribution ----
        distribution = _aggregate(option_scores, alpha=alpha, temperature=temperature)

        # ===================== supports ranking ======================
        # Build per-doc confidence map (best option confidence)
        doc_best_conf: Dict[str, float] = {}
        for did, (_, conf) in doc2lab.items():
            doc_best_conf[did] = float(conf)

        # Per-option buckets (for top-ups)
        opt2docs: Dict[str, List[Tuple[str, float]]] = {o: [] for o in options}
        for did, (opt, conf) in doc2lab.items():
            opt2docs[opt].append((did, float(conf)))
        for arr in opt2docs.values():
            arr.sort(key=lambda x: -x[1])

        # quota per option (ensure coverage)
        per_opt_quota = max(10, support_k // max(3, len(options)))
        seeded: List[str] = []
        for opt, arr in opt2docs.items():
            seeded.extend([d for d, _ in arr[:per_opt_quota]])

        # Global candidate list (seeded + remaining in fused order)
        picked_ids = [tfidf_id_map[r] for r in picked_rows]
        candidates = list(dict.fromkeys(seeded + picked_ids))  # preserve order, dedup

        # Tiebreaker by fused rank position
        row_pos = {tfidf_id_map[r]: pos for pos, r in enumerate(picked_rows)}

        def fused_pos(d: str) -> int:
            return row_pos.get(d, 10 ** 9)

        # Sort: primary by LLM confidence desc (0 if missing), secondary by fused rank asc
        candidates.sort(key=lambda d: (-(doc_best_conf.get(d, 0.0)), fused_pos(d)))

        # Subreddit diversity cap
        supports: List[str] = []
        seen = set()
        sub_seen: Dict[str, int] = {}
        for did in candidates:
            if did in seen:
                continue
            sub = (documents.get(did, {}).get("group") or "")[:128]
            if sub and sub_seen.get(sub, 0) >= div_cap:
                continue
            supports.append(did)
            seen.add(did)
            if sub:
                sub_seen[sub] = sub_seen.get(sub, 0) + 1
            if len(supports) >= support_k:
                break

        # Fill remainder if needed
        if len(supports) < support_k:
            for did in picked_ids:
                if did in seen:
                    continue
                supports.append(did)
                seen.add(did)
                if len(supports) >= support_k:
                    break

        if len(supports) < support_k:
            print(f"[mas_survey.run] Warning: only {len(supports)} supports available; use full corpus for 100.", flush=True)

        rows.append((q_text, distribution, supports))
        audit_lines.append(json.dumps({
            "question": q_text,
            "n_options": len(options),
            "picked": len(picked_rows),
            "used_for_label": len(consider_rows),
            "option_scores": option_scores,
            "distribution": distribution,
            "supports_len": len(supports),
            "per_opt_quota": per_opt_quota
        }))

    out_csv = run_cfg.get("output_csv", "./artifacts/submission_js.csv")
    _write_csv(rows, out_csv)
    print(f"[run] Wrote CSV → {out_csv}")

    audit_path = run_cfg.get("audit_log", "./artifacts/audit.jsonl")
    Path(audit_path).parent.mkdir(parents=True, exist_ok=True)
    with open(audit_path, "w", encoding="utf-8") as f:
        for line in audit_lines:
            f.write(line + "\n")
    print(f"[run] Wrote audit log → {audit_path}")


if __name__ == "__main__":
    main()
