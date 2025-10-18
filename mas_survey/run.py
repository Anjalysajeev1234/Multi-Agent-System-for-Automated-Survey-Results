# mas_survey/run.py
from __future__ import annotations
import argparse, csv, json, random, time, re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer

from mas_survey.retrieval import load_tfidf, Indices, retrieve_candidates
from mas_survey.llm_stance import label_with_llm
from mas_survey.aggregate import aggregate_votes, select_supports


def _set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)


def _load_docs_map(documents_path: str) -> Dict[str, str]:
    mp: Dict[str, str] = {}
    with open(documents_path, "r", encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            did = str(o["id"])
            title = o.get("title") or ""
            desc = o.get("description") or ""
            pc = o.get("post_content") or ""
            content = o.get("content") or ""
            mp[did] = "\n".join([x for x in [title, desc, pc, content] if x]).strip()
    return mp


def _load_questions(path: str) -> Dict[str, dict]:
    return json.loads(open(path, "r", encoding="utf-8").read())


def _audit_write(audit_path: Path, obj: dict):
    with audit_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=True) + "\n")


def _split_sents(txt: str) -> List[str]:
    # crude sentence split, fast & deterministic
    return re.split(r'(?<=[.!?])\s+', (txt or "").strip())


def _build_vocab_vectorizer(vocab: Dict[str, int]) -> TfidfVectorizer:
    tv = TfidfVectorizer(vocabulary=vocab, lowercase=True, token_pattern=r"(?u)\b\w\w+\b")
    # hack-fit to enable transform; fit on dummy corpus of vocab tokens
    tv.fit([" ".join(vocab.keys())])
    return tv


def _best_snippet(txt: str, tv: TfidfVectorizer, top_k_terms: int = 10, max_sent: int = 3) -> str:
    sents = _split_sents(txt)[:14]  # limit to 14 sentences
    if not sents:
        return ""
    Xs = tv.transform(sents).tocsr()
    # pseudo-query: top vocab tokens (stable proxy for matching)
    # (we could also build from question terms; this keeps it general + fast)
    qs = tv.transform([" ".join(list(tv.vocabulary_.keys())[:top_k_terms])])
    sims = (qs @ Xs.T).A.ravel()
    idx = np.argsort(-sims)[:max_sent]
    return " ".join(sents[i] for i in idx)


def main():
    ap = argparse.ArgumentParser(description="Run MAS pipeline (BM25 per-option + snippets + LLM).")
    ap.add_argument("--config", type=str, default="config.yaml")
    ap.add_argument("--api_key", type=str, default=None, help="Together API key if using LLM.")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    _set_seed(int(cfg.get("seed", 42)))

    # load indices
    index_dir = Path(cfg["index"]["output_dir"])
    tfidf_X, vocab, row_id_map = load_tfidf(index_dir)
    whoosh_dir = Path(cfg["whoosh"]["index_dir"]) if Path(cfg["whoosh"]["index_dir"]).exists() else None
    faiss_index_path = index_dir / "faiss.index"
    if not faiss_index_path.exists():
        faiss_index_path = None

    idx = Indices(
        tfidf_X=tfidf_X, vocab=vocab, row_id_map=row_id_map,
        whoosh_dir=whoosh_dir,
        faiss_index_path=faiss_index_path,
    )

    docs_map = _load_docs_map(cfg["data"]["documents_path"])
    corpus_all_ids = sorted(docs_map.keys())  # deterministic fallback pool

    questions = _load_questions(cfg["data"]["questions_path"])

    out_csv = Path(cfg["run"]["output_csv"])
    audit_path = Path(cfg["run"]["audit_log"])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    if audit_path.exists():
        audit_path.unlink()

    provider = cfg["labeling"]["provider"]
    llm_model = cfg["labeling"]["model"]
    batch_size = int(cfg["labeling"]["batch_size"])
    max_output_tokens = int(cfg["labeling"]["max_output_tokens"])
    temperature = float(cfg["labeling"]["temperature"])

    dirichlet_alpha = float(cfg["calibration"]["dirichlet_alpha"])
    temp_cal = float(cfg["calibration"]["temperature"])

    supports_k = int(cfg["supports"]["size"])
    per_q_cap = float(cfg["run"].get("per_question_time_cap_sec", 0.0))

    # TF-IDF vectorizer for snippet scoring
    tv = _build_vocab_vectorizer(vocab)

    with out_csv.open("w", encoding="utf-8", newline="") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(["question", "distribution", "supports"])

        for q_idx, (q_text, q_obj) in enumerate(questions.items(), 1):
            options = list(q_obj["distribution"].keys())
            t0 = time.time()

            # === retrieval: BM25 fan-out + RRF + dedup ===
            cand_ids = retrieve_candidates(
                question=q_text,
                options_for_q=options,
                cfg=cfg,
                idx=idx,
                docid2text=docs_map,
            )

            # bounded fallback: if too few, add one TF-IDF pass and merge
            if len(cand_ids) < supports_k:
                cfg2 = dict(cfg)
                cfg2["retrieval"] = dict(cfg["retrieval"])
                cfg2["retrieval"]["sparse_backend"] = "tfidf"
                extra = retrieve_candidates(q_text, options, cfg2, idx, docs_map)
                seen = set(cand_ids)
                for did in extra:
                    if did not in seen:
                        cand_ids.append(did); seen.add(did)

            # prepare texts for labeling (top precision chunk)
            L = int(cfg["labeling"]["top_docs_for_label"])
            label_ids = cand_ids[:L]

            # --- snippets (2–3 best sentences) ---
            label_texts = [_best_snippet(docs_map.get(d, ""), tv) for d in label_ids]

            # === labeling ===
            if provider == "llm":
                if not args.api_key:
                    raise RuntimeError("LLM labeling selected but no --api_key provided.")
                doc2vote = label_with_llm(
                    question=q_text,
                    options=options,
                    doc_ids=label_ids,
                    doc_texts=label_texts,
                    model=llm_model,
                    api_key=args.api_key,
                    batch_size=batch_size,
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                )
                parse_success = sum(1 for v in doc2vote.values() if v[0] in options)
                # confidence shaping with rank weights
                rank_weight = np.linspace(1.0, 0.6, num=max(1, len(label_ids)))
                for i, did in enumerate(label_ids):
                    if did in doc2vote:
                        opt, conf = doc2vote[did]
                        doc2vote[did] = (opt, float(0.5*conf + 0.5*rank_weight[i]))
            else:
                # cosine-only fallback: assign option[0] with a rank-decay confidence
                doc2vote = {}
                confs = np.linspace(1.0, 0.6, num=max(1, len(label_ids)))
                for did, c in zip(label_ids, confs):
                    doc2vote[did] = (options[0], float(c))
                parse_success = len(label_ids)

            # === aggregate ===
            dist = aggregate_votes(
                options=options,
                doc2vote=doc2vote,
                dirichlet_alpha=dirichlet_alpha,
                temperature=temp_cal,
            )
            # strict renorm and safety
            s = sum(dist.values()) or 1.0
            dist = {k: max(0.0, v) for k, v in dist.items()}
            s = sum(dist.values()) or 1.0
            dist = {k: v / s for k, v in dist.items()}

            # === supports (exactly 100) ===
            supports = select_supports(
                ranked_unique_ids=cand_ids[:supports_k*3],   # prioritize same-query pool
                required_k=supports_k,
                all_candidates_fallback=cand_ids,
                corpus_all_ids=corpus_all_ids,               # rarely needed now
            )

            # === write row ===
            dist_json = json.dumps(dist, ensure_ascii=True)
            supports_json = json.dumps(supports, ensure_ascii=True)
            writer.writerow([q_text, dist_json, supports_json])

            # === audit ===
            dt = time.time() - t0
            _audit_write(audit_path, {
                "question_idx": q_idx,
                "question": q_text[:200],
                "n_candidates": len(cand_ids),
                "n_labeled": len(label_ids),
                "parse_success": int(parse_success),
                "provider": provider,
                "whoosh": bool(whoosh_dir is not None),
                "used_dense": False,  # query embeddings not built
                "time_sec": round(dt, 3),
            })

            if per_q_cap > 0 and dt > per_q_cap:
                print(f"[run] Time cap exceeded for a question ({dt:.2f}s > {per_q_cap:.2f}s). Continuing...")

    print(f"[run] Wrote {out_csv}")
    print(f"[run] Audit at {audit_path}")


if __name__ == "__main__":
    main()
