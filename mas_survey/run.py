# mas_survey/run.py
from __future__ import annotations
import argparse, csv, json, time, yaml
from pathlib import Path
from typing import Dict, List, Tuple

from mas_survey.retrieval import Indices, load_tfidf, retrieve_candidates
from mas_survey.llm_stance import label_with_llm, together_health_check
from mas_survey.aggregate import aggregate_votes, select_supports


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run MAS pipeline")
    p.add_argument("--config", type=str, default="config.yaml",
                   help="Path to config.yaml (default: config.yaml)")
    p.add_argument("--api_key", type=str, default=None,
                   help="API key (forwarded to Together when labeling=llm).")
    return p.parse_args()


def _read_docs(path: Path) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                obj = json.loads(ln)
                did = obj.get("id")
                if did:
                    out[did] = obj
            except Exception:
                continue
    return out


def _doc_text(obj: dict) -> str:
    title = obj.get("title") or ""
    desc = obj.get("description") or ""
    pc = obj.get("post_content") or ""
    content = obj.get("content") or ""
    return "\n".join([t for t in (title, desc, pc, content) if t]).strip()


def _snippet(text: str, max_chars: int = 550) -> str:
    t = " ".join((text or "").split())
    return (t[:max_chars] + " ...") if len(t) > max_chars else t


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))

    docs_path = Path(cfg["data"]["documents_path"])
    q_path = Path(cfg["data"]["questions_path"])
    out_csv = Path(cfg["run"]["output_csv"])
    audit_log = Path(cfg["run"]["audit_log"])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    audit_log.parent.mkdir(parents=True, exist_ok=True)

    # Load docs & questions
    docs = _read_docs(docs_path)
    questions = json.load(open(q_path, "r", encoding="utf-8"))
    print(f"[run] loaded {len(docs)} docs from {docs_path}")
    print(f"[run] loaded {len(questions)} questions from {q_path}")

    # Load indices built by index.build
    tfidf_X, vocab, row_id_map = load_tfidf(Path(cfg["index"]["output_dir"]))
    whoosh_dir = Path(cfg["whoosh"]["index_dir"]) if cfg["index"].get("build_whoosh", True) else None
    idx = Indices(tfidf_X=tfidf_X, vocab=vocab, row_id_map=row_id_map, whoosh_dir=whoosh_dir)
    print(f"[run] tfidf rows={tfidf_X.shape[0]} cols={tfidf_X.shape[1]} | whoosh_dir={whoosh_dir}")

    # If provider==llm, proactively health-check Together
    provider = str(cfg["labeling"]["provider"]).lower()
    model = cfg["labeling"]["model"] if provider == "llm" else None
    if provider == "llm":
        assert args.api_key, "LLM provider selected but no --api_key given"
        print(f"[run] LLM labeling ENABLED -> model={model}, batch={cfg['labeling']['batch_size']}")
        try:
            ok = together_health_check(
                model=model,
                api_key=args.api_key or "",
                timeout_tokens=8,
            )
            print(f"[run] Together health-check: {ok}")
        except Exception as e:
            raise SystemExit(f"[run] Together health-check FAILED: {e}")

    # Writers
    with out_csv.open("w", encoding="utf-8", newline="") as fw, audit_log.open("w", encoding="utf-8") as flog:
        writer = csv.DictWriter(fw, fieldnames=["question", "distribution", "supports"])
        writer.writeheader()

        for qi, (qtext, qmeta) in enumerate(questions.items(), start=1):
            t0 = time.time()
            options: List[str] = list(qmeta["distribution"].keys())
            print(f"\n[run] Q{qi}/{len(questions)}: options={len(options)}  '{qtext[:110]}{'...' if len(qtext)>110 else ''}'")

            # 1) retrieve
            cand_ids: List[str] = retrieve_candidates(qtext, options, cfg, idx)
            print(f"[run] retrieved candidates={len(cand_ids)} (deduped)")

            # 2) label with LLM (or fallback)
            topN = int(cfg["labeling"]["top_docs_for_label"])
            label_ids = cand_ids[:topN]
            votes: Dict[str, Tuple[str, float]] = {}

            if provider == "llm":
                snippets = [_snippet(_doc_text(docs.get(did, {}))) for did in label_ids]
                votes = label_with_llm(
                    question=qtext,
                    options=options,
                    doc_ids=label_ids,
                    doc_texts=snippets,
                    model=model,
                    api_key=args.api_key or "",
                    batch_size=int(cfg["labeling"]["batch_size"]),
                    temperature=float(cfg["labeling"]["temperature"]),
                    max_output_tokens=int(cfg["labeling"]["max_output_tokens"]),
                )
                parse_success = sum(1 for (opt, _) in votes.values() if opt in options)
                print(f"[run] LLM parse success: {parse_success}/{len(label_ids)}")
            else:
                # simple baseline: assign weak decaying weight to first option
                for i, did in enumerate(label_ids, start=1):
                    votes[did] = (options[0], max(0.05, 1.0 / (i + 10.0)))
                parse_success = len(label_ids)  # by construction

            # 3) aggregate/calibrate
            dist = aggregate_votes(
                options=options,
                votes=votes,
                rank_ids=label_ids,
                temperature=float(cfg["calibration"]["temperature"]),
                dirichlet_alpha=float(cfg["calibration"]["dirichlet_alpha"]),
            )
            # quick sanity: sum to ~1.0
            s = sum(dist.values())
            if not (0.999999 <= s <= 1.000001):
                print(f"[run][warn] distribution sum={s:.8f} (will still write)")

            # 4) supports (exactly N)
            supports = select_supports(
                fused_rank=cand_ids,
                size=int(cfg["supports"]["size"]),
                all_candidates_fallback=idx.row_id_map,  # deterministic padding
            )

            # write row + audit
            writer.writerow({
                "question": qtext,
                "distribution": json.dumps(dist, ensure_ascii=True),
                "supports": json.dumps(supports, ensure_ascii=True),
            })
            flog.write(json.dumps({
                "question": qtext,
                "provider": provider,
                "model": model if provider == "llm" else None,
                "n_candidates": len(cand_ids),
                "labeled": len(label_ids),
                "llm_parse_success": parse_success if provider == "llm" else None,
                "supports_len": len(supports),
                "elapsed_sec": round(time.time() - t0, 3),
            }) + "\n")

    print(f"\n[run] Wrote {out_csv} and audit {audit_log}")


if __name__ == "__main__":
    main()
