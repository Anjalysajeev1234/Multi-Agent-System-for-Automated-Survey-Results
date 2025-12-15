# index/build.py
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
import yaml
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

# Whoosh is optional; controlled by config.index.build_whoosh
WHOOSH_OK = True
try:
    from whoosh import fields, index as windex
except Exception:
    WHOOSH_OK = False


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build indexes from documents.jsonl")
    p.add_argument("--config", type=str, default="config.yaml")
    p.add_argument("--api_key", type=str, default=None)  # accepted but unused here
    return p.parse_args()


def _load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _read_docs(path: Path) -> list[dict]:
    docs = []
    with path.open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                obj = json.loads(ln)
            except Exception:
                continue
            docs.append(obj)
    return docs


def _doc_text(obj: dict) -> str:
    title = obj.get("title") or ""
    desc = obj.get("description") or ""
    pc = obj.get("post_content") or ""
    content = obj.get("content") or ""
    return "\n".join([t for t in (title, desc, pc, content) if t]).strip()


def _build_tfidf(cfg: dict, docs: list[dict], outdir: Path) -> None:
    tf = cfg["tfidf"]
    nmin, nmax = int(tf["ngram_min"]), int(tf["ngram_max"])
    max_feat, min_df = int(tf["max_features"]), int(tf["min_df"])

    texts = [_doc_text(d) for d in docs]
    vec = TfidfVectorizer(
        lowercase=True,
        ngram_range=(nmin, nmax),
        max_features=max_feat,
        min_df=min_df,
        token_pattern=r"(?u)\b\w\w+\b",
        dtype=np.float32,
        norm="l2",
        use_idf=True,
    )
    X = vec.fit_transform(texts).astype(np.float32)
    vocab = {str(tok): int(idx) for tok, idx in vec.vocabulary_.items()}
    # Persist
    outdir.mkdir(parents=True, exist_ok=True)
    sparse.save_npz(outdir / "tfidf_index.npz", X)
    (outdir / "tfidf_vocab.json").write_text(
        json.dumps({"vocabulary": vocab}, ensure_ascii=True),
        encoding="utf-8",
    )
    (outdir / "tfidf_id_map.json").write_text(
        json.dumps([d.get("id") for d in docs], ensure_ascii=True),
        encoding="utf-8",
    )
    print(f"[index.build] TF-IDF saved to {outdir}")


def _build_whoosh(cfg: dict, docs: list[dict]) -> None:
    if not WHOOSH_OK:
        print("[index.build] Whoosh not available; skipping.")
        return
    wdir = Path(cfg["whoosh"]["index_dir"])
    wdir.mkdir(parents=True, exist_ok=True)
    schema = fields.Schema(
        id=fields.ID(stored=True, unique=True),
        text=fields.TEXT(stored=False),
    )
    if not windex.exists_in(str(wdir)):
        ix = windex.create_in(str(wdir), schema)
    else:
        ix = windex.open_dir(str(wdir))
        ix.close()
        for p in wdir.glob("*"):
            try: p.unlink()
            except Exception: pass
        ix = windex.create_in(str(wdir), schema)

    writer = ix.writer(limitmb=512, procs=1, multisegment=True)
    for d in docs:
        did = d.get("id")
        if not did:
            continue
        writer.add_document(id=str(did), text=_doc_text(d))
    writer.commit()
    print(f"[index.build] Whoosh BM25 index built at {wdir}")


def main() -> None:
    args = parse_args()
    cfg = _load_cfg(args.config)

    docs_path = Path(cfg["data"]["documents_path"])
    outdir = Path(cfg["index"]["output_dir"])
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"[index.build] Documents: {docs_path}")
    docs = _read_docs(docs_path)
    print(f"[index.build] Loaded {len(docs)} docs.")

    # TF-IDF (always used for cosine/dedup)
    if bool(cfg["index"].get("build_sparse", True)):
        _build_tfidf(cfg, docs, outdir)

    # Whoosh BM25 (optional)
    if bool(cfg["index"].get("build_whoosh", True)):
        _build_whoosh(cfg, docs)

    print("[index.build] Done.")


if __name__ == "__main__":
    main()
