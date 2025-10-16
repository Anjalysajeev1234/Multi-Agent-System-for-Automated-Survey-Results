# index/build.py
"""
Builds local indexes for retrieval:
- TF-IDF (sparse CSR) with JSON-safe metadata
- Whoosh BM25 inverted index (optional)
- FAISS dense index from id_to_embedding.npz (optional)

CLI (as required by the brief):
    python -m index.build --config config.yaml --api_key <KEY>

Notes:
- Respects config.index.force_rebuild: if false, skips work when artifacts exist.
- FAISS: set index.faiss_type: "flat" for instant build, or "hnsw" for faster queries after a slower build.
"""

from __future__ import annotations
import argparse, json, shutil
from pathlib import Path
from typing import Dict, List

import numpy as np
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from scipy import sparse
import yaml

# Optional deps: Whoosh (BM25)
WHOOSH_OK = True
try:
    from whoosh.fields import Schema, ID, TEXT
    from whoosh import index as windex
    from whoosh.analysis import StandardAnalyzer
except Exception:
    WHOOSH_OK = False

# Optional deps: FAISS (dense)
FAISS_OK = True
try:
    import faiss  # type: ignore
except Exception:
    FAISS_OK = False


# ---------------------------
# Helpers
# ---------------------------
def _load_cfg(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _concat_text(obj: Dict) -> str:
    title = obj.get("title") or ""
    desc = obj.get("description") or ""
    pc = obj.get("post_content") or ""
    content = obj.get("content") or ""
    return "\n".join([p for p in (title, desc, pc, content) if p]).strip()


def _load_docs(jsonl_path: str) -> List[Dict]:
    docs: List[Dict] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            docs.append(json.loads(s))
    return docs


# ---------------------------
# TF-IDF
# ---------------------------
def _tfidf_exists(outdir: Path) -> bool:
    return all(
        (outdir / p).exists()
        for p in ("tfidf_index.npz", "tfidf_vocab.json", "tfidf_id_map.json")
    )


def _build_tfidf(cfg: Dict, docs: List[Dict], outdir: Path, force: bool) -> None:
    if _tfidf_exists(outdir) and not force:
        print(f"[index.build] TF-IDF already present at {outdir} — skipping (force_rebuild:true to rebuild).")
        return

    tf = cfg.get("tfidf", {})
    vec = TfidfVectorizer(
        lowercase=True,
        ngram_range=(int(tf.get("ngram_min", 1)), int(tf.get("ngram_max", 1))),
        token_pattern=r"(?u)\b\w\w+\b",
        max_features=int(tf.get("max_features", 80000)),
        min_df=int(tf.get("min_df", 2)),
        dtype=np.float32,
    )

    texts = [_concat_text(d) for d in docs]
    X = vec.fit_transform(texts).tocsr().astype(np.float32)
    X = normalize(X, norm="l2", axis=1, copy=False)

    vocab = vec.vocabulary_                  # dict[str, int-like]
    idf_attr = getattr(vec, "idf_", None)    # np.ndarray or None
    id_list = [str(d["id"]) for d in docs]

    # JSON-safe casts (avoid numpy types)
    vocab_out = {str(k): int(v) for k, v in vocab.items()}
    idf_out = idf_attr.astype(float).tolist() if idf_attr is not None else []
    id_map_out = [str(i) for i in id_list]

    outdir.mkdir(parents=True, exist_ok=True)
    sparse.save_npz(outdir / "tfidf_index.npz", X)
    with open(outdir / "tfidf_vocab.json", "w", encoding="utf-8") as f:
        json.dump({"vocabulary": vocab_out, "idf": idf_out}, f, ensure_ascii=True)
    with open(outdir / "tfidf_id_map.json", "w", encoding="utf-8") as f:
        json.dump(id_map_out, f, ensure_ascii=True)

    print(f"[index.build] TF-IDF saved to {outdir} (rows={X.shape[0]}, cols={X.shape[1]})")


# ---------------------------
# Whoosh (BM25)
# ---------------------------
def _whoosh_exists(wdir: Path) -> bool:
    try:
        return windex.exists_in(str(wdir))
    except Exception:
        return False


def _build_whoosh(cfg: Dict, docs: List[Dict], force: bool) -> None:
    if not WHOOSH_OK:
        print("[index.build] Whoosh not installed — skipping BM25.")
        return

    wdir = Path(cfg.get("whoosh", {}).get("index_dir", "./artifacts/whoosh_index"))
    wdir.mkdir(parents=True, exist_ok=True)

    if _whoosh_exists(wdir) and not force:
        print(f"[index.build] Whoosh index already present at {wdir} — skipping (force_rebuild:true to rebuild).")
        return

    # Recreate dir if forcing
    if _whoosh_exists(wdir) and force:
        shutil.rmtree(wdir)
        wdir.mkdir(parents=True, exist_ok=True)

    schema = Schema(id=ID(stored=True, unique=True), text=TEXT(stored=False, analyzer=StandardAnalyzer()))
    ix = windex.create_in(str(wdir), schema)
    writer = ix.writer(limitmb=256, procs=1, multisegment=True)
    for d in tqdm(docs, desc="Whoosh (BM25)"):
        writer.add_document(id=str(d["id"]), text=_concat_text(d))
    writer.commit()
    print(f"[index.build] Whoosh BM25 index at {wdir}")


# ---------------------------
# FAISS (dense)
# ---------------------------
def _faiss_exists(outdir: Path) -> bool:
    return (outdir / "faiss.index").exists()


def _build_faiss(cfg: Dict, npz_path: Path | None, outdir: Path, force: bool) -> None:
    if not FAISS_OK:
        print("[index.build] FAISS not installed — skipping dense.")
        return
    if not npz_path or not npz_path.exists():
        print("[index.build] Embeddings NPZ not found — skipping dense.")
        return
    if _faiss_exists(outdir) and not force:
        print(f"[index.build] FAISS index already present at {outdir} — skipping (force_rebuild:true to rebuild).")
        return

    data = np.load(npz_path, allow_pickle=True)
    emb = data["embeddings"].astype(np.float32)
    norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
    emb = emb / norms  # cosine/IP
    d = emb.shape[1]

    faiss_type = cfg.get("index", {}).get("faiss_type", "flat").lower()
    if faiss_type == "flat":
        index = faiss.IndexFlatIP(d)  # instant build
    else:
        index = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = int(cfg.get("index", {}).get("hnsw_ef_construction", 100))
        index.hnsw.efSearch = int(cfg.get("index", {}).get("hnsw_ef_search", 40))

    index.add(emb)
    out = outdir / "faiss.index"
    faiss.write_index(index, str(out))
    print(f"[index.build] FAISS {faiss_type.upper()} index built at {out} (vectors={emb.shape[0]})")


# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser(description="Build TF-IDF, Whoosh (BM25), and optional FAISS indexes.")
    ap.add_argument("--config", type=str, default="config.yaml")
    ap.add_argument("--api_key", type=str, default=None, help="Accepted but unused here.")
    args = ap.parse_args()

    cfg = _load_cfg(args.config)
    docs_path = cfg["data"]["documents_path"]
    outdir = Path(cfg.get("index", {}).get("output_dir", "./artifacts/index"))
    force = bool(cfg.get("index", {}).get("force_rebuild", False))

    build_sparse = bool(cfg.get("index", {}).get("build_sparse", True))
    build_whoosh = bool(cfg.get("index", {}).get("build_whoosh", True))
    build_dense = bool(cfg.get("index", {}).get("build_dense", False))
    emb_path = Path(cfg["data"].get("embeddings_path", "")) if cfg["data"].get("embeddings_path") else None

    print(f"[index.build] Documents: {docs_path}")
    docs = _load_docs(docs_path)
    print(f"[index.build] Loaded {len(docs)} docs.")

    if build_sparse:
        _build_tfidf(cfg, docs, outdir, force)
    if build_whoosh:
        _build_whoosh(cfg, docs, force)
    if build_dense:
        _build_faiss(cfg, emb_path, outdir, force)

    print("[index.build] Done.")


if __name__ == "__main__":
    main()
