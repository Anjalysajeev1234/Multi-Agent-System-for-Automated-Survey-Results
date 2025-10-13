# index/build.py
"""
Build indices from the corpus.

CLI:
    python -m index.build --config config.yaml --api_key <YOUR_KEY or DUMMY>

Controlled by config.index.*:
- Sparse TF-IDF matrix  -> artifacts/index/tfidf_index.npz
- TF-IDF vocab/params   -> artifacts/index/tfidf_vocab.json
- Row -> doc_id mapping -> artifacts/index/tfidf_id_map.json
- (optional) Whoosh BM25-> artifacts/whoosh_index/
- (optional) FAISS HNSW  -> artifacts/index/faiss.index
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import yaml
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

# ---------- FAISS (optional) ----------
FAISS_OK = False
try:
    import faiss  # type: ignore
    FAISS_OK = True
except Exception:
    FAISS_OK = False

# ---------- Whoosh (optional) ----------
try:
    from whoosh import index as windex
    from whoosh.fields import Schema, ID, TEXT
    from whoosh.analysis import StandardAnalyzer
    WHOOSH_OK = True
except Exception:
    WHOOSH_OK = False


# --------------------- utils ---------------------

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


def _save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=True, indent=2), encoding="utf-8")


# -------------------- TF-IDF ---------------------

def build_tfidf(docs: Dict[str, Dict], out_dir: Path) -> None:
    print(f"[index.build] Building TF-IDF …")
    doc_ids: List[str] = list(docs.keys())
    texts: List[str] = [_concat_text(docs[i]) for i in doc_ids]

    vec = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        token_pattern=r"(?u)\b\w\w+\b",
        max_features=300000,
        dtype=np.float32,
    )
    X = vec.fit_transform(texts).astype(np.float32)
    X = normalize(X, norm="l2", axis=1, copy=False)  # cosine = dot

    out_dir.mkdir(parents=True, exist_ok=True)
    sparse.save_npz(out_dir / "tfidf_index.npz", X)

    vocab_payload = {
        "lowercase": vec.lowercase,
        "ngram_range": list(vec.ngram_range),
        "token_pattern": vec.token_pattern,
        "max_features": vec.max_features,
        "vocabulary": {k: int(v) for k, v in (vec.vocabulary_ or {}).items()},
        "idf": vec.idf_.tolist() if hasattr(vec, "idf_") else None,
    }
    _save_json(out_dir / "tfidf_vocab.json", vocab_payload)
    _save_json(out_dir / "tfidf_id_map.json", doc_ids)
    print(f"[index.build] TF-IDF saved: {X.shape[0]} docs, {X.shape[1]} terms.")


# -------------------- Whoosh BM25 ---------------------

def build_whoosh_index(docs: Dict[str, Dict], whoosh_dir: Path) -> None:
    if not WHOOSH_OK:
        print("[index.build] Whoosh not installed; skipping BM25 index.")
        return
    whoosh_dir.mkdir(parents=True, exist_ok=True)
    schema = Schema(
        id=ID(stored=True, unique=True),
        text=TEXT(stored=False, analyzer=StandardAnalyzer(), phrase=True),
    )
    if windex.exists_in(str(whoosh_dir)):
        ix = windex.open_dir(str(whoosh_dir))
    else:
        ix = windex.create_in(str(whoosh_dir), schema)

    writer = ix.writer(limitmb=512, procs=1, multisegment=True)
    n = 0
    for did, obj in docs.items():
        text = _concat_text(obj)
        if not text:
            continue
        writer.update_document(id=str(did), text=text)
        n += 1
        if n % 10000 == 0:
            print(f"[index.build] Whoosh indexed {n} docs …")
    writer.commit()
    print(f"[index.build] Whoosh BM25 index built with {n} docs at {whoosh_dir}")


# --------------------- FAISS ---------------------

def build_faiss_from_npz(npz_path: Path, out_dir: Path) -> None:
    if not FAISS_OK:
        print("[index.build] FAISS not available (faiss-cpu not installed). Skipping dense index.")
        return
    if not npz_path.exists():
        print(f"[index.build] Embeddings NPZ not found at {npz_path}. Skipping dense index.")
        return

    print(f"[index.build] Loading embeddings from {npz_path} …")
    data = np.load(npz_path, allow_pickle=True)
    ids = list(map(str, data["ids"].tolist()))
    emb = data["embeddings"].astype(np.float32)

    norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
    emb = emb / norms

    d = emb.shape[1]
    index = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = 200
    index.hnsw.efSearch = 64

    print(f"[index.build] FAISS HNSW index building for {len(ids)} vectors (d={d}) …")
    index.add(emb)

    out_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out_dir / "faiss.index"))
    _save_json(out_dir / "dense_id_map.json", ids)
    print("[index.build] Dense index saved.")


# ---------------------- CLI ----------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build sparse TF-IDF, optional Whoosh BM25, optional FAISS.")
    p.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    p.add_argument("--api_key", type=str, default=None, help="Accepted but unused.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = _load_cfg(args.config)

    docs_path = Path(cfg["data"]["documents_path"])
    embeddings_npz = cfg["data"].get("embeddings_path", "")
    emb_path = Path(embeddings_npz) if embeddings_npz else Path("")

    out_dir = Path(cfg.get("index", {}).get("output_dir", "./artifacts/index"))
    build_sparse = bool(cfg.get("index", {}).get("build_sparse", True))
    build_whoosh = bool(cfg.get("index", {}).get("build_whoosh", True))
    build_dense = bool(cfg.get("index", {}).get("build_dense", True))

    print(f"[index.build] Documents: {docs_path}")
    docs = _load_documents(str(docs_path))
    print(f"[index.build] Loaded {len(docs)} docs.")

    if build_sparse:
        build_tfidf(docs, out_dir)
    else:
        print("[index.build] Skipping TF-IDF (build_sparse=false).")

    if build_whoosh:
        whoosh_dir = Path(cfg.get("whoosh", {}).get("index_dir", "./artifacts/whoosh_index"))
        build_whoosh_index(docs, whoosh_dir)
    else:
        print("[index.build] Skipping Whoosh (build_whoosh=false).")

    if build_dense:
        build_faiss_from_npz(emb_path, out_dir)
    else:
        print("[index.build] Skipping FAISS (build_dense=false).")

    print("[index.build] Done.")


if __name__ == "__main__":
    main()
