# index/build.py
"""
Builds sparse TF-IDF and (optionally) FAISS HNSW indexes.
CPU-only, Python 3.9.

Writes into config.index.output_dir:
  - tfidf_index.npz
  - tfidf_vocab.json
  - tfidf_id_map.json
  - (optional) faiss.index
  - (optional) dense_id_map.json
"""
import argparse, json, os
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np, yaml
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

FAISS_OK = False
try:
    import faiss  # type: ignore
    FAISS_OK = True
except Exception:
    FAISS_OK = False


def _load_cfg(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _load_documents(jsonl_path: str) -> Tuple[List[str], List[str]]:
    ids, texts = [], []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            obj = json.loads(line)
            doc_id = obj.get("id")
            title = obj.get("title") or ""
            description = obj.get("description") or ""
            post_content = obj.get("post_content") or ""
            content = obj.get("content") or ""
            text = "\n".join([p for p in [title, description, post_content, content] if p]).strip()
            if doc_id and text:
                ids.append(doc_id); texts.append(text)
    return ids, texts

def _build_tfidf(texts: List[str], max_features: int = 200_000):
    vec = TfidfVectorizer(lowercase=True, ngram_range=(1,2),
                          max_features=max_features, dtype=np.float32,
                          token_pattern=r"(?u)\b\w+\b")
    X = vec.fit_transform(texts)
    X = normalize(X, norm="l2", axis=1, copy=False)
    return vec, X

def _save_tfidf(out_dir: Path, vec: TfidfVectorizer, X: sparse.csr_matrix, id_map: List[str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    sparse.save_npz(out_dir / "tfidf_index.npz", X)
    payload = {
        "vocabulary": {k:int(v) for k,v in vec.vocabulary_.items()},
        "idf": vec.idf_.tolist() if hasattr(vec, "idf_") else None,
        "ngram_range": vec.ngram_range,
        "token_pattern": vec.token_pattern,
        "lowercase": vec.lowercase,
        "max_features": vec.max_features,
    }
    (out_dir / "tfidf_vocab.json").write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")
    (out_dir / "tfidf_id_map.json").write_text(json.dumps(id_map, ensure_ascii=True), encoding="utf-8")

def _maybe_build_faiss(npz_path: str, out_dir: Path) -> bool:
    if not FAISS_OK:
        print("[index.build] faiss-cpu not installed; skipping dense index."); return False
    if not npz_path or not os.path.exists(npz_path):
        print(f"[index.build] embeddings NPZ not found at {npz_path}; skipping dense index."); return False
    data = np.load(npz_path, allow_pickle=True)
    ids = data["ids"]
    emb = data["embeddings"].astype(np.float32)
    emb /= (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
    d = emb.shape[1]
    index = faiss.IndexHNSWFlat(d, 32)
    index.hnsw.efConstruction = 80
    index.add(emb)
    out_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out_dir / "faiss.index"))
    (out_dir / "dense_id_map.json").write_text(json.dumps(list(map(str, ids.tolist())), ensure_ascii=True), encoding="utf-8")
    print(f"[index.build] FAISS HNSW index built for {emb.shape[0]} vectors.")
    return True

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build sparse/dense indexes")
    p.add_argument("--config", type=str, default="config.yaml")
    p.add_argument("--api_key", type=str, default=None)  # accepted but unused
    return p.parse_args()

def main() -> None:
    args = parse_args()
    cfg = _load_cfg(args.config)
    docs_path = cfg["data"]["documents_path"]
    out_dir = Path(cfg.get("index", {}).get("output_dir", "./artifacts/index"))
    npz_path = cfg["data"].get("embeddings_path", "")
    print(f"[index.build] Documents: {docs_path}")
    ids, texts = _load_documents(docs_path)
    if not ids: raise SystemExit("[index.build] ERROR: No documents loaded.")
    print(f"[index.build] Loaded {len(ids)} docs. Building TF-IDF …")
    vec, X = _build_tfidf(texts)
    _save_tfidf(out_dir, vec, X, ids)
    built_dense = _maybe_build_faiss(npz_path, out_dir)
    print("[index.build] Dense index saved." if built_dense else "[index.build] Dense index not built (optional).")
    print("[index.build] Done.")

if __name__ == "__main__":
    main()
