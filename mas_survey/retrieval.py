# mas_survey/retrieval.py
"""
Retrieval utilities:
- Query building (base + per-option)
- Sparse retrieval: BM25 (Whoosh) and/or TF-IDF
- Optional dense retrieval: FAISS centroid PRF from BM25 seeds
- RRF fusion and TF-IDF-based near-dup removal
- Robust fallbacks so we always return candidates

All functions are CPU-only and respect the config toggles.

Expected config keys (with defaults):
retrieval:
  sparse_backend: "bm25"       # "bm25" | "tfidf" | "both"
  top_k_sparse: 800
  use_dense: false
  top_k_dense: 800
  rrf_k: 60
whoosh:
  index_dir: "./artifacts/whoosh_index"
  hits_per_query: 1200
index:
  output_dir: "./artifacts/index"
data:
  embeddings_path: "./data/id_to_embedding.npz"  # for FAISS centroid PRF
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict, Counter
import json
import numpy as np
from scipy import sparse
from sklearn.preprocessing import normalize

# ---------- Optional deps ----------
# Whoosh for BM25
try:
    from whoosh import index as windex
    from whoosh.qparser import QueryParser, OrGroup
    from whoosh import scoring as wscoring
    from whoosh.query import Every
    WHOOSH_OK = True
except Exception:
    WHOOSH_OK = False

# FAISS for dense search
FAISS_OK = False
try:
    import faiss  # type: ignore
    FAISS_OK = True
except Exception:
    FAISS_OK = False


# ---------- Data structures ----------

@dataclass
class RetrievalResources:
    # TF-IDF
    X_tfidf: sparse.csr_matrix
    vocab: Dict[str, int]
    idf: np.ndarray
    id_map: List[str]
    id_to_row: Dict[str, int]
    # Whoosh
    whoosh_dir: Path | None = None
    # Dense (optional)
    faiss_index_path: Path | None = None
    dense_id_map: List[str] | None = None
    dense_embeddings: np.ndarray | None = None   # only if centroid PRF needed


# ---------- Helpers ----------

def build_queries(q_text: str, options: List[str]) -> List[str]:
    # Base + per-option expansion (simple, deterministic)
    opt_qs = [f"{q_text} {opt}" for opt in options] if options else []
    return [q_text] + opt_qs

def rrf_fuse(rank_lists: List[List[str]], k: int = 60) -> Dict[str, float]:
    scores = defaultdict(float)
    for rl in rank_lists:
        for r, did in enumerate(rl):
            scores[did] += 1.0 / (k + r + 1)
    return scores

def _tfidf_transform_short(texts: List[str], vocab: Dict[str, int], idf: np.ndarray) -> sparse.csr_matrix:
    # Minimal TF-IDF transform using the fitted vocab/idf (from build time)
    rows, cols, data = [], [], []
    import re
    for r, t in enumerate(texts):
        toks = [tok.lower() for tok in re.findall(r"\b\w\w+\b", (t or ""))]
        if not toks:
            continue
        tf = Counter(toks)
        for tok, cnt in tf.items():
            j = vocab.get(tok)
            if j is None:
                continue
            rows.append(r); cols.append(j); data.append(float(cnt))
    if not rows:
        return sparse.csr_matrix((len(texts), len(vocab)), dtype=np.float32)
    M = sparse.csr_matrix((np.array(data, dtype=np.float32), (np.array(rows), np.array(cols))),
                          shape=(len(texts), len(vocab)), dtype=np.float32)
    if idf is not None and idf.size:
        # defensive: align by index if available; else treat idf=1
        idf_vec = np.ones((M.shape[1],), dtype=np.float32)
        n = min(len(idf), len(idf_vec))
        idf_vec[:n] = idf[:n]
        M = M @ sparse.diags(idf_vec, 0, dtype=np.float32)
    return normalize(M, norm="l2", axis=1, copy=False)

def _cosine_rows(A: sparse.csr_matrix, b: sparse.csr_matrix) -> np.ndarray:
    # A: (N,V) rows normalized; b: (1,V) normalized
    return (A @ b.T).toarray().ravel()

def _dedup_by_tfidf(ids_in_order: List[str], X_rows: sparse.csr_matrix, thresh: float = 0.96) -> List[int]:
    # Greedy near-dup removal by cosine >= thresh
    n = len(ids_in_order)
    keep = []
    removed = np.zeros(n, dtype=bool)
    for i in range(n):
        if removed[i]:
            continue
        keep.append(i)
        xi = X_rows[i]
        sims = (X_rows @ xi.T).toarray().ravel()
        removed = np.logical_or(removed, sims >= thresh)
    return keep


# ---------- Sparse retrieval ----------

def bm25_rank(whoosh_dir: Path, queries: List[str], hits_per_query: int, rrf_k: int) -> List[str]:
    if not WHOOSH_OK:
        return []
    ix = windex.open_dir(str(whoosh_dir))
    qp = QueryParser("text", schema=ix.schema, group=OrGroup)
    rank_lists: List[List[str]] = []
    with ix.searcher(weighting=wscoring.BM25F()) as searcher:
        for qtext in queries:
            ranked: List[str] = []
            try:
                q = qp.parse(qtext)
                hits = searcher.search(q, limit=hits_per_query)
                ranked = [hit["id"] for hit in hits]
            except Exception:
                ranked = []
            if not ranked:
                # fallback: global top docs
                hits = searcher.search(Every(), limit=hits_per_query)
                ranked = [hit["id"] for hit in hits]
            rank_lists.append(ranked)
        if all(len(rl) == 0 for rl in rank_lists):
            hits = searcher.search(Every(), limit=hits_per_query)
            rank_lists = [[hit["id"] for hit in hits]]
    fused = rrf_fuse(rank_lists, k=rrf_k)
    return [did for did, _ in sorted(fused.items(), key=lambda x: x[1], reverse=True)]

def tfidf_rank(queries: List[str], X: sparse.csr_matrix, vocab: Dict[str,int], idf: np.ndarray, id_map: List[str], rrf_k: int) -> List[str]:
    # Rank all docs by cosine to each query; fuse via RRF
    rank_lists: List[List[str]] = []
    for q in queries:
        qv = _tfidf_transform_short([q], vocab, idf)  # (1,V)
        sims = _cosine_rows(X, qv)                    # (N,)
        order = np.argsort(-sims)
        rank_lists.append([id_map[i] for i in order.tolist()])
    fused = rrf_fuse(rank_lists, k=rrf_k)
    return [did for did, _ in sorted(fused.items(), key=lambda x: x[1], reverse=True)]


# ---------- Dense retrieval (optional) ----------

def load_faiss_and_dense(npz_path: Path, faiss_index_path: Path) -> Tuple[faiss.Index, np.ndarray, List[str]]:
    # Load embeddings NPZ (ids, embeddings) and FAISS index
    data = np.load(npz_path, allow_pickle=True)
    ids = list(map(str, data["ids"].tolist()))
    emb = data["embeddings"].astype(np.float32)
    # Normalize for cosine/IP
    norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
    emb = emb / norms
    index = faiss.read_index(str(faiss_index_path))
    return index, emb, ids

def faiss_centroid_prf(
    seeds: List[str],
    id_to_dense_row: Dict[str, int],
    dense_emb: np.ndarray,
    faiss_index: "faiss.Index",
    k: int,
) -> List[str]:
    # Average seed embeddings to form a query; search FAISS
    seed_idx = [id_to_dense_row[s] for s in seeds if s in id_to_dense_row]
    if not seed_idx:
        return []
    q = dense_emb[seed_idx].mean(axis=0, keepdims=True).astype(np.float32)
    faiss.normalize_L2(q)  # ensure norm=1
    _, I = faiss_index.search(q, k)
    hits = I[0].tolist()
    # Map back via ids order
    # We need ids list to invert; pass caller-side
    return hits


# ---------- Entry point ----------

def retrieve_candidates(
    cfg: Dict,
    q_text: str,
    options: List[str],
    res: RetrievalResources,
) -> Tuple[List[str], List[int], sparse.csr_matrix, Dict[str, str]]:
    """
    Returns:
      cand_ids: ordered list of document IDs after fusion and dedup
      cand_rows: their TF-IDF row indices
      X_rows: TF-IDF rows (CSR)
      dbg: small debug dict
    """
    top_k_sparse = int(cfg.get("retrieval", {}).get("top_k_sparse", 800))
    rrf_k = int(cfg.get("retrieval", {}).get("rrf_k", 60))
    sparse_backend = cfg.get("retrieval", {}).get("sparse_backend", "bm25").lower()
    use_dense = bool(cfg.get("retrieval", {}).get("use_dense", False))
    top_k_dense = int(cfg.get("retrieval", {}).get("top_k_dense", 800))
    hits_per_query = int(cfg.get("whoosh", {}).get("hits_per_query", 1200))
    dedup_tau = float(cfg.get("filtering", {}).get("dedup_sim_thresh", 0.96))

    queries = build_queries(q_text, options)

    # ----- sparse candidates -----
    sparse_rank: List[str] = []
    if sparse_backend in ("bm25", "both") and WHOOSH_OK and res.whoosh_dir:
        bm25_ids_all = bm25_rank(res.whoosh_dir, queries, hits_per_query, rrf_k)
        sparse_rank = bm25_ids_all
    if sparse_backend in ("tfidf", "both"):
        tfidf_ids_all = tfidf_rank(queries, res.X_tfidf, res.vocab, res.idf, res.id_map, rrf_k)
        if sparse_rank:
            # fuse sparse lists via RRF
            fused = rrf_fuse([sparse_rank, tfidf_ids_all], k=rrf_k)
            sparse_rank = [did for did, _ in sorted(fused.items(), key=lambda x: x[1], reverse=True)]
        else:
            sparse_rank = tfidf_ids_all

    # fallback if still empty
    if not sparse_rank:
        sparse_rank = list(res.id_map)  # global order as last resort

    sparse_rank = sparse_rank[:top_k_sparse]

    # ----- optional dense via FAISS centroid PRF -----
    final_rank: List[str] = list(sparse_rank)
    dbg = {"used_dense": "false", "sparse_backend": sparse_backend}
    if use_dense and FAISS_OK and res.faiss_index_path and res.faiss_index_path.exists() and res.dense_embeddings is not None and res.dense_id_map is not None:
        # Build id->row for dense matrix
        id_to_dense_row = {did: i for i, did in enumerate(res.dense_id_map)}
        # centroid from top-M sparse seeds
        M = min(128, len(sparse_rank))
        seed_ids = sparse_rank[:M]
        index = faiss.read_index(str(res.faiss_index_path))  # small, cheap
        # Query FAISS
        # Translate returned rows -> ids using dense_id_map
        # faiss_centroid_prf returns indices in the FAISS space (row ids)
        # Implement inline to also map to IDs:
        seed_idx = [id_to_dense_row[s] for s in seed_ids if s in id_to_dense_row]
        if seed_idx:
            q = res.dense_embeddings[seed_idx].mean(axis=0, keepdims=True).astype(np.float32)
            faiss.normalize_L2(q)
            _, I = index.search(q, top_k_dense)
            dense_ids = [res.dense_id_map[j] for j in I[0] if 0 <= j < len(res.dense_id_map)]
            # fuse sparse + dense via RRF
            fused = rrf_fuse([final_rank, dense_ids], k=rrf_k)
            final_rank = [did for did, _ in sorted(fused.items(), key=lambda x: x[1], reverse=True)]
            dbg["used_dense"] = "true"

    # ----- dedup by TF-IDF cosine -----
    cand_ids = final_rank
    cand_rows = [res.id_to_row[did] for did in cand_ids if did in res.id_to_row]
    X_rows = res.X_tfidf[cand_rows] if cand_rows else sparse.csr_matrix((0, res.X_tfidf.shape[1]), dtype=np.float32)
    keep_idx = _dedup_by_tfidf([cand_ids[i] for i in range(len(cand_rows))], X_rows, thresh=dedup_tau) if cand_rows else []
    if keep_idx:
        cand_rows = [cand_rows[i] for i in keep_idx]
        cand_ids = [cand_ids[i] for i in keep_idx]
        X_rows = res.X_tfidf[cand_rows]

    # Safety: never return empty
    if not cand_ids:
        cand_ids = list(res.id_map[:top_k_sparse])
        cand_rows = [res.id_to_row[did] for did in cand_ids if did in res.id_to_row]
        X_rows = res.X_tfidf[cand_rows] if cand_rows else sparse.csr_matrix((0, res.X_tfidf.shape[1]), dtype=np.float32)

    return cand_ids, cand_rows, X_rows, dbg
