# mas_survey/retrieval.py
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

WHOOSH_OK, FAISS_OK = True, True
try:
    from whoosh import index as windex
    from whoosh.qparser import QueryParser
except Exception:
    WHOOSH_OK = False
try:
    import faiss  # type: ignore
except Exception:
    FAISS_OK = False


@dataclass
class Indices:
    tfidf_X: sparse.csr_matrix
    vocab: Dict[str, int]
    row_id_map: List[str]
    whoosh_dir: Optional[Path]
    faiss_index_path: Optional[Path]


def load_tfidf(index_dir: Path) -> Tuple[sparse.csr_matrix, Dict[str, int], List[str]]:
    X = sparse.load_npz(index_dir / "tfidf_index.npz").tocsr().astype(np.float32)
    meta = json.loads((index_dir / "tfidf_vocab.json").read_text(encoding="utf-8"))
    vocab = {k: int(v) for k, v in meta["vocabulary"].items()}
    row_id_map = json.loads((index_dir / "tfidf_id_map.json").read_text(encoding="utf-8"))
    return X, vocab, row_id_map


def build_query_vector(text: str, vocab: Dict[str, int]) -> sparse.csr_matrix:
    tv = TfidfVectorizer(
        lowercase=True, ngram_range=(1,1),
        token_pattern=r"(?u)\b\w\w+\b",
        vocabulary=vocab,
        dtype=np.float32, use_idf=False, norm=None
    )
    qX = tv.fit_transform([text]).astype(np.float32)
    qX = normalize(qX, norm="l2", axis=1, copy=False)
    return qX


def cosine_topk(q: sparse.csr_matrix, X: sparse.csr_matrix, k: int) -> Tuple[np.ndarray, np.ndarray]:
    sims = (q @ X.T).toarray().ravel()
    if k >= sims.shape[0]:
        idx = np.argsort(-sims)
    else:
        idx = np.argpartition(-sims, k)[:k]
        idx = idx[np.argsort(-sims[idx])]
    return idx, sims[idx]


def whoosh_search_list(wdir: Path, texts: List[str], hits: int) -> List[List[str]]:
    if not WHOOSH_OK or not windex.exists_in(str(wdir)):
        return [[] for _ in texts]
    ix = windex.open_dir(str(wdir))
    qp = QueryParser("text", schema=ix.schema)
    out: List[List[str]] = []
    with ix.searcher() as s:
        for t in texts:
            q = qp.parse(t)
            out.append([hit["id"] for hit in s.search(q, limit=hits)])
    return out


def load_faiss(path: Path):
    if not FAISS_OK or not path or not path.exists():
        return None
    return faiss.read_index(str(path))


def rrf_fuse(ranklists: List[List[str]], k: int = 60, limit: int = 1000) -> List[str]:
    scores: Dict[str, float] = {}
    for rl in ranklists:
        for r, did in enumerate(rl, start=1):
            scores[did] = scores.get(did, 0.0) + 1.0 / (k + r)
    out = sorted(scores.items(), key=lambda x: -x[1])
    return [did for did, _ in out[:limit]]


def dedup_by_tfidf_cosine(candidates: List[str], id2row: Dict[str, int], X: sparse.csr_matrix, thresh: float) -> List[str]:
    keep: List[str] = []
    rows: List[int] = []
    for did in candidates:
        r = id2row.get(did)
        if r is None:
            continue
        xr = X[r]
        ok = True
        for kept_r in rows:
            sim = (xr @ X[kept_r].T).A[0, 0]
            if sim >= thresh:
                ok = False
                break
        if ok:
            keep.append(did)
            rows.append(r)
    return keep


def retrieve_candidates(
    question: str,
    options_for_q: List[str],
    cfg: dict,
    idx: Indices,
    docid2text: Dict[str, str],
) -> List[str]:
    """
    Retrieval with:
      - BM25 fan-out: [question] + [question + option_i] for each option
      - RRF fusion
      - TF-IDF cosine dedup
    """
    hits = int(cfg["whoosh"]["hits_per_query"])
    backend = cfg["retrieval"]["sparse_backend"]
    rrf_k = int(cfg["retrieval"]["rrf_k"])
    min_cand = int(cfg["retrieval"]["min_candidates"])

    # Build ranklists
    ranklists: List[List[str]] = []

    # A) sparse BM25 (preferred)
    if backend in ("bm25", "both") and idx.whoosh_dir:
        q_texts = [question] + [f"{question} {opt}" for opt in options_for_q]
        ranklists = whoosh_search_list(idx.whoosh_dir, q_texts, hits)

    # B) fallback TF-IDF if BM25 unavailable
    if not ranklists or all(len(rl) == 0 for rl in ranklists):
        if backend in ("tfidf", "both"):
            qX = build_query_vector(question, idx.vocab)
            rows, _ = cosine_topk(qX, idx.tfidf_X, int(cfg["retrieval"]["top_k_sparse"]))
            ranklists = [[idx.row_id_map[i] for i in rows]]

    # Fuse
    fused = rrf_fuse(ranklists, k=rrf_k, limit=max(min_cand, int(cfg["retrieval"]["top_k_sparse"])))

    # Dedup
    id2row = {d: i for i, d in enumerate(idx.row_id_map)}
    deduped = dedup_by_tfidf_cosine(fused, id2row, idx.tfidf_X, float(cfg["filtering"]["dedup_sim_thresh"]))
    return deduped
