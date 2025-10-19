# mas_survey/retrieval.py
# Python 3.9 compatible
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

# ---- Optional Whoosh imports
try:
    from whoosh.index import open_dir as whoosh_open_dir
    from whoosh.qparser import MultifieldParser, OrGroup
    HAVE_WHOOSH = True
except Exception:
    HAVE_WHOOSH = False


# -------------------------------------------------------------------
# Index container
# -------------------------------------------------------------------
class Indices:
    def __init__(
        self,
        *,
        tfidf_X: csr_matrix,
        vocab: Dict[str, int],
        row_id_map: List[str],
        whoosh_dir: Optional[Path] = None,
    ):
        self.tfidf_X = tfidf_X
        self.vocab = vocab
        self.row_id_map = row_id_map                 # row -> doc_id
        self.id2row = {d: i for i, d in enumerate(row_id_map)}
        self.whoosh_dir = whoosh_dir if whoosh_dir and whoosh_dir.exists() else None

        self._windex = None
        if HAVE_WHOOSH and self.whoosh_dir:
            try:
                self._windex = whoosh_open_dir(str(self.whoosh_dir))
            except Exception:
                self._windex = None


# -------------------------------------------------------------------
# Load TF-IDF artifacts produced by index.build
# Accepts both id_map.json and tfidf_id_map.json
# -------------------------------------------------------------------
def load_tfidf(outdir: Path) -> Tuple[csr_matrix, Dict[str, int], List[str]]:
    m = np.load(str(outdir / "tfidf_index.npz"))
    X = csr_matrix((m["data"], m["indices"], m["indptr"]), shape=tuple(m["shape"]))

    meta = json.loads((outdir / "tfidf_vocab.json").read_text(encoding="utf-8"))
    vocab = {str(k): int(v) for k, v in meta["vocabulary"].items()}

    id_map_path = outdir / "id_map.json"
    if not id_map_path.exists():
        alt = outdir / "tfidf_id_map.json"
        if alt.exists():
            id_map_path = alt
        else:
            raise FileNotFoundError(f"Missing id map: {id_map_path} or {alt}")

    row_id_map = json.loads(id_map_path.read_text(encoding="utf-8"))
    return X, vocab, row_id_map


# -------------------------------------------------------------------
# Query prep helpers
# -------------------------------------------------------------------
_SPLIT_RE = re.compile(r"\s*//\s*")
_PUNCT_RE = re.compile(r"[^\w\s]+")

def _clean(s: str) -> str:
    s = s or ""
    # Take the clause before ' // ' to avoid super long, noisy tails
    s = _SPLIT_RE.split(s, maxsplit=1)[0].lower().strip()
    s = _PUNCT_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _mk_queries(question: str, options: List[str], fanout: bool) -> List[str]:
    root = _clean(question)
    qs = [root]
    if fanout:
        for o in options:
            oo = _clean(o)
            if oo and oo not in qs:
                qs.append(f"{root} {oo}")
    # tiny domain heuristic to broaden some topics
    if "immigr" in root:
        qs.append(root + " united states america US values opinions")
    if "priority" in root and "immigr" in root:
        qs.append(root + " policy support should priority immigration")
    return qs


# -------------------------------------------------------------------
# Whoosh BM25 search (optional)
# -------------------------------------------------------------------
def _whoosh_search(idx: Indices, queries: List[str], hits_per_query: int) -> List[Tuple[str, float]]:
    if not (HAVE_WHOOSH and idx._windex):
        return []
    out: List[Tuple[str, float]] = []
    with idx._windex.searcher() as searcher:
        parser = MultifieldParser(
            ["title", "description", "post_content", "content"],
            schema=idx._windex.schema,
            group=OrGroup,
        )
        for q in queries:
            try:
                qobj = parser.parse(q)
                res = searcher.search(qobj, limit=hits_per_query)
                for hit in res:
                    did = hit.get("id")
                    if did:
                        out.append((did, float(hit.score)))
            except Exception:
                continue
    return out


# -------------------------------------------------------------------
# TF-IDF cosine ranking (always available)
# -------------------------------------------------------------------
def _tfidf_rank(idx: Indices, queries: List[str], top_k: int) -> List[Tuple[str, float]]:
    if idx.tfidf_X.shape[0] == 0:
        return []
    def qvec(q: str) -> csr_matrix:
        toks = q.split()
        cols = [idx.vocab[t] for t in toks if t in idx.vocab]
        if not cols:
            return csr_matrix((1, idx.tfidf_X.shape[1]), dtype=np.float32)
        data = np.ones(len(cols), dtype=np.float32)
        rows = np.zeros(len(cols), dtype=np.int32)
        return csr_matrix((data, (rows, np.array(cols))), shape=(1, idx.tfidf_X.shape[1]), dtype=np.float32)

    scores = np.zeros(idx.tfidf_X.shape[0], dtype=np.float32)
    for q in queries:
        v = qvec(_clean(q))
        if v.nnz == 0:
            continue
        cs = cosine_similarity(v, idx.tfidf_X, dense_output=False)  # 1 x N
        scores += np.asarray(cs.todense()).ravel().astype(np.float32)

    if float(scores.max(initial=0.0)) <= 0.0:
        return []
    top = np.argpartition(-scores, min(top_k, len(scores)-1))[:top_k]
    top = top[np.argsort(-scores[top])]
    return [(idx.row_id_map[i], float(scores[i])) for i in top]


# -------------------------------------------------------------------
# Fusion & de-dup
# -------------------------------------------------------------------
def _rrf_fuse(runs: List[List[Tuple[str, float]]], k: int = 60) -> List[Tuple[str, float]]:
    agg: Dict[str, float] = {}
    for run in runs:
        for rnk, (did, _) in enumerate(run):
            agg[did] = agg.get(did, 0.0) + 1.0 / (k + (rnk + 1))
    return sorted(agg.items(), key=lambda x: -x[1])

def _dedup_by_tfidf(idx: Indices, ranked_ids: List[str], cos_thresh: float, cap: Optional[int]) -> List[str]:
    keep: List[str] = []
    seen_rows: List[int] = []
    for did in ranked_ids:
        row = idx.id2row.get(did)
        if row is None:
            continue
        v = idx.tfidf_X[row]
        dup = False
        for r in seen_rows:
            if cosine_similarity(v, idx.tfidf_X[r])[0, 0] >= cos_thresh:
                dup = True
                break
        if not dup:
            keep.append(did)
            seen_rows.append(row)
            if cap and len(keep) >= cap:
                break
    return keep


# -------------------------------------------------------------------
# Public API: build indices handle and retrieve candidates
# -------------------------------------------------------------------
def build_indices(cfg: dict) -> Indices:
    outdir = Path(cfg["index"]["output_dir"])
    tfidf_X, vocab, row_id_map = load_tfidf(outdir)
    whoosh_dir = Path(cfg.get("whoosh", {}).get("index_dir", "")) if cfg.get("whoosh") else None
    return Indices(tfidf_X=tfidf_X, vocab=vocab, row_id_map=row_id_map, whoosh_dir=whoosh_dir)

def retrieve_candidates(question: str, options: List[str], cfg: dict, idx: Indices) -> List[str]:
    fanout = bool(cfg.get("retrieval", {}).get("fanout", True))
    queries = _mk_queries(question, options, fanout)

    hits_per_query = int(cfg.get("whoosh", {}).get("hits_per_query", 600))
    top_k_sparse  = int(cfg.get("retrieval", {}).get("top_k_sparse", 600))
    min_cands     = int(cfg.get("retrieval", {}).get("min_candidates", 160))
    rrf_k         = int(cfg.get("retrieval", {}).get("rrf_k", 60))
    dedup_t       = float(cfg.get("filtering", {}).get("dedup_sim_thresh", 0.96))

    runs: List[List[Tuple[str, float]]] = []

    # BM25 (if available)
    bm25 = _whoosh_search(idx, queries, hits_per_query) if idx._windex is not None else []
    if bm25:
        runs.append(bm25[:top_k_sparse])

    # TF-IDF cosine (always)
    tfidf = _tfidf_rank(idx, queries, top_k_sparse)
    if tfidf:
        runs.append(tfidf[:top_k_sparse])

    # If nothing at all, return a deterministic slice so downstream never breaks
    if all(len(r) == 0 for r in runs):
        return idx.row_id_map[:min_cands]

    fused = _rrf_fuse(runs, k=rrf_k)
    fused_ids = [did for did, _ in fused]

    deduped = _dedup_by_tfidf(idx, fused_ids, cos_thresh=dedup_t, cap=top_k_sparse)

    # Guarantee minimum by padding with TF-IDF order
    if len(deduped) < min_cands:
        for did, _ in tfidf:
            if did not in deduped:
                deduped.append(did)
                if len(deduped) >= min_cands:
                    break

    # Return a capped but sufficient list
    final_cap = max(min_cands, top_k_sparse)
    return deduped[:final_cap]
