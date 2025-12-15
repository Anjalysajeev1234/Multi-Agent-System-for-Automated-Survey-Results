# mas_survey/aggregate.py
from __future__ import annotations
from typing import Dict, List, Tuple
import math
import numpy as np

def _rank_weight(rank_index: int) -> float:
    """
    Rank-aware weight: higher weight for earlier ranks.
    rank_index is 0-based.
    """
    # 1 / log2(2 + rank)
    return 1.0 / math.log2(2.0 + float(rank_index))

def _softmax_temp(x: np.ndarray, temperature: float) -> np.ndarray:
    t = max(1e-6, float(temperature))
    z = (x / t)
    z = z - np.max(z)  # numerical stability
    e = np.exp(z)
    s = e.sum()
    return e / (s if s > 0 else 1.0)

def aggregate_votes(
    *,
    options: List[str],
    votes: Dict[str, Tuple[str, float]],   # doc_id -> (option, confidence)
    rank_ids: List[str],                   # ranked doc ids that were labeled
    temperature: float = 1.0,
    dirichlet_alpha: float = 0.2,
) -> Dict[str, float]:
    """
    Turn doc-level (option, confidence) into a calibrated probability distribution.

    Steps:
      1) Build an option score by summing (confidence * rank_weight).
      2) Apply temperature softmax to get a sharp/soft distribution.
      3) Dirichlet smoothing with alpha, then renormalize and return as {option: prob}.
    """
    n = len(options)
    opt2idx = {o: i for i, o in enumerate(options)}
    scores = np.zeros(n, dtype=np.float64)

    # aggregate scores
    rank_pos = {did: i for i, did in enumerate(rank_ids)}
    for did, (opt, conf) in votes.items():
        if opt not in opt2idx:
            continue
        r = rank_pos.get(did)
        # If doc wasn't in rank_ids (shouldn't happen), give a tiny rank weight
        rw = _rank_weight(r) if r is not None else 1.0 / math.log2(2.0 + 500.0)
        scores[opt2idx[opt]] += max(0.0, min(1.0, float(conf))) * rw

    # if everything is zero (e.g., no parse), fall back to uniform
    if float(scores.sum()) <= 0.0:
        probs = np.full(n, 1.0 / n, dtype=np.float64)
    else:
        probs = _softmax_temp(scores, temperature=max(1e-6, float(temperature)))

    # Dirichlet smoothing toward uniform
    alpha = max(0.0, float(dirichlet_alpha))
    if alpha > 0.0:
        uniform = np.full(n, 1.0 / n, dtype=np.float64)
        probs = (probs + alpha * uniform) / (1.0 + alpha)

    # final normalize (safety) and convert to plain floats
    s = probs.sum()
    if s > 0:
        probs = probs / s
    else:
        probs = np.full(n, 1.0 / n, dtype=np.float64)

    return {options[i]: float(probs[i]) for i in range(n)}

# mas_survey/aggregate.py
from typing import List, Optional

def select_supports(
    *,
    fused_rank: List[str],
    size: int,
    all_candidates_fallback: Optional[List[str]] = None,
) -> List[str]:
    """Return exactly `size` unique IDs.
    1) take fused_rank in order (dedup),
    2) pad from fallback in order (dedup),
    3) if still short, cycle deterministically over fallback,
    4) truncate to size.
    """
    out: List[str] = []
    seen = set()

    def push(seq):
        for did in seq:
            if did not in seen:
                seen.add(did)
                out.append(did)
                if len(out) >= size:
                    return True
        return False

    # 1) from fused
    if push(fused_rank):
        return out[:size]

    # 2) pad from fallback
    fb = all_candidates_fallback or []
    if push(fb):
        return out[:size]

    # 3) cycle if still short (only if we have any pool at all)
    i = 0
    L = len(fb)
    while len(out) < size and L > 0:
        did = fb[i % L]
        # already deduped via `seen`; only grows when unseen appear (rare if L small)
        if did not in seen:
            seen.add(did)
            out.append(did)
        i += 1
        # safety break: if we looped L times without adding, nothing more to add
        if i >= 3 * L:   # conservative guard, avoids spins on tiny sets
            break

    # 4) If still short (tiny corpora), return what we have (valid but < size).
    # For full runs, ensure size==100 as required by Kaggle.
    return out[:size]
