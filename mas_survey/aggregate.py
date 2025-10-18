# mas_survey/aggregate.py
import math
from typing import Dict, List, Tuple, Optional

def aggregate_votes(
    options: List[str],
    doc2vote: Dict[str, Tuple[str, float]],
    *,
    dirichlet_alpha: float,
    temperature: float,
) -> Dict[str, float]:
    # temperature softening on confidences
    w = {}
    for _, (opt, conf) in doc2vote.items():
        w[opt] = w.get(opt, 0.0) + (conf ** (1.0 / max(1e-6, temperature)))

    # Dirichlet smoothing across options
    counts = {o: w.get(o, 0.0) + dirichlet_alpha for o in options}
    Z = sum(counts.values()) or 1.0
    dist = {o: counts[o] / Z for o in options}

    s = sum(dist.values())
    if not math.isfinite(s) or s <= 0:
        k = len(options)
        return {o: 1.0 / k for o in options}
    return {o: v / s for o, v in dist.items()}


def select_supports(
    ranked_unique_ids: List[str],
    *,
    required_k: int,
    all_candidates_fallback: Optional[List[str]] = None,
    corpus_all_ids: Optional[List[str]] = None,
) -> List[str]:
    """
    Returns exactly `required_k` unique IDs.
    Priority: ranked_unique_ids -> all_candidates_fallback -> corpus_all_ids.
    Deterministic padding; never crashes on empty inputs.
    """
    out: List[str] = []
    seen = set()

    def add_list(src: Optional[List[str]]):
        nonlocal out, seen
        if not src:
            return
        for did in src:
            if did not in seen:
                out.append(did)
                seen.add(did)
                if len(out) == required_k:
                    return

    add_list(ranked_unique_ids)
    if len(out) < required_k:
        add_list(all_candidates_fallback)
    if len(out) < required_k and corpus_all_ids:
        add_list(corpus_all_ids)

    # last resort deterministic padding (should be rare)
    if len(out) < required_k and corpus_all_ids:
        while len(out) < required_k and corpus_all_ids:
            out.append(corpus_all_ids[len(out) % len(corpus_all_ids)])

    return out[:required_k]
