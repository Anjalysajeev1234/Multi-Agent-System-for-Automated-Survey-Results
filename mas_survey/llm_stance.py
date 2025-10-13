# mas_survey/llm_stance.py
import hashlib, json, os, time
from pathlib import Path
from typing import List, Dict, Tuple
from retry import retry

CACHE_DIR = Path("./artifacts/cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

SYSTEM = (
    "You are a careful annotator. For each document, pick exactly ONE option "
    "that best matches the author's stance to the survey question. If unclear, choose the closest."
)

PROMPT_TMPL = """Question:
{question}

Options (choose one EXACTLY as written):
{options_block}

Return a JSON array, one object per document, in this exact schema:
[{{"doc_id":"<id>","option":"<one_of_options>","confidence": <float 0..1>}}, ...]

Documents:
{docs_block}
"""

def _hash_key(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def _cache_path(model: str, qhash: str, doc_id: str) -> Path:
    return CACHE_DIR / f"{model.replace('/','_')}__{qhash}__{doc_id}.json"

def _read_cache(model: str, qhash: str, doc_id: str):
    p = _cache_path(model, qhash, doc_id)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return None

def _write_cache(model: str, qhash: str, doc_id: str, obj: dict):
    p = _cache_path(model, qhash, doc_id)
    p.write_text(json.dumps(obj, ensure_ascii=True), encoding="utf-8")

def _mk_docs_block(ids: List[str], texts: List[str], max_chars: int = 1600) -> str:
    items = []
    for i, (did, t) in enumerate(zip(ids, texts), 1):
        t = t.replace("\n", " ").strip()
        if len(t) > max_chars:
            t = t[:max_chars] + " ..."
        items.append(f"{i}. [doc_id={did}] {t}")
    return "\n".join(items)

@retry(tries=4, delay=0.6, backoff=2, jitter=(0.0, 0.2))
def _together_call(model: str, api_key: str, system: str, prompt: str, max_output_tokens: int = 512) -> str:
    # Lazy import so this module is optional if not used
    from together import Together
    client = Together(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=max_output_tokens,
        top_p=0.95,
    )
    return resp.choices[0].message.content

def label_with_llm(
    question: str,
    options: List[str],
    doc_ids: List[str],
    doc_texts: List[str],
    model: str,
    api_key: str,
    batch_size: int = 8,
) -> Dict[str, Tuple[str, float]]:
    """
    Returns: mapping doc_id -> (option_str, confidence_float)
    (uses disk cache; only calls LLM for uncached docs)
    """
    qhash = _hash_key(question + "||" + "||".join(options))
    result: Dict[str, Tuple[str, float]] = {}

    # fill from cache where possible
    for did in doc_ids:
        cached = _read_cache(model, qhash, did)
        if cached:
            result[did] = (cached["option"], float(cached.get("confidence", 0.6)))

    # batch remaining
    remaining = [(i, did, txt) for i, (did, txt) in enumerate(zip(doc_ids, doc_texts)) if did not in result]
    if not remaining:
        return result

    options_block = "\n".join([f"- {o}" for o in options])

    for start in range(0, len(remaining), batch_size):
        chunk = remaining[start:start + batch_size]
        ids = [did for _, did, _ in chunk]
        texts = [txt for _, _, txt in chunk]

        docs_block = _mk_docs_block(ids, texts)
        prompt = PROMPT_TMPL.format(question=question, options_block=options_block, docs_block=docs_block)

        raw = _together_call(model=model, api_key=api_key, system=SYSTEM, prompt=prompt, max_output_tokens=512)

        # Try parse JSON anywhere in the output
        parsed = None
        try:
            start_idx = raw.find("[")
            end_idx = raw.rfind("]")
            parsed = json.loads(raw[start_idx:end_idx + 1])
        except Exception:
            parsed = []

        # Index by doc_id and write cache
        by_id = {obj.get("doc_id"): obj for obj in parsed if isinstance(obj, dict) and obj.get("doc_id") in ids}
        for did in ids:
            obj = by_id.get(did)
            if obj and obj.get("option") in options:
                conf = float(obj.get("confidence", 0.6))
                result[did] = (obj["option"], max(0.0, min(1.0, conf)))
                _write_cache(model, qhash, did, {"option": obj["option"], "confidence": result[did][1]})
            else:
                # fallback if parsing failed: mark as lowest confidence to reduce weight
                result[did] = (options[0], 0.2)  # option placeholder; aggregator weight will be tiny
                _write_cache(model, qhash, did, {"option": options[0], "confidence": 0.2})

    return result
