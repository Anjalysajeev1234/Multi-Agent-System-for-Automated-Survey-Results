# mas_survey/llm_stance.py
import hashlib, json
from pathlib import Path
from typing import List, Dict, Tuple
from retry import retry

CACHE_DIR = Path("./artifacts/cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

SYSTEM = (
    "You are a careful annotator. For each document, select EXACTLY ONE survey option "
    "that best reflects the author's stance to the question. If unclear, pick the closest. "
    "Return strict JSON only."
)

PROMPT_TMPL = """Question:
{question}

Options (choose one EXACTLY as written):
{options_block}

Return a JSON array with one object per document, schema:
[{{"doc_id":"<id>","option":"<one_of_options>","confidence": <float 0..1>}}, ...]

Documents:
{docs_block}
"""

def _hash_key(s: str) -> str:
    import hashlib
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def _cache_path(model: str, qhash: str, doc_id: str) -> Path:
    return CACHE_DIR / f"{model.replace('/','_')}__{qhash}__{doc_id}.json"

def _read_cache(model: str, qhash: str, doc_id: str):
    p = _cache_path(model, qhash, doc_id)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return None

def _write_cache(model: str, qhash: str, doc_id: str, obj: dict):
    _cache_path(model, qhash, doc_id).write_text(json.dumps(obj, ensure_ascii=True), encoding="utf-8")

def _mk_docs_block(ids: List[str], texts: List[str], max_chars: int = 1600) -> str:
    items = []
    for i, (did, t) in enumerate(zip(ids, texts), 1):
        t = (t or "").replace("\n", " ").strip()
        if len(t) > max_chars:
            t = t[:max_chars] + " ..."
        items.append(f"{i}. [doc_id={did}] {t}")
    return "\n".join(items)

@retry(tries=4, delay=0.6, backoff=2, jitter=(0.0, 0.2))
def _together_call(model: str, api_key: str, system: str, prompt: str, temperature: float, max_output_tokens: int) -> str:
    from together import Together
    client = Together(api_key=api_key)  # together==0.2.11
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_output_tokens,
        top_p=0.95,
    )
    return resp.choices[0].message.content

def _safe_parse_array(raw: str) -> List[dict]:
    try:
        s = raw.index("["); e = raw.rindex("]") + 1
        arr = json.loads(raw[s:e])
        if isinstance(arr, list):
            return arr
    except Exception:
        pass
    return []

def label_with_llm(
    question: str,
    options: List[str],
    doc_ids: List[str],
    doc_texts: List[str],
    *,
    model: str,
    api_key: str,
    batch_size: int,
    temperature: float,
    max_output_tokens: int,
) -> Dict[str, Tuple[str, float]]:
    """
    Returns: mapping doc_id -> (option_str, confidence_float in [0,1])
    Uses on-disk cache per (model, question-hash, doc_id).
    """
    qhash = _hash_key(question + "||" + "||".join(options))
    result: Dict[str, Tuple[str, float]] = {}

    # cache
    for did in doc_ids:
        c = _read_cache(model, qhash, did)
        if c:
            result[did] = (c["option"], float(c.get("confidence", 0.6)))

    remaining = [(did, txt) for (did, txt) in zip(doc_ids, doc_texts) if did not in result]
    if not remaining:
        return result

    options_block = "\n".join([f"- {o}" for o in options])

    for i in range(0, len(remaining), batch_size):
        chunk = remaining[i:i+batch_size]
        ids = [did for did, _ in chunk]
        texts = [txt for _, txt in chunk]

        docs_block = _mk_docs_block(ids, texts)
        prompt = PROMPT_TMPL.format(question=question, options_block=options_block, docs_block=docs_block)

        raw = _together_call(model=model, api_key=api_key, system=SYSTEM, prompt=prompt, temperature=temperature, max_output_tokens=max_output_tokens)
        arr = _safe_parse_array(raw)

        by_id = {obj.get("doc_id"): obj for obj in arr if isinstance(obj, dict)}
        for did in ids:
            obj = by_id.get(did)
            if obj and obj.get("option") in options:
                conf = max(0.0, min(1.0, float(obj.get("confidence", 0.6))))
                result[did] = (obj["option"], conf)
                _write_cache(model, qhash, did, {"option": obj["option"], "confidence": conf})
            else:
                # low-confidence fallback to avoid holes
                result[did] = (options[0], 0.2)
                _write_cache(model, qhash, did, {"option": options[0], "confidence": 0.2})

    return result
