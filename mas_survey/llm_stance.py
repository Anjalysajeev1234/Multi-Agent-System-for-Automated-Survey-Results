# mas_survey/llm_stance.py
from __future__ import annotations
import hashlib, json
from pathlib import Path
from typing import Dict, List, Tuple
from retry import retry

CACHE_DIR = Path("./artifacts/cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

SYSTEM = (
    "You are a careful annotator. For each document, pick exactly ONE option "
    "that best matches the author's stance to the survey question. "
    "If unclear, choose the closest. Return strict JSON as requested."
)

PROMPT_TMPL = """Question:
{question}

Options (choose ONE, exactly as written):
{options_block}

Return a JSON array with one object per document in this exact schema:
[{{"doc_id":"<id>","option":"<one_of_options>","confidence": <float 0..1>}}, ...]

Documents:
{docs_block}
"""

def _hash_key(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def _cache_path(model: str, qhash: str, doc_id: str) -> Path:
    safe_model = model.replace("/", "_")
    return CACHE_DIR / f"{safe_model}__{qhash}__{doc_id}.json"

def _read_cache(model: str, qhash: str, doc_id: str):
    p = _cache_path(model, qhash, doc_id)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None

def _write_cache(model: str, qhash: str, doc_id: str, obj: dict):
    p = _cache_path(model, qhash, doc_id)
    p.write_text(json.dumps(obj, ensure_ascii=True), encoding="utf-8")

def _mk_docs_block(ids: List[str], texts: List[str], max_chars: int = 1600) -> str:
    items = []
    for i, (did, t) in enumerate(zip(ids, texts), 1):
        t = (t or "").replace("\n", " ").strip()
        if len(t) > max_chars:
            t = t[:max_chars] + " ..."
        items.append(f"{i}. [doc_id={did}] {t}")
    return "\n".join(items)

def together_health_check(*, model: str, api_key: str, timeout_tokens: int = 8) -> str:
    """
    Tiny probe to ensure the key+model are usable BEFORE we start labeling.
    Raises on Together errors; returns the model name on success.
    """
    from together import Together, error as t_errors
    try:
        client = Together(api_key=api_key)
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Health check"},
                {"role": "user", "content": "Reply with OK"},
            ],
            temperature=0.1,
            max_tokens=int(timeout_tokens),
            top_p=0.95,
        )
        _ = resp.choices[0].message.content
        return model
    except t_errors.TogetherError as e:
        raise RuntimeError(f"Together health-check failed: {e}") from e


@retry(tries=4, delay=0.6, backoff=2, jitter=(0.0, 0.2))
def _together_call(
    *,
    model: str,
    api_key: str,
    system: str,
    prompt: str,
    max_output_tokens: int = 512,
    temperature: float = 0.2,
) -> str:
    from together import Together, error as t_errors
    try:
        client = Together(api_key=api_key)
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_output_tokens,
            top_p=0.95,
        )
        return resp.choices[0].message.content
    except t_errors.TogetherError as e:
        # Bubble up so caller sees the error instead of silently falling back
        raise RuntimeError(f"Together API failed: {e}") from e


def label_with_llm(
    *,
    question: str,
    options: List[str],
    doc_ids: List[str],
    doc_texts: List[str],
    model: str,
    api_key: str,
    batch_size: int = 8,
    temperature: float = 0.2,
    max_output_tokens: int = 512,
) -> Dict[str, Tuple[str, float]]:
    """
    Returns: mapping doc_id -> (option_str, confidence_float)
    Uses a simple disk cache so repeated runs are cheap.
    """
    qhash = _hash_key(question + "||" + "||".join(options))
    result: Dict[str, Tuple[str, float]] = {}

    # fill from cache where possible
    for did in doc_ids:
        cached = _read_cache(model, qhash, did)
        if cached:
            result[did] = (cached["option"], float(cached.get("confidence", 0.6)))

    # remaining to annotate
    remaining_ids: List[str] = [did for did in doc_ids if did not in result]
    if not remaining_ids:
        return result

    options_block = "\n".join([f"- {o}" for o in options])
    id2text = {did: txt for did, txt in zip(doc_ids, doc_texts)}

    # process in batches
    for start in range(0, len(remaining_ids), batch_size):
        chunk_ids = remaining_ids[start:start + batch_size]
        texts = [id2text.get(did, "") for did in chunk_ids]

        docs_block = _mk_docs_block(chunk_ids, texts)
        prompt = PROMPT_TMPL.format(
            question=question,
            options_block=options_block,
            docs_block=docs_block,
        )

        raw = _together_call(
            model=model,
            api_key=api_key,
            system=SYSTEM,
            prompt=prompt,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
        )

        # Try parse JSON array from anywhere in the response
        parsed = []
        try:
            s = raw.find("["); e = raw.rfind("]")
            if s != -1 and e != -1 and e > s:
                parsed = json.loads(raw[s:e+1])
        except Exception:
            parsed = []

        # index by doc_id
        by_id = {}
        for obj in parsed:
            if not isinstance(obj, dict):
                continue
            did = obj.get("doc_id")
            opt = obj.get("option")
            conf = obj.get("confidence", 0.6)
            if did in chunk_ids and isinstance(opt, str) and opt in options:
                try:
                    by_id[did] = (opt, float(conf))
                except Exception:
                    by_id[did] = (opt, 0.6)

        # write outputs + cache
        for did in chunk_ids:
            if did in by_id:
                opt, conf = by_id[did]
                conf = max(0.0, min(1.0, float(conf)))
                result[did] = (opt, conf)
                _write_cache(model, qhash, did, {"option": opt, "confidence": conf})
            else:
                # parsing miss: pick first option at low confidence (down-weighted later)
                result[did] = (options[0], 0.2)
                _write_cache(model, qhash, did, {"option": options[0], "confidence": 0.2})

    return result
