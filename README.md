[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/JG5U42VD)
# MAS for Automatic Survey

**Submission CSV header (exact):**

```
question,distribution,supports
````

- **`distribution`** — JSON object (single CSV cell) mapping the **exact option strings** to probabilities that **sum to 1.0 ± 1e-6**.  
- **`supports`** — JSON array (single CSV cell) of document IDs.

> **Note:** On the full competition you must output **exactly 100 unique** supports per question (deduplicated, deterministic). On the mini set you may have fewer.

---

## Quick start (Python 3.9, CPU)

```bash
python3.9 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
````

Edit `config.yaml` if needed, then:

```bash
# 1) Build (validate / prepare artifacts)
python -m index.build config.yaml --api_key dummy_api

# 2) Run (write the CSV submission)
python -m mas_survey.run --config config.yaml --api_key dummy_api
```

**Output:** `./artifacts/submission_js.csv`

---

## File layout

```
.
├── data/
│   ├── mini_documents.jsonl
│   └── dev/mini_dev.json
├── config.yaml
├── requirements.txt
├── index/
│   └── build.py
└── mas_survey/
    └── run.py
```

You provide the data files (mini or full):

* `data/mini_documents.jsonl` — one JSON per line with fields like `"id"`, `"title"`, `"description"`, `"post_content"`, `"content"`, …
* `data/dev/mini_dev.json` — a single JSON object keyed by the **full question string**; each value has a `distribution` stub (option keys with zeros) and an empty `supports` list.
* Full `documents.jsonl` (and embeddings) are on the Kaggle dataset;  
  they are **too large for this repo**.  
  **Do not commit large artifacts** (keep repo size <10 MB).  
  Always add them to `.gitignore`.

---

## CSV schema reminder

* Header **exactly**: `question,distribution,supports`
* UTF-8; follow CSV quoting (double quotes inside a quoted cell are doubled).
* Probabilities **≥ 0** and sum to **1.0 ± 1e-6**.

---

## FAQ

**Can I use GPUs?**
No. CPU-only.

**Can I add libraries?**
Yes, within course constraints. Start here, then add BM25 / FAISS / scikit-learn as needed.
