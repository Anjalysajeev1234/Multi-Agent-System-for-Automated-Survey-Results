
# MAS for Automatic Survey
### Project Overview

I designed and implemented a modular Multi-Agent System (MAS) for automated public-opinion estimation, capable of predicting population-level response distributions for complex survey questions. The system integrates LLMs, dense+sparse retrieval, evidence attribution, and calibration pipelines to produce statistically grounded predictions with full traceability.

This project simulates large-scale social science workflows by leveraging 230k+ Reddit documents to infer real-world attitude distributions about U.S. national survey questions .
What I Built
1. Multi-Agent Reasoning Architecture

A cooperative set of agents with explicit state-passing layers:

Query Planner Agent – parses questions, generates diversified queries.

Retriever Agent – hybrid BM25 + dense embeddings search over 230k documents.

Evidence Filtering Agent – quality checks, deduplication, and stance extraction.

Aggregator Agent – probabilistic modelling + calibration (normalization, smoothing).

Attribution Agent – compiles exactly 100 evidence-backed support documents per question.

This ensured high coverage, reduced hallucination risk, and improved interpretability — all core MAS goals described in the assignment spec .

 ### Tech Stack & Methods Used
Information Retrieval (IR)

Whoosh BM25 index for sparse retrieval.

FAISS / Dense vector search over provided embeddings (m2-bert-80M-32k-retrieval) .

Retrieval fusion (RRF / weighted hybrid) to stabilize quality.

Deduplication heuristics + cosine similarity filtering.

#### Large Language Models

LLM-based:

Query expansion

Stance classification (per document → survey answer option)

Evidence summarization & structured JSON outputs

Prompting strategies:

Chain-of-thought

Function-style structured output parsing

Self-consistency sampling

#### Pipeline Engineering

Deterministic seeds, bounded loops, reproducible configs

Two CLI interfaces required by the spec

python -m index.build

python -m mas_survey.run

CPU-only pipelines, optimized for grading environment.

Data Engineering & Preprocessing

Aligned with Week 2 curriculum topics (cleaning, normalization, deduplication) :

HTML/boilerplate cleanup

Unicode normalization

Tokenization / stopword-aware filtering

Near-duplicate removal to avoid skew

### Results & Measurable Impact
🎯 Predictive Performance

Delivered calibrated probability distributions for all survey questions, compliant with all Kaggle rules (sum to 1.0 ± tolerance) .

Achieved competitive Jensen–Shannon (JS) scores in development evaluations.

Produced a deterministic pipeline with stable performance across runs.

📚 Evidence Quality

Consistently generated 100 unique evidence documents per question, improving:

Interpretability

Transparency

Downstream MAP scores in leaderboard evaluations

⚡ Efficiency & Engineering

System successfully processed the mini-set (50 docs, 1 question) within required time constraints for reproducibility.

Robust error handling: retry logic, backoff, API-safe operations.

🏆 Key Achievements

Built a fully reproducible MAS that can analyze large-scale social-media text to infer human opinion distributions.

Demonstrated full-stack IR + LLM orchestration, from indexing → retrieval → reasoning → aggregation → evaluation.

Delivered transparent, evidence-grounded predictions — addressing real limitations of naive LLM opinion generation like bias, hallucination, and lack of traceability .
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

