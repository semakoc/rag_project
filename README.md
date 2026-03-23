# AI-Mediated Dialogue — Dual RAG Pipeline

Production-style Python backend and Streamlit frontend supporting **IRB Protocol 26.0670** (Boston College): *AI-Mediated Dialogue for Intergroup Understanding*. The system helps participants explore the **other** group’s materials through empathetic, non-debate conversation.

## Project overview

Participants who identify as **pro-life** query a chatbot grounded in **pro-choice** documents, and **pro-choice** participants query **pro-life** documents. Retrieval combines **semantic** (embedding + FAISS L2) and **keyword** (stopword-filtered overlap) scores, then an LLM (or offline fallback) produces a warm, non-adversarial reply. The app does **not** facilitate debate; it supports curiosity and perspective-taking.

## Architecture (dual RAG + cross-group routing)

```
Participant (pro-life) ──► Router picks opposing corpus (pro-choice)
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │  Chunks + FAISS + embed model │
                    └───────────────┬───────────────┘
                                    │
              ┌─────────────────────┴─────────────────────┐
              ▼                                           ▼
     Semantic search (L2)                    Keyword overlap score
              └─────────────────────┬─────────────────────┘
                                    ▼
                         Weighted fusion (dual RAG)
                                    │
                                    ▼
                         LLM prompt (ethics + context)
                                    │
                                    ▼
                           Response to participant
```

## Quick start

1. **Clone or copy** this repository and `cd` into the project folder.
2. **Create a virtual environment** (recommended): `python -m venv venv && source venv/bin/activate` (Windows: `venv\Scripts\activate`).
3. **Install dependencies:** `pip install -r requirements.txt`  
   (`transformers` is pinned to 4.x so the embedding model loads reliably with `sentence-transformers`.)
4. **Create data folders:** `mkdir -p data/pro_life data/pro_choice`
5. **Add materials:** place `.pdf` or `.txt` files in each folder (those directories are gitignored by default).
6. **Optional — OpenAI:** set `OPENAI_API_KEY` in a `.env` file in the project root (the app runs without a key using a context-only fallback).

Tests download the embedding model into a project-local `.hf_cache/` directory (see `tests/conftest.py`) so they do not require writing to `~/.cache`.

## How to add knowledge base materials

- Put **PDFs or TXT files** in `data/pro_life/` for pro-life-sourced text and `data/pro_choice/` for pro-choice-sourced text.
- On startup, the pipeline ingests all supported files, chunks text, embeds with `all-MiniLM-L6-v2`, and builds a FAISS index in memory (Streamlit caches this with `@st.cache_resource`).

## Running the demo

- **End-to-end smoke test:** `python main.py` (expects `data/pro_life/` and `data/pro_choice/` to exist; add at least one file per side for meaningful retrieval).
- **CLI for researchers:** `python demo/cli_app.py`
- **Participant UI:** `streamlit run demo/streamlit_app.py`

## Running tests

```bash
pytest tests/ -v --cov=src
```

Use a virtual environment with dev dependencies from `requirements.txt`.

## Dual RAG explained

- **Semantic search** finds passages whose *meaning* is close to the question (dense vectors + L2 distance).
- **Keyword search** boosts passages that share important *words* with the query, normalized by length, with common stopwords removed.
- **Fusion** normalizes each signal to \([0,1]\), combines with configurable weights, deduplicates by exact chunk text, and returns the top unique chunks. Together, this reduces both “missed keywords” and “right words, wrong sense” failures.

## Experimental design (cross-group routing)

Routing sends each participant to the **opposing** group’s index so they hear how that side’s materials frame values and experiences. That mismatch is intentional: it supports empathy across groups rather than reinforcing one’s own talking points.

## IRB and privacy

- **Logging** (`logs/interactions.txt`) stores **metadata only**: timestamp, anonymous `session_id`, group, query/response **lengths**, and number of context chunks. **Raw query and response text are not logged** — to protect participant confidentiality under Protocol 26.0670.
- **No PII** should be entered into the app for research use.
- The **FAISS index is static** after build; there is no online learning from user input.

## Environment variables

| Variable            | Purpose                                      |
|---------------------|----------------------------------------------|
| `OPENAI_API_KEY`    | Optional; enables GPT replies via OpenAI API |

Load from `.env` in the project root (`python-dotenv`).

---

*Protocol 26.0670 — Boston College Department of Psychology and Neuroscience; PI: Prof. Liane Young.*
