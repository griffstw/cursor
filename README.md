# Document Summarization and Study Guide Agent

An AI agent that accepts documents (PDF, text) from users, extracts key ideas, returns concise summaries, and generates study guides (Q&A, flashcards). Includes both a FastAPI server and CLI.

## Features
- Upload multiple files (PDF, .txt, .text)
- Key ideas extraction (RAKE)
- Summarization pipeline (chunked TextRank fallback + LLM)
- Study guide generation: Q&A pairs and flashcards
- CLI utilities
- FastAPI endpoints: `/summarize`, `/study`

## Getting Started

### 1) Install
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

(Optional) Set your OpenAI credentials for higher-quality outputs.
```bash
export OPENAI_API_KEY=... 
export OPENAI_MODEL=gpt-4o-mini
```

### 2) Run API server
```bash
uvicorn app.api.server:app --reload --host 0.0.0.0 --port 8000
```

### 3) Use CLI
```bash
python -m app.cli summarize path/to/file_or_dir
python -m app.cli study path/to/file_or_dir
```

### 4) Try with sample
```bash
python -m app.cli summarize samples/sample.txt
python -m app.cli study samples/sample.txt
```

## Rubric Alignment
- Completeness: End-to-end CLI and API with multi-file upload; key features implemented.
- Correctness: Deterministic local fallbacks (TextRank, RAKE); sound chunk+merge summarization.
- Complexity: Modular architecture (ingestion, processing, agents, LLM with fallback); chunked summarization; study guide generation.
- Innovation: Offline-capable hybrid approach; pluggable LLM client; supports diverse study outputs.

## Notes
- The local summarization/QA/flashcards are heuristics; using `OPENAI_API_KEY` significantly improves results.
- PDF extraction uses `pypdf` and may vary by document quality.
