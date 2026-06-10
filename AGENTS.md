# AGENTS.md

## Project Overview

Python 3.10+ / Streamlit app ("Infrastructure AI Radar Hub") — an academic paper knowledge base with AI-powered analysis, hybrid search (FTS5 + LanceDB vector), and automated paper discovery from arXiv / Semantic Scholar.

## Commands

```bash
# Run the app
streamlit run app.py

# V2 pipeline test (creates temp test DB, validates WriteWorker concurrency, 2PC rollback, hybrid retrieval)
python scripts/test_pipeline.py

# Migrate V1 papers into V2 retrieval layer
python scripts/migrate_v1_to_v2.py --dry-run
python scripts/migrate_v1_to_v2.py --batch-size 5

# Retrieval quality regression (requires migration done + golden_benchmark.yaml doc_ids in DB)
python scripts/run_golden_benchmark.py
```

No lint, typecheck, formatter, or test framework is configured. No `requirements.txt` or `pyproject.toml` — dependencies are listed only in README.

## Architecture

- **Entry point**: `app.py` (Streamlit). Calls `init_db()`, `load_api_config()`, `start_scheduler()` at import time.
- **V1 layer**: `papers` + `ai_summaries` tables in SQLite (`storage/radar_hub.db`). Legacy but still active.
- **V2 layer**: `documents`, `chunks`, `search_chunks` (FTS5), `document_contents`, `entity_lexicon`, `entity_relations`, `quota_ledger`, `ingestion_errors` tables + LanceDB vector tables in `storage/lancedb/`.
- **SQLite writes**: All go through `WriteWorker` (singleton, async queue, single writer thread, WAL mode, `busy_timeout=30000`). Never write to SQLite directly from concurrent code.
- **LanceDB**: Vector dim hardcoded to `1024` in `core/lancedb_client.py`. Tables split by `source_type` (e.g., `vector_chunks_arxiv_paper`, `vector_chunks_local_pdf`).
- **2PC ingestion**: `IngestionCoordinator` writes SQLite first, then LanceDB. On LanceDB failure, SQLite status rolls back to `failed` and logs to `ingestion_errors`.

## Local Services

- Embedding: `http://127.0.0.1:8081/v1/embeddings` (model `qwen3-embedding`, dim must match `VECTOR_DIM=1024`)
- Reranker: `http://127.0.0.1:8082/v1/rerank` (model `qwen3-reranker`)
- If embedding is down, ingestion degrades to zero vectors — FTS5 still works but vector recall is empty.

## Critical Constraints

- `config/api_config.json` and `config/briefing_config.json` are **gitignored** (contain API keys). Never commit them. Auto-generated on first run if missing.
- `storage/` is **gitignored** (DB, PDFs, LanceDB data). Scripts that need a DB must handle `init_db()` themselves.
- `test_pipeline.py` copies the real `storage/radar_hub.db` to a temp test DB — it requires the real DB to exist first (run the app once).
- Golden benchmark thresholds: MRR >= 0.75, Hit@5 >= 0.80.
- API keys: prefer env vars (`DEEPSEEK_API_KEY`, `GEMINI_API_KEY`, `EXA_API_KEY`, `FIRECRAWL_API_KEY`) over hardcoded values in config JSON.
- Env var access: use `get_env_var()` from `core/env_helper.py` (not `os.environ.get()`). It falls back to Windows registry and root `.env` file — 5 core modules depend on this.
- `cost_manager.py` enforces daily/weekly USD budget circuit-breakers on paid API calls.

## Code Conventions

- All source files use `# -*- coding: utf-8 -*-` header.
- UI text, comments, and log messages are primarily in Chinese (Simplified).
- Config loading goes through `core/config_loader.py` — never read `api_config.json` directly.
- PDF paths stored in DB use relative `storage/library/filename.pdf` form. `resolve_pdf_path()` in `core/database.py` handles cross-machine portability by extracting filenames from legacy absolute paths.
