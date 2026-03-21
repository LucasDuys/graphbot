# GraphBot Quickstart

## Setup
1. Clone: `git clone https://github.com/LucasDuys/graphbot`
2. Install: `pip install -e ".[dev,langsmith]"`
3. Create `.env.local` with your API keys:
   ```
   OPENROUTER_API_KEY=sk-or-v1-...
   LANGSMITH_API_KEY=lsv2_pt_...
   LANGSMITH_ENDPOINT=https://eu.api.smith.langchain.com
   LANGSMITH_PROJECT=graphbot
   LANGSMITH_TRACING=true
   ```

## Run
- Seed graph: `python scripts/seed_graph.py`
- Live test: `python scripts/test_live.py`
- Benchmarks: `python scripts/bench_graph.py`
- A/B comparison: `python scripts/compare.py "your task here"`
- Tests: `python -m pytest tests/ -v`

## Architecture
See CLAUDE.md for full architecture documentation.
