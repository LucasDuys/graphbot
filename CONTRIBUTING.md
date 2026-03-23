# Contributing to GraphBot

Thanks for your interest in contributing. GraphBot is built with a simple belief: powerful AI agents shouldn't cost a fortune.

## Quick Setup

```bash
git clone https://github.com/LucasDuys/graphbot
cd graphbot
pip install -e ".[dev,langsmith,embeddings]"
python scripts/healthcheck.py              # Verify setup
python -m pytest tests/test_core/ -q       # Run core tests
```

## How to Contribute

1. **Find an issue** -- Check [open issues](https://github.com/LucasDuys/graphbot/issues) or create one
2. **Fork and branch** -- `git checkout -b feature/your-feature`
3. **Write tests first** -- We follow TDD. Tests go in `tests/`
4. **Implement** -- Follow existing code patterns
5. **Run tests** -- `python -m pytest tests/ -q` (1500+ tests should pass)
6. **Commit** -- Format: `[component] brief description`
7. **Open a PR** -- Reference the issue number

## Code Standards

- Full type hints on all public functions
- No dead code, no magic numbers
- Structured logging (`loguru`)
- Error handling on all external calls

## Areas We Need Help With

- Browser automation workflows (Playwright)
- Additional LLM provider integrations
- Performance optimization (context assembly, retrieval)
- Documentation and examples
- Multi-language README translations (helps with GitHub Trending in other languages)

## Need Help?

Open an issue or start a [discussion](https://github.com/LucasDuys/graphbot/discussions).
