.PHONY: help setup data train audit corpus demo eval pipeline test test-unit lint format clean

help:  ## Show available targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  %-10s %s\n", $$1, $$2}'

setup:  ## Install dependencies and git hooks
	uv sync
	uv run pre-commit install

data:  ## Download UCI dataset 296 (checksum-verified) and build splits
	uv run python -m cliniq.data_prep

train:  ## Train, calibrate, and evaluate the risk model
	uv run python -m cliniq.risk_model

audit:  ## Run the fairness audit
	uv run python -m cliniq.fairness_audit

corpus:  ## Build the retrieval corpus
	uv run python -m cliniq.rag.build_corpus

demo:  ## Run the agent on two sample patients
	uv run python -m cliniq.agent.clinical_agent

eval:  ## Run all evals and write reports/eval_report.md
	uv run python -m evals.run_all

pipeline: data train audit corpus eval  ## Full pipeline, end to end

test:  ## Run all tests (integration tests skip without artifacts)
	uv run pytest

test-unit:  ## Run fast unit tests only
	uv run pytest tests/unit

lint:  ## Lint, format check, type check
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy

format:  ## Auto-format and fix lint issues
	uv run ruff format .
	uv run ruff check --fix .

clean:  ## Remove generated data, models, and caches (keeps reports/)
	rm -rf data/processed data/corpus/chunks.jsonl models .pytest_cache .mypy_cache .ruff_cache
