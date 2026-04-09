.PHONY: install dev lint format test run

install:
	uv pip install -e .

dev:
	uv pip install -e ".[dev]"

lint:
	ruff check .

format:
	ruff format .

test:
	pytest

run:
	aria run "What are the latest advances in quantum error correction?"
