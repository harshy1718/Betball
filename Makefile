.PHONY: install lint format typecheck test check ingest
install:
	python -m pip install -e ".[dev,web]"
lint:
	ruff check src tests
format:
	ruff format src tests
typecheck:
	mypy src
test:
	pytest
check: lint typecheck test
ingest:
	football-edge ingest --src "raw data"
