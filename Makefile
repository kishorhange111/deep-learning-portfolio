# Common developer commands. Run `make help` to list them.
.DEFAULT_GOAL := help
.PHONY: help install install-dev lint format test check clean

help:           ## list the available commands
	@grep -E '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  %-12s %s\n", $$1, $$2}'

install:        ## GPU runtime stack for the training scripts + the shared helpers package
	pip install -r requirements.txt -e .

install-dev:    ## lint/test tooling + the shared helpers package (CPU torch is enough)
	pip install -r requirements-dev.txt -e .

lint:           ## ruff lint + format check (what CI runs)
	ruff check .
	ruff format --check .

format:         ## auto-fix lint findings and reformat
	ruff check --fix .
	ruff format .

test:           ## run the unit tests (CPU, no downloads)
	pytest

check: lint test ## everything CI runs

clean:          ## remove caches and build artefacts
	rm -rf .pytest_cache .ruff_cache build dist src/*.egg-info
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
