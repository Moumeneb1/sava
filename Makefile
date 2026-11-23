# Copyright 2026 Inworld AI — Apache-2.0
.DEFAULT_GOAL := help

PYTHON ?= .venv/bin/python
STYLE_PATHS := src bench/scripts docs examples tests $(wildcard */src */tests)
ISORT_PATHS := $(STYLE_PATHS)
FORMAT_PATHS := $(STYLE_PATHS)

.PHONY: help install test lint format format-check

help: ## Show the available commands
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*## / {printf "%-14s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# Python 3.11 lints as CI's 3.10 does: from 3.12 pylint reads quotes inside
# f-strings differently and reports inconsistent-quotes that CI does not. An
# existing .venv is reused only on 3.10 or 3.11.
install: ## Create .venv on Python 3.11, or reuse one on 3.10 or 3.11, and install the dev and eval extras (requires uv)
	@if [ -x .venv/bin/python ]; then \
		.venv/bin/python -c 'import sys; sys.exit(not (3, 10) <= sys.version_info[:2] <= (3, 11))' || { \
			echo ".venv is not on Python 3.10 or 3.11: older is unsupported and newer lints unlike CI; remove it (rm -rf .venv) and run make install again"; \
			exit 1; }; \
	else uv venv --python 3.11 .venv; fi
	uv pip install --python $(PYTHON) -e ".[dev,eval]"

test: ## Run the CPU tests and report optional and golden-parity skips
	$(PYTHON) -m pytest -q -rs

lint: ## Check Google Python style and tracked-file hygiene
	PYTHONPATH=src $(PYTHON) -m pylint --rcfile=.pylintrc inworld
	$(PYTHON) -m pylint --rcfile=.pylintrc $(filter-out src,$(STYLE_PATHS))
	@if git grep -nE '(^|[^[:alnum:]_/.])/(home|inworld|mnt)/' -- . ':!AGENTS.md' ':!Makefile' ':!.github/workflows/ci.yml'; then exit 1; fi

format: ## Sort imports and format Python in place
	$(PYTHON) -m isort $(FORMAT_PATHS)
	$(PYTHON) -m yapf --in-place --recursive --style=.style.yapf $(FORMAT_PATHS)

format-check: ## Check all Python formatting without changing files
	$(PYTHON) -m yapf --diff --recursive --style=.style.yapf $(STYLE_PATHS)
	$(PYTHON) -m isort --check-only --diff $(ISORT_PATHS)
