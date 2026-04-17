# Simple makefile to simplify repetitive build env management tasks under posix

.DEFAULT_GOAL := test

.PHONY: all clean coverage coverage-xml coverage-html coverage-docs docstyle sync-deps lint typecheck test test-core test-plotting doctest docs docs-test integration

# `all` is a POSIX convention; alias it to the default test target.
all: test

# Remove build / coverage artifacts from the repo root.
clean:
	@echo "Cleaning build and coverage artifacts"
	@rm -rf build dist .tox .pytest_cache htmlcov coverage*.xml coverage*.html .coverage

# Directories to run style checks against
CODE_DIRS ?= doc examples examples_trame pyvista tests
# Files in top level directory
CODE_FILES ?= *.py *.rst *.md

coverage:
	@echo "Running coverage"
	@pytest -v --cov pyvista

coverage-xml:
	@echo "Reporting XML coverage"
	@pytest -v --cov pyvista --cov-report xml

coverage-html:
	@echo "Reporting HTML coverage"
	@pytest -v --cov pyvista --cov-report html

coverage-docs:
	@echo "Reporting documentation coverage"
	@make -C doc html SPHINXOPTS="-Q" -b coverage
	@cat doc/_build/coverage/python.txt

# Vale is pinned to match CI (.github/workflows/style-docstring.yml).
# Install with: `uv tool install vale@2.29.5`
# Newer vale versions currently fail on the pyvista vocab config.
docstyle:
	@echo "Running vale"
	@vale --config doc/.vale.ini doc pyvista examples

sync-deps:
	@echo "Installing dev dependencies"
	@uv sync --group dev

lint:
	@echo "Running pre-commit"
	@uv run pre-commit run --all-files

typecheck:
	@echo "Running mypy"
	@uv run tox -e mypy

# Run tests via tox so local runs match CI exactly. Filter/flag definitions
# live in tox.ini so they are maintained in one place.
# Extra pytest args can be passed via ARGS, e.g. `make test ARGS="-n 10 -k filters"`
test:
	@echo "Running full test suite (matches CI flags)"
	@uv run tox -e test -- $(ARGS)

# Core tests only (matches CI `pyX.Y-core` env).
test-core:
	@echo "Running core tests (matches CI)"
	@uv run tox -e test-core -- $(ARGS)

# Plotting tests only (matches CI `pyX.Y-plotting` env).
test-plotting:
	@echo "Running plotting tests (matches CI)"
	@uv run tox -e test-plotting -- $(ARGS)

# Run all docstring tests (matches CI `tox -f doctest`).
# Executes both doctest-modules and doctest-local tox envs.
doctest:
	@echo "Running docstring tests (matches CI)"
	@uv run tox -f doctest -- $(ARGS)

# Build the full documentation (matches CI `tox -e docs-build`).
# Runs pre-gen steps (make_tables, make_external_gallery) and sphinx.
docs:
	@echo "Building documentation (matches CI)"
	@uv run tox -e docs-build -- $(ARGS)

# Test the built documentation (matches CI `tox -e docs-test`).
# Requires `make docs` to have been run first.
docs-test:
	@echo "Testing built documentation (matches CI)"
	@uv run tox -e docs-test -- $(ARGS)

# Run an integration test env (matches CI `tox -e integration-<project>`).
# Specify project via PROJECT, e.g. `make integration PROJECT=trame`.
# Supported projects: trame, geovista, mne, pyvistaqt
integration:
	@test -n "$(PROJECT)" || { echo "Error: PROJECT is required (trame|geovista|mne|pyvistaqt)"; exit 1; }
	@echo "Running integration-$(PROJECT) tests (matches CI)"
	@uv run tox -e integration-$(PROJECT) -- $(ARGS)
