# Simple makefile to simplify repetitive build env management tasks under posix

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

# Install vale first with `pip install vale`
docstyle:
	@echo "Running vale"
	@vale --config doc/.vale.ini doc pyvista examples ./*.rst --glob='!*{_build,AUTHORS.rst,_autosummary,source/examples}*'
