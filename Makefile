# Simple makefile to simplify repetitive build env management tasks under posix

# Directories to run style checks against
CODE_DIRS ?= doc examples examples_flask pyvista tests
# Files in top level directory
CODE_FILES ?= *.py *.rst *.md

# doctest-modules-local-namespace must be off screen to avoid plotting everything
doctest-modules-local-namespace: export PYVISTA_OFF_SCREEN = True

doctest-modules:
	@echo "Runnnig module doctesting"
	pytest -v --doctest-modules pyvista

doctest-modules-local-namespace:
	@echo "Running module doctesting using docstring local namespace"
	python tests/check_doctest_names.py

example-coverage:
	python -m ansys.tools.example_coverage -f pyvista

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
