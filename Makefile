# Simple makefile to simplify repetitive build env management tasks under posix

# Directories to run style checks against
CODE_DIRS ?= doc examples examples_flask pyvista tests
# All files in toplevel directory
CODE_FILES ?= *.py *.rst *.md
CODESPELL_SKIP ?= "*.pyc,*.txt,*.gif,*.png,*.jpg,*.ply,*.vtk,*.vti,*.vtu,*.js,*.html,*.doctree,*.ttf,*.woff,*.woff2,*.eot,*.mp4,*.inv,*.pickle,*.ipynb,flycheck*,./.git/*,./.hypothesis/*,*.yml,./doc/_build/*,./doc/images/*,./dist/*,*~,.hypothesis*,./doc/examples/*,*.mypy_cache/*,*cover,./tests/tinypages/_build/*,*/_autosummary/*"
CODESPELL_IGNORE ?= "ignore_words.txt"

# doctest modules must be off screen to avoid plotting everything
doctest-modules: export PYVISTA_OFF_SCREEN = True
doctest-modules-local-namespace: export PYVISTA_OFF_SCREEN = True

stylecheck: codespell pydocstyle lint

codespell:
	@echo "Running codespell"
	@codespell $(CODE_DIRS) $(CODE_FILES) -S $(CODESPELL_SKIP) -I $(CODESPELL_IGNORE)

pydocstyle:
	@echo "Running pydocstyle"
	@pydocstyle pyvista --match='(?!coverage).*.py'

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

mypy:
	@echo "Running mypy static type checking"
	mypy pyvista/core/ --no-incremental
	mypy pyvista/themes.py --no-incremental

lint:
	@echo "Linting with flake8"
	@flake8 $(CODE_DIRS) $(CODE_FILES)

isort:
	@echo "Formatting with isort"
	isort .
