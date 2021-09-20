# Simple makefile to simplify repetitive build env management tasks under posix

CODESPELL_DIRS ?= ./
CODESPELL_SKIP ?= "*.pyc,*.txt,*.gif,*.png,*.jpg,*.ply,*.vtk,*.vti,*.js,*.html,*.doctree,*.ttf,*.woff,*.woff2,*.eot,*.mp4,*.inv,*.pickle,*.ipynb,flycheck*,./.git/*,./.hypothesis/*,*.yml,./doc/_build/*,./doc/images/*,./dist/*,*~,.hypothesis*,./doc/examples/*,*.mypy_cache/*,*cover,./tests/tinypages/_build/*,*/_autosummary/*"
CODESPELL_IGNORE ?= "ignore_words.txt"

# doctest modules must be off screen to avoid plotting everything
doctest-modules: export PYVISTA_OFF_SCREEN = True
doctest-modules-local-namespace: export PYVISTA_OFF_SCREEN = True

stylecheck: codespell pydocstyle

codespell:
	@echo "Running codespell"
	@codespell $(CODESPELL_DIRS) -S $(CODESPELL_SKIP) -I $(CODESPELL_IGNORE)

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

mypy:
	@echo "Running mypy static type checking"
	mypy pyvista/core/ --no-incremental
	mypy pyvista/themes.py --no-incremental
