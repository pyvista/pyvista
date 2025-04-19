# Simple makefile to simplify repetitive build env management tasks under posix

# Directories to run style checks against
CODE_DIRS ?= doc examples examples_trame pyvista tests
# Files in top level directory
CODE_FILES ?= *.py *.rst *.md

### Installation targets ###
ci-install-test:
	@echo "Installing PyVista test dependencies"
	pip install '.[test]'

ci-install-docs:
	@echo "Installing PyVista docs dependencies"
	pip install '.[docs]'

ci-install-typing:
	@echo "Installing PyVista typing dependencies"
	pip install '.[typing]'

ci-install-vtk-dev:
	@echo "Installing latest VTK dev wheel"
	pip install vtk --upgrade --pre --no-cache --extra-index-url https://wheels.vtk.org

### Report targets ###
PYVISTA_REPORT = python -c "import pyvista; print(pyvista.Report(gpu=$(GPU_FLAG))); from pyvista import examples; print('User data path:', examples.USER_DATA_PATH)"

ci-report:
	@$(eval GPU_FLAG = True)
	$(PYVISTA_REPORT)

ci-report-no-gpu:
	@$(eval GPU_FLAG = False)
	$(PYVISTA_REPORT)

### Pytest targets ###
PYTEST_ENV = PYTEST_ADDOPTS="--color=yes"
PYTEST_ARGS = -v
PYTEST = $(PYTEST_ENV) python -m pytest $(PYTEST_ARGS)

# Definitions for CORE tests
PYTEST_CORE_ARGS = --ignore=tests/plotting
PYTEST_CORE_ARGS_COV = --cov=pyvista --cov-branch
PYTEST_CORE = $(PYTEST) $(PYTEST_CORE_ARGS)
PYTEST_CORE_COV = $(PYTEST_CORE) $(PYTEST_CORE_ARGS_COV)

# Definitions for PLOTTING tests
PYTEST_PLOTTING_ARGS = tests/plotting --fail_extra_image_cache --generated_image_dir debug_images
PYTEST_PLOTTING_ARGS_COV = $(PYTEST_CORE_ARGS_COV) --cov-append --cov-report=xml
PYTEST_PLOTTING = $(PYTEST) $(PYTEST_PLOTTING_ARGS)
PYTEST_PLOTTING_COV = $(PYTEST_PLOTTING) $(PYTEST_PLOTTING_ARGS_COV)

ci-test-core:
	@echo "Running core tests"
	$(PYTEST_CORE)

ci-test-core-cov:
	@echo "Running core tests with test coverage"
	$(PYTEST_CORE_COV)

ci-test-plotting:
	@echo "Running plotting tests"
	$(PYTEST_PLOTTING)

ci-test-plotting-cov:
	@echo "Running plotting tests with test coverage"
	$(PYTEST_PLOTTING_COV)

ci-test-doc:
	@echo "Running documentation tests"
	$(PYTEST) tests/doc/tst_doc_images.py

# must be off screen to avoid plotting everything
ci-doctest-modules: export PYVISTA_OFF_SCREEN = True
ci-doctest-modules-local-namespace: export PYVISTA_OFF_SCREEN = True

ci-doctest-modules:
	@echo "Running module doctesting"
	$(PYTEST) --doctest-modules pyvista

ci-doctest-modules-local-namespace:
	@echo "Running module doctesting using docstring local namespace"
	python tests/check_doctest_names.py

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
