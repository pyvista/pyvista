# Minimal makefile for Sphinx documentation
#

export PYVISTA_OFF_SCREEN := True

# You can set these variables from the command line.
SPHINXOPTS    =
DEFAULTS	  = -w sphinx_warnings.txt -W --keep-going
SPHINXBUILD   = python -msphinx
SPHINXPROJ    = pyvista
SOURCEDIR     = source
BUILDDIR      = _build


# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(DEFAULTS) $(O)

.PHONY: help Makefile

clean-except-examples:
	rm -rf $(BUILDDIR)/*
	rm -f errors.txt
	rm -f sphinx_warnings.txt
	rm -rf $(SOURCEDIR)/images/auto-generated
	rm -f $(SOURCEDIR)/getting-started/external_examples.rst
	rm -rf $(SOURCEDIR)/tags
	rm -rf $(SOURCEDIR)/api/examples/dataset-gallery/
	find $(SOURCEDIR) -type d -name "_autosummary" -exec rm -rf {} +

clean: clean-except-examples
	rm -rf $(SOURCEDIR)/examples/

# remove autosummary files
clean-autosummary:
	find $(SOURCEDIR) -type d -name "_autosummary" -exec rm -rf {} +

# build html docs in parallel using all available CPUs
# WARNING: this is a resource hog
phtml:
	$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(DEFAULTS) $(O) -j auto

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(DEFAULTS) $(O)

update-intersphinx:
	cd intersphinx && ./update.sh

build-gallery-cards:
	cd source && python make_tables.py

# nitpicky linkcheck
linkcheck:
	@$(SPHINXBUILD) -b linkcheck "$(SOURCEDIR)" -D nitpicky=1 -d _build/doctrees . _build/linkcheck

linkcheck-grep:
	@! grep -h "^.*:.*: \[\(\(local\)\|\(broken\)\)\]" _build/linkcheck/output.txt

# Spin up a local http server to view the rendered documentation.
# This is required for interactive examples to work.
serve-html:
	python -m http.server 11000 --directory "$(BUILDDIR)"/html
