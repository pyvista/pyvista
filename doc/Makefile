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

clean-dataset-gallery:
	rm -rf "$(SOURCEDIR)/api/examples/dataset-gallery"

clean-external-examples:
	rm -f $(SOURCEDIR)/getting-started/external_examples.rst

# remove autosummary files
clean-autosummary:
	find $(SOURCEDIR) -type d -name "_autosummary" -exec rm -rf {} +

clean-except-examples: clean-dataset-gallery clean-external-examples clean-autosummary
	rm -rf $(BUILDDIR)/*
	rm -f errors.txt
	rm -f sphinx_warnings.txt
	rm -rf $(SOURCEDIR)/images/auto-generated
	rm -rf $(SOURCEDIR)/tags

clean: clean-except-examples
	rm -rf $(SOURCEDIR)/examples/

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

gen-dataset-gallery:
	@if [ ! -d "$(SOURCEDIR)/api/examples/dataset-gallery" ]; then \
		echo "Dataset gallery directory not found, running make_table.py..."; \
		mkdir -p "$(SOURCEDIR)/api/examples/dataset-gallery"; \
		cd "$(SOURCEDIR)" && python make_tables.py; \
	else \
		echo "Dataset gallery directory already exists, skipping generation."; \
		echo "Run clean-dataset-gallery to clear the gallery."; \
	fi

gen-external-examples:
	@if [ ! -f "$(SOURCEDIR)/getting-started/external_examples.rst" ]; then \
		echo "external_examples.rst not found, running make_external_examples.py..."; \
		cd "$(SOURCEDIR)" && python make_external_gallery.py; \
	else \
		echo "The file external_examples.rst already exists, skipping generation."; \
		echo "Run clean-external-examples to clear the file."; \
	fi

# Spin up a local http server to view the rendered documentation.
# This is required for interactive examples to work.
serve-html:
	python -m http.server 11000 --directory "$(BUILDDIR)"/html
