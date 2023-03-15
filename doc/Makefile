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

clean:
	rm -rf $(BUILDDIR)/*
	rm -rf $(SOURCEDIR)/examples/
	rm -f errors.txt
	rm -f sphinx_warnings.txt
	rm -rf $(SOURCEDIR)/images/auto-generated
	rm -f $(SOURCEDIR)/getting-started/external_examples.rst
	find $(SOURCEDIR) -type d -name "_autosummary" -exec rm -rf {} +

clean-except-examples:
	rm -rf $(BUILDDIR)/*
	rm -f errors.txt
	rm -f sphinx_warnings.txt
	rm -rf $(SOURCEDIR)/images/auto-generated
	rm -f $(SOURCEDIR)/getting-started/external_examples.rst
	find $(SOURCEDIR) -type d -name "_autosummary" -exec rm -rf {} +

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

# deploy to https://github.com/pyvista/pyvista-docs
# WARNING: Use with care as this overwrites history of gh-pages
deploy:
	@echo "*** Warning ***"
	@echo "You are about to deploy to 'pyvista docs'."
	@echo "This overwrites the history of gh-pages."
	@echo "Are you sure? [y/N] " && read answer && [ $${answer:-N} = y ]
	@echo "Deploying..."
	cp README.md _build/html
	cp ads.txt _build/html
	touch _build/html/.nojekyll
	echo "docs.pyvista.org" >> _build/html/CNAME
	cd _build/html && git init
	cd _build/html && git add .
	cd _build/html && git checkout -b gh-pages
	cd _build/html && git commit -am "manual build"
	cd _build/html && git remote add origin https://github.com/pyvista/pyvista-docs
	cd _build/html && git push -u origin gh-pages --force
	rm -rf _build/html/.git

update-intersphinx:
	cd intersphinx && ./update.sh

# nitpicky linkcheck
linkcheck:
	@$(SPHINXBUILD) -b linkcheck -D nitpicky=1 -d _build/doctrees . _build/linkcheck

linkcheck-grep:
	@! grep -h "^.*:.*: \[\(\(local\)\|\(broken\)\)\]" _build/linkcheck/output.txt
