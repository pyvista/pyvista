# Minimal makefile for Sphinx documentation
#

export PYVISTA_OFF_SCREEN := True

# You can set these variables from the command line.
SPHINXOPTS    = -j auto
SPHINXBUILD   = python -msphinx
SPHINXPROJ    = pyvista
SOURCEDIR     = .
BUILDDIR      = _build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

.PHONY: all
all:
    echo PYVISTA_OFF_SCREEN: $$PYVISTA_OFF_SCREEN

clean:
	rm -rf $(BUILDDIR)/*
	rm -rf examples/
	rm -f errors.txt
	rm -rf images/auto-generated
	rm -f getting-started/external_examples.rst
	find . -type f -name "_autosummary" -delete
	find . -type f -name "api-autogen" -delete


# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# deploy to https://github.com/pyvista/pyvista-docs
# WARNING: Use with care as this overwrites history of gh-pages
deploy:
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
