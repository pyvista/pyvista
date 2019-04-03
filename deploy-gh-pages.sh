#!/bin/sh
#
# Push HTML pages to the vtki-docs repository

set -e;
doctr deploy --built-docs docs/_build/html/ --deploy-repo vtkiorg/vtki-docs docs/;
