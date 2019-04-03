#!/bin/bash
#
# Push HTML pages to the vtki-docs repository

if [[ $TRAVIS_PYTHON_VERSION == 3.7 ]]; then
    set -e;
    doctr deploy --built-docs docs/_build/html/ --deploy-repo vtkiorg/vtki-docs docs/;
fi
