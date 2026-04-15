#!/usr/bin/env bash

python -m pip install -e . --group dev --no-cache-dir
pre-commit install --install-hooks

# Ensure vtk 9.5 is installed for best offscreen rendering support
python -m pip install 'vtk>=9.5'
