#!/usr/bin/env bash

python -m pip install -e . --group dev --no-cache-dir
pre-commit install --install-hooks

pip uninstall vtk -y
pip install --extra-index-url https://wheels.vtk.org vtk-osmesa
