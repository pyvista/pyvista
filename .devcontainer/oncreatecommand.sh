#!/usr/bin/env bash

python -m pip install --upgrade pip
python -m pip install -e . --group dev --no-cache-dir
pre-commit install --install-hooks
