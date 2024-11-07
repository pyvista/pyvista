#!/usr/bin/env bash

python -m pip install -e .[dev] --no-cache-dir
pre-commit install --install-hooks
