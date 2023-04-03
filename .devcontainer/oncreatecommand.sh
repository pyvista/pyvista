#!/usr/bin/env bash

python -m pip install -r requirements.txt
python -m pip install -r requirements_test.txt

python -m pip install pre-commit
pre-commit install --install-hooks

python -m pip install -e .
