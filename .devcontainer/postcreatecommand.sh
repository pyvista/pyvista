#!/usr/bin/env bash

python -m pip install -r requirements.txt
python -m pip install -r requirements_test.py

python -m pip install pre-commit
pre-commit install

python -m pip install -e .
