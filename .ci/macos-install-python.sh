#!/usr/bin/env bash

PYTHON_VERSION="$1"

case $PYTHON_VERSION in
2.7)
  FULL_VERSION=2.7.16
  ;;
3.6)
  FULL_VERSION=3.6.8
  ;;
3.7)
  FULL_VERSION=3.7.7
  ;;
3.8)
  FULL_VERSION=3.8.3
  ;;
esac

INSTALLER_NAME=python-$FULL_VERSION-macosx10.9.pkg
URL=https://www.python.org/ftp/python/$FULL_VERSION/$INSTALLER_NAME

PY_PREFIX=/Library/Frameworks/Python.framework/Versions

set -e -x

curl $URL > $INSTALLER_NAME

sudo installer -pkg $INSTALLER_NAME -target /

sudo rm /usr/local/bin/python
sudo ln -s /usr/local/bin/python$PYTHON_VERSION /usr/local/bin/python

which python
python --version
python -m ensurepip
python -m pip install --upgrade pip
python -m pip install setuptools twine wheel
