#!/bin/bash

# this script updates the intersphinx files here
# make sure to follow potential redirects, and fail on an HTTP error
# rather than saving the error page as an inventory
set -euo pipefail
curl -fsSL https://docs.python.org/3/objects.inv >python-objects.inv
curl -fsSL https://docs.scipy.org/doc/scipy/objects.inv >scipy-objects.inv
curl -fsSL https://numpy.org/doc/stable/objects.inv >numpy-objects.inv
curl -fsSL https://matplotlib.org/stable/objects.inv >matplotlib-objects.inv
curl -fsSL https://imageio.readthedocs.io/en/stable/objects.inv >imageio-objects.inv
curl -fsSL https://pandas.pydata.org/pandas-docs/stable/objects.inv >pandas-objects.inv
curl -fsSL https://arrow.apache.org/docs/objects.inv >pyarrow-objects.inv
curl -fsSL https://docs.pytest.org/en/stable/objects.inv >pytest-objects.inv
curl -fsSL https://qt.pyvista.org/objects.inv >pyvistaqt-objects.inv
curl -fsSL https://validation.pyvista.org/objects.inv >pyvista-validation-objects.inv
curl -fsSL https://trimesh.org/objects.inv >trimesh-objects.inv
