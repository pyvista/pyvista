#!/bin/bash

# this script updates the intersphinx files here
# make sure to follow potential redirects
curl -L https://docs.python.org/3/objects.inv > python-objects.inv
curl -L https://docs.scipy.org/doc/scipy/objects.inv > scipy-objects.inv
curl -L https://numpy.org/doc/stable/objects.inv > numpy-objects.inv
curl -L https://matplotlib.org/stable/objects.inv > matplotlib-objects.inv
curl -L https://imageio.readthedocs.io/en/stable/objects.inv > imageio-objects.inv
curl -L https://pandas.pydata.org/pandas-docs/stable/objects.inv > pandas-objects.inv
curl -L https://docs.pytest.org/en/stable/objects.inv > pytest-objects.inv
curl -L https://qtdocs.pyvista.org/objects.inv > pyvistaqt-objects.inv
curl -L https://trimsh.org/objects.inv > trimesh-objects.inv
