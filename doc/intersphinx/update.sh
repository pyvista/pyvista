#!/bin/bash

# this script updates the intersphinx files here
curl https://docs.python.org/dev/objects.inv > python-objects.inv
curl https://docs.scipy.org/doc/scipy/reference/objects.inv > scipy-objects.inv
curl https://numpy.org/doc/stable/objects.inv > numpy-objects.inv
curl https://matplotlib.org/stable/objects.inv > matplotlib-objects.inv
curl https://imageio.readthedocs.io/en/stable/objects.inv > imageio-objects.inv
curl https://pandas.pydata.org/pandas-docs/stable/objects.inv > pandas-objects.inv
curl https://docs.pytest.org/en/stable/objects.inv > pytest-objects.inv
curl https://qtdocs.pyvista.org/objects.inv > pyvistaqt-objects.inv
