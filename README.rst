#######
PyVista
#######

    3D plotting and mesh analysis through a streamlined interface for the Visualization Toolkit (VTK)

.. image:: https://github.com/pyvista/pyvista/raw/main/doc/source/_static/pyvista_banner_small.png
   :target: https://docs.pyvista.org/examples/index.html
   :alt: pyvista

PyVista is:

* *Pythonic VTK*: a high-level API to the `Visualization Toolkit`_ (VTK)
* mesh data structures and filtering methods for spatial datasets
* 3D plotting made simple and built for large/complex data geometries

.. _Visualization Toolkit: https://vtk.org

.. image:: https://github.com/pyvista/pyvista/raw/main/assets/pyvista_ipython_demo.gif
   :alt: pyvista ipython demo

PyVista is a helper module for the Visualization Toolkit (VTK) that wraps the VTK library
through NumPy and direct array access through a variety of methods and classes.
This package provides a Pythonic, well-documented interface exposing
VTK's powerful visualization backend to facilitate rapid prototyping, analysis,
and visual integration of spatially referenced datasets.

This module can be used for scientific plotting for presentations and research
papers as well as a supporting module for other mesh 3D rendering dependent
Python modules; see Connections for a list of projects that leverage
PyVista.

PyVista is a NumFOCUS affiliated project

.. image:: https://raw.githubusercontent.com/numfocus/templates/master/images/numfocus-logo.png
   :target: https://numfocus.org/sponsored-projects/affiliated-projects
   :alt: NumFOCUS affiliated projects
   :height: 60px

Status badges
=============

.. |zenodo| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.8415866.svg
   :target: https://zenodo.org/records/8415866

.. |joss| image:: http://joss.theoj.org/papers/10.21105/joss.01450/status.svg
   :target: https://doi.org/10.21105/joss.01450

.. |pypi| image:: https://img.shields.io/pypi/v/pyvista.svg?logo=python&logoColor=white
   :target: https://pypi.org/project/pyvista/

.. |conda| image:: https://img.shields.io/conda/vn/conda-forge/pyvista.svg?logo=conda-forge&logoColor=white
   :target: https://anaconda.org/conda-forge/pyvista

.. |GH-CI| image:: https://github.com/pyvista/pyvista/actions/workflows/testing-and-deployment.yml/badge.svg
   :target: https://github.com/pyvista/pyvista/actions/workflows/testing-and-deployment.yml

.. |codecov| image:: https://codecov.io/gh/pyvista/pyvista/branch/main/graph/badge.svg
   :target: https://app.codecov.io/gh/pyvista/pyvista

.. |codacy| image:: https://app.codacy.com/project/badge/Grade/779ac6aed37548839384acfc0c1aab44
   :target: https://app.codacy.com/gh/pyvista/pyvista/dashboard

.. |MIT| image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :target: https://opensource.org/license/mit/

.. |slack| image:: https://img.shields.io/badge/Slack-pyvista-green.svg?logo=slack
   :target: https://communityinviter.com/apps/pyvista/pyvista

.. |PyPIact| image:: https://img.shields.io/pypi/dm/pyvista.svg?label=PyPI%20downloads
   :target: https://pypi.org/project/pyvista/

.. |condaact| image:: https://img.shields.io/conda/dn/conda-forge/pyvista.svg?label=Conda%20downloads
   :target: https://anaconda.org/conda-forge/pyvista

.. |discuss| image:: https://img.shields.io/badge/GitHub-Discussions-green?logo=github
   :target: https://github.com/pyvista/pyvista/discussions

.. |prettier| image:: https://img.shields.io/badge/code_style-prettier-ff69b4.svg?style=flat
  :target: https://github.com/prettier/prettier
  :alt: prettier

.. |python| image:: https://img.shields.io/badge/python-3.10+-blue.svg
   :target: https://www.python.org/downloads/

.. |NumFOCUS Affiliated| image:: https://img.shields.io/badge/affiliated-NumFOCUS-orange.svg?style=flat&colorA=E1523D&colorB=007D8A
   :target: https://numfocus.org/sponsored-projects/affiliated-projects

.. |pre-commit.ci status| image:: https://results.pre-commit.ci/badge/github/pyvista/pyvista/main.svg
   :target: https://results.pre-commit.ci/latest/github/pyvista/pyvista/main

.. |Ruff| image:: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json
   :target: https://github.com/astral-sh/ruff
   :alt: Ruff

.. |Awesome Scientific Computing| image:: https://awesome.re/mentioned-badge.svg
   :target: https://github.com/nschloe/awesome-scientific-computing

.. |Packaging status| image:: https://repology.org/badge/tiny-repos/python:pyvista.svg
   :target: https://repology.org/project/python:pyvista/versions

.. |Good first issue| image:: https://img.shields.io/github/issues/pyvista/pyvista/good%20first%20issue
   :target: https://github.com/pyvista/pyvista/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22

.. |GitHub Repo stars| image:: https://img.shields.io/github/stars/pyvista/pyvista
   :target: https://github.com/pyvista/pyvista/stargazers

.. |pyversions| image:: https://img.shields.io/pypi/pyversions/pyvista.svg?color=orange&logo=python&label=python&logoColor=white
    :target: https://pypi.org/project/pyvista
    :alt: Python versions

+----------------------+------------------------------------------------+
| Deployment           | |pypi| |pyversions| |conda| |Packaging status| |
+----------------------+------------------------------------------------+
| Build Status         | |GH-CI| |python| |pre-commit.ci status|        |
+----------------------+------------------------------------------------+
| Metrics              | |codacy| |codecov|                             |
+----------------------+------------------------------------------------+
| Activity             | |PyPIact| |condaact|                           |
+----------------------+------------------------------------------------+
| Citation             | |joss| |zenodo|                                |
+----------------------+------------------------------------------------+
| License              | |MIT|                                          |
+----------------------+------------------------------------------------+
| Community            | |slack| |discuss| |Good first issue|           |
|                      | |GitHub Repo stars|                            |
+----------------------+------------------------------------------------+
| Formatter            | |prettier|                                     |
+----------------------+------------------------------------------------+
| Linter               | |Ruff|                                         |
+----------------------+------------------------------------------------+
| Affiliated           | |NumFOCUS Affiliated|                          |
+----------------------+------------------------------------------------+
| Mentioned            | |Awesome Scientific Computing|                 |
+----------------------+------------------------------------------------+


Highlights
==========

.. |binder| image:: https://static.mybinder.org/badge_logo.svg
   :target: https://mybinder.org/v2/gh/pyvista/pyvista-examples/master
   :alt: Launch on Binder

Head over to the `Quick Examples`_ page in the docs to explore our gallery of
examples showcasing what PyVista can do. Want to test-drive PyVista?
All of the examples from the gallery are live on MyBinder for you to test
drive without installing anything locally: |binder|

.. _Quick Examples: http://docs.pyvista.org/examples/index.html


Overview of Features
--------------------

* Extensive gallery of examples (see `Quick Examples`_)
* Interactive plotting in Jupyter Notebooks with server-side and client-side
  rendering with `trame`_.
* Filtering/plotting tools built for interactivity (see `Widgets`_)
* Direct access to mesh analysis and transformation routines (see Filters_)
* Intuitive plotting routines with ``matplotlib`` similar syntax (see Plotting_)
* Import meshes from many common formats (use ``pyvista.read()``). Support for all formats handled by `meshio`_ is built-in.
* Export meshes as VTK, STL, OBJ, or PLY (``mesh.save()``) file types or any formats supported by meshio_ (``pyvista.save_meshio()``)

.. _trame: https://github.com/Kitware/trame
.. _Widgets: https://docs.pyvista.org/api/plotting/index.html#widget-api
.. _Filters: https://docs.pyvista.org/api/core/filters.html
.. _Plotting: https://docs.pyvista.org/api/plotting/index.html
.. _meshio: https://github.com/nschloe/meshio


Documentation
=============

Refer to the `documentation <http://docs.pyvista.org/>`_ for detailed
installation and usage details.

For general questions about the project, its applications, or about software
usage, please create a discussion in `pyvista/discussions`_
where the community can collectively address your questions. You are also
welcome to join us on Slack_.

.. _pyvista/discussions: https://github.com/pyvista/pyvista/discussions
.. _Slack: https://communityinviter.com/apps/pyvista/pyvista


Installation
============

PyVista can be installed from `PyPI <https://pypi.org/project/pyvista/>`_
using ``pip`` on Python >= 3.10::

    pip install pyvista

You can also visit `PyPI <https://pypi.org/project/pyvista/>`_,
`Anaconda <https://anaconda.org/conda-forge/pyvista>`_, or
`GitHub <https://github.com/pyvista/pyvista>`_ to download the source.

See the `Installation <http://docs.pyvista.org/getting-started/installation.html#install-ref.>`_
for more details regarding optional dependencies or if the installation through pip doesn't work out.


Connections
===========

PyVista is a powerful tool that researchers can harness to create compelling,
integrated visualizations of large datasets in an intuitive, Pythonic manner.

Learn more about how PyVista is used across science and engineering disciplines
by a diverse community of users on our `Connections page`_.

.. _Connections page: https://docs.pyvista.org/getting-started/connections.html


Authors
=======

.. |contrib.rocks| image:: https://contrib.rocks/image?repo=pyvista/pyvista
   :target: https://github.com/pyvista/pyvista/graphs/contributors
   :alt: contrib.rocks

Please take a look at the `contributors page`_ and the active `list of authors`_
to learn more about the developers of PyVista.

|contrib.rocks|

Made with `contrib rocks`_.

.. _contributors page: https://github.com/pyvista/pyvista/graphs/contributors/
.. _list of authors: https://docs.pyvista.org/getting-started/authors.html#authors
.. _contrib rocks: https://contrib.rocks


Contributing
============

.. |Contributor Covenant| image:: https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg
   :target: CODE_OF_CONDUCT.md

.. |codetriage| image:: https://www.codetriage.com/pyvista/pyvista/badges/users.svg
   :target: https://www.codetriage.com/pyvista/pyvista
   :alt: Code Triage

.. |Open in GitHub Codespaces| image:: https://github.com/codespaces/badge.svg
   :target: https://codespaces.new/pyvista/pyvista
   :alt: Open in GitHub Codespaces

|Contributor Covenant|
|codetriage|
|Open in GitHub Codespaces|

We absolutely welcome contributions and we hope that our `Contributing Guide`_
will facilitate your ability to make PyVista better. PyVista is mostly
maintained on a volunteer basis and thus we need to foster a community that can
support user questions and develop new features to make this software a useful
tool for all users while encouraging every member of the community to share
their ideas. To learn more about contributing to PyVista, please see the
`Contributing Guide`_ and our `Code of Conduct`_.

.. _Contributing Guide: https://github.com/pyvista/pyvista/blob/main/CONTRIBUTING.rst
.. _Code of Conduct: https://github.com/pyvista/pyvista/blob/main/CODE_OF_CONDUCT.md

Star History
============

.. image:: https://api.star-history.com/svg?repos=pyvista/pyvista&type=Date
   :alt: Star History Chart
   :target: https://star-history.com/#pyvista/pyvista&Date

Citing PyVista
==============

There is a `paper about PyVista <https://doi.org/10.21105/joss.01450>`_.

If you are using PyVista in your scientific research, please help our scientific
visibility by citing our work.


    Sullivan and Kaszynski, (2019). PyVista: 3D plotting and mesh analysis through a streamlined interface for the Visualization Toolkit (VTK). Journal of Open Source Software, 4(37), 1450, https://doi.org/10.21105/joss.01450


BibTex:

.. code:: latex

    @article{sullivan2019pyvista,
      doi = {10.21105/joss.01450},
      url = {https://doi.org/10.21105/joss.01450},
      year = {2019},
      month = {May},
      publisher = {The Open Journal},
      volume = {4},
      number = {37},
      pages = {1450},
      author = {Bane Sullivan and Alexander Kaszynski},
      title = {{PyVista}: {3D} plotting and mesh analysis through a streamlined interface for the {Visualization Toolkit} ({VTK})},
      journal = {Journal of Open Source Software}
    }

Professional Support
====================

While PyVista is an Open Source project with a big community, you might be looking for professional support.
This section aims to list companies with VTK/PyVista expertise who can help you with your software project.

+---------------+-----------------------------------------+
| Company Name  | Kitware Inc.                            |
+---------------+-----------------------------------------+
| Description   | Kitware is dedicated to build solutions |
|               | for our customers based on our          |
|               | well-established open source platforms. |
+---------------+-----------------------------------------+
| Expertise     | CMake, VTK, PyVista, ParaView, Trame    |
+---------------+-----------------------------------------+
| Contact       | https://www.kitware.com/contact/        |
+---------------+-----------------------------------------+
