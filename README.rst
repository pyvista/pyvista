#######
PyVista
#######

.. image:: https://github.com/pyvista/pyvista/raw/main/doc/_static/pyvista_banner_small.png
   :target: https://docs.pyvista.org/examples/index.html
   :alt: pyvista


.. |zenodo| image:: https://zenodo.org/badge/92974124.svg
   :target: https://zenodo.org/badge/latestdoi/92974124

.. |joss| image:: http://joss.theoj.org/papers/10.21105/joss.01450/status.svg
   :target: https://doi.org/10.21105/joss.01450

.. |pypi| image:: https://img.shields.io/pypi/v/pyvista.svg?logo=python&logoColor=white
   :target: https://pypi.org/project/pyvista/

.. |conda| image:: https://img.shields.io/conda/vn/conda-forge/pyvista.svg?logo=conda-forge&logoColor=white
   :target: https://anaconda.org/conda-forge/pyvista

.. |GH-CI| image:: https://github.com/pyvista/pyvista/actions/workflows/testing-and-deployment.yml/badge.svg
   :target: https://github.com/pyvista/pyvista/actions/workflows/testing-and-deployment.yml

.. |codecov| image:: https://codecov.io/gh/pyvista/pyvista/branch/main/graph/badge.svg
   :target: https://codecov.io/gh/pyvista/pyvista

.. |codacy| image:: https://app.codacy.com/project/badge/Grade/779ac6aed37548839384acfc0c1aab44
   :target: https://www.codacy.com/gh/pyvista/pyvista/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=pyvista/pyvista&amp;utm_campaign=Badge_Grade

.. |MIT| image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :target: https://opensource.org/licenses/MIT

.. |slack| image:: https://img.shields.io/badge/Slack-pyvista-green.svg?logo=slack
   :target: http://slack.pyvista.org

.. |PyPIact| image:: https://img.shields.io/pypi/dm/pyvista.svg?label=PyPI%20downloads
   :target: https://pypi.org/project/pyvista/

.. |condaact| image:: https://img.shields.io/conda/dn/conda-forge/pyvista.svg?label=Conda%20downloads
   :target: https://anaconda.org/conda-forge/pyvista

.. |discuss| image:: https://img.shields.io/badge/GitHub-Discussions-green?logo=github
   :target: https://github.com/pyvista/pyvista/discussions

.. |isort| image:: https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat
  :target: https://timothycrosley.github.io/isort
  :alt: isort

.. |black| image:: https://img.shields.io/badge/code%20style-black-000000.svg?style=flat
  :target: https://github.com/psf/black
  :alt: black


+----------------------+-----------+------------+
| Deployment           | |pypi|    |   |conda|  |
+----------------------+-----------+------------+
| Build Status         | |GH-CI|                |
+----------------------+-----------+------------+
| Metrics              | |codacy|  |  |codecov| |
+----------------------+-----------+------------+
| Activity             | |PyPIact| | |condaact| |
+----------------------+-----------+------------+
| Citation             | |joss|    |  |zenodo|  |
+----------------------+-----------+------------+
| License              | |MIT|     |            |
+----------------------+-----------+------------+
| Community            | |slack|   |  |discuss| |
+----------------------+-----------+------------+
| Formatter            | |black|   |  |isort|   |
+----------------------+-----------+------------+


    3D plotting and mesh analysis through a streamlined interface for the Visualization Toolkit (VTK)

PyVista is...

* *Pythonic VTK*: a high-level API to the `Visualization Toolkit`_ (VTK)
* mesh data structures and filtering methods for spatial datasets
* 3D plotting made simple and built for large/complex data geometries

.. _Visualization Toolkit: https://vtk.org

PyVista is a helper module for the Visualization Toolkit (VTK) that wraps the VTK library
through NumPy and direct array access through a variety of methods and classes.
This package provides a Pythonic, well-documented interface exposing
VTK's powerful visualization backend to facilitate rapid prototyping, analysis,
and visual integration of spatially referenced datasets.

This module can be used for scientific plotting for presentations and research
papers as well as a supporting module for other mesh 3D rendering dependent
Python modules; see Connections for a list of projects that leverage
PyVista.


.. |tweet| image:: https://img.shields.io/twitter/url.svg?style=social&url=http%3A%2F%2Fshields.io
   :target: https://twitter.com/intent/tweet?text=Check%20out%20this%20project%20for%203D%20visualization%20in%20Python&url=https://github.com/pyvista/pyvista&hashtags=3D,visualization,Python,vtk,mesh,plotting,PyVista

Share this project on Twitter: |tweet|


Highlights
==========

.. |binder| image:: https://static.mybinder.org/badge_logo.svg
   :target: https://mybinder.org/v2/gh/pyvista/pyvista-examples/master
   :alt: Launch on Binder

Head over to the `Quick Examples`_ page in the docs to explore our gallery of
examples showcasing what PyVista can do! Want to test-drive PyVista?
All of the examples from the gallery are live on MyBinder for you to test
drive without installing anything locally: |binder|

.. _Quick Examples: http://docs.pyvista.org/examples/index.html


Overview of Features
--------------------

* Extensive gallery of examples (see `Quick Examples`_)
* Interactive plotting in Jupyter Notebooks using server-side rendering
  with `ipyvtklink`_ or client-side rendering with ``panel`` or ``ipygany``.
* Filtering/plotting tools built for interactivity (see `Widgets`_)
* Direct access to mesh analysis and transformation routines (see Filters_)
* Intuitive plotting routines with ``matplotlib`` similar syntax (see Plotting_)
* Import meshes from many common formats (use ``pyvista.read()``). Support for all formats handled by `meshio`_ is built-in!
* Export meshes as VTK, STL, OBJ, or PLY (``mesh.save()``) file types or any formats supported by meshio_ (``pyvista.save_meshio()``)

.. _ipyvtklink: https://github.com/Kitware/ipyvtklink
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
welcome to join us on Slack_ or send one of the developers an email.
The project support team can be reached at `info@pyvista.org`_.

.. _pyvista/discussions: https://github.com/pyvista/pyvista/discussions
.. _Slack: http://slack.pyvista.org
.. _info@pyvista.org: mailto:info@pyvista.org


Installation
============

PyVista can be installed from `PyPI <https://pypi.org/project/pyvista/>`_
using ``pip`` on Python >= 3.7::

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

Please take a look at the `contributors page`_ and the active `list of authors`_
to learn more about the developers of PyVista.

.. _contributors page: https://github.com/pyvista/pyvista/graphs/contributors/
.. _list of authors: https://docs.pyvista.org/getting-started/authors.html#authors


Contributing
============

.. |Contributor Covenant| image:: https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg
   :target: CODE_OF_CONDUCT.md

.. |codetriage| image:: https://www.codetriage.com/pyvista/pyvista/badges/users.svg
   :target: https://www.codetriage.com/pyvista/pyvista
   :alt: Code Triage

|Contributor Covenant|
|codetriage|

We absolutely welcome contributions and we hope that our `Contributing Guide`_
will facilitate your ability to make PyVista better. PyVista is mostly
maintained on a volunteer basis and thus we need to foster a community that can
support user questions and develop new features to make this software a useful
tool for all users while encouraging every member of the commutinity to share
their ideas. To learn more about contributing to PyVista, please see the
`Contributing Guide`_ and our `Code of Conduct`_.

.. _Contributing Guide: https://github.com/pyvista/pyvista/blob/main/CONTRIBUTING.rst
.. _Code of Conduct: https://github.com/pyvista/pyvista/blob/main/CODE_OF_CONDUCT.md


Citing PyVista
==============

There is a `paper about PyVista <https://doi.org/10.21105/joss.01450>`_!

If you are using PyVista in your scientific research, please help our scientific
visibility by citing our work!


    Sullivan and Kaszynski, (2019). PyVista: 3D plotting and mesh analysis through a streamlined interface for the Visualization Toolkit (VTK). Journal of Open Source Software, 4(37), 1450, https://doi.org/10.21105/joss.01450


BibTex:

.. code::

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
