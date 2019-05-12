#######
PyVista
#######

.. image:: https://github.com/pyvista/pyvista/raw/master/docs/_static/pyvista_logo.png
    :alt: pyvista


.. image:: https://zenodo.org/badge/92974124.svg
   :target: https://zenodo.org/badge/latestdoi/92974124


.. |pypi| image:: https://img.shields.io/pypi/v/pyvista.svg?logo=python&logoColor=white
   :target: https://pypi.org/project/pyvista/

.. |conda| image:: https://img.shields.io/conda/vn/conda-forge/pyvista.svg
   :target: https://anaconda.org/conda-forge/pyvista

.. |travis| image:: https://img.shields.io/travis/pyvista/pyvista/master.svg?label=build&logo=travis
   :target: https://travis-ci.org/pyvista/pyvista

.. |appveyor| image:: https://img.shields.io/appveyor/ci/banesullivan/pyvista.svg?label=AppVeyor&style=flat&logo=appveyor
   :target: https://ci.appveyor.com/project/banesullivan/pyvista/history

.. |codecov| image:: https://codecov.io/gh/akaszynski/pyvista/branch/master/graph/badge.svg
   :target: https://codecov.io/gh/akaszynski/pyvista

.. |codacy| image:: https://api.codacy.com/project/badge/Grade/e927f0afec7e4b51aeb7785847d0fd47
   :target: https://www.codacy.com/app/banesullivan/pyvista?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=akaszynski/pyvista&amp;utm_campaign=Badge_Grade


+----------------------+------------------------+
| Deployment           | |pypi| |conda|         |
+----------------------+------------------------+
| Build Status         | |travis| |appveyor|    |
+----------------------+------------------------+
| Metrics              | |codacy| |codecov|     |
+----------------------+------------------------+


PyVista is a helper module for the Visualization Toolkit (VTK) that takes a
different approach on interfacing with VTK through NumPy and direct array
access. This package provides a Pythonic, well-documented interface exposing
VTK's powerful visualization backend to facilitate rapid prototyping, analysis,
and visual integration of spatially referenced datasets.

This module can be used for scientific plotting for presentations and research
papers as well as a supporting module for other mesh dependent Python modules.


Documentation
=============

Refer to the `documentation <http://docs.pyvista.org/>`_ for detailed
installation and usage details.

For general questions about the project, its applications, or about software
usage, please do not create an issue but join us on Slack_ or send one
of the developers an email. The project support team can be reached at
`info@pyvista.org`_.

.. _Slack: http://slack.opengeovis.org
.. _info@pyvista.org: mailto:info@pyvista.org


Installation
============

PyVista can be installed from `PyPI <http://pypi.python.org/pypi/pyvista>`_
using ``pip`` on Python >= 3.5::

    pip install pyvista

You can also visit `PyPi <http://pypi.python.org/pypi/pyvista>`_,
`Anaconda <https://anaconda.org/conda-forge/pyvista>`_, or
`GitHub <https://github.com/pyvista/pyvista>`_ to download the source.

See the `Installation <http://docs.pyvista.org/getting-started/installation.html#install-ref.>`_
for more details if the installation through pip doesn't work out.


Highlights
==========

Head over to the `Quick Examples`_ page in the docs to learn more about using
PyVista.

.. _Quick Examples: http://docs.pyvista.org/examples/index.html

Want to test-drive PyVista? Check out our live examples on MyBinder:

.. image:: https://mybinder.org/badge_logo.svg
   :target: https://mybinder.org/v2/gh/pyvista/pyvista-examples/master
   :alt: Launch on Binder


* Pythonic interface to VTK's Python-C++ bindings
* Filtering/plotting tools built for interactivity in Jupyter notebooks (see `IPython Tools`_)
* Direct access to common VTK filters (see Filters_)
* Intuitive plotting routines with ``matplotlib`` similar syntax (see Plotting_)


.. _IPython Tools: http://docs.pyvista.org/tools/ipy_tools.html
.. _Filters: http://docs.pyvista.org/tools/filters.html
.. _Plotting: http://docs.pyvista.org/tools/plotting.html


Connections
===========

PyVista is a powerful tool that researchers can harness to create compelling,
integrated visualizations of large datasets in an intuitive, Pythonic manner.
Here are a few open-source projects that leverage PyVista:

* pyansys_: Pythonic interface to ANSYS result, full, and archive files
* PVGeo_: Python package of VTK-based algorithms to analyze geoscientific data and models. PyVista is used to make the inputs and outputs of PVGeo's algorithms more accessible.
* omfvista_: 3D visualization for the Open Mining Format (omf). PyVista provides the foundation for this library's visualization.
* discretize_: Discretization tools for finite volume and inverse problems. ``discretize`` provides ``toVTK`` methods that return PyVista versions of their data types for `creating compelling visualizations`_.
* pymeshfix_: Python/Cython wrapper of Marco Attene's wonderful, award-winning MeshFix software.
* tetgen_: Python Interface to Hang Si's C++ TetGen Library


.. _pymeshfix: https://github.com/akaszynski/pymeshfix
.. _pyansys: https://github.com/akaszynski/pyansys
.. _PVGeo: https://github.com/OpenGeoVis/PVGeo
.. _omfvista: https://github.com/OpenGeoVis/omfvista
.. _discretize: http://discretize.simpeg.xyz/en/master/
.. _creating compelling visualizations: http://discretize.simpeg.xyz/en/master/content/mixins.html#module-discretize.mixins.vtkModule
.. _pymeshfix: https://github.com/akaszynski/pymeshfix
.. _MeshFix: https://github.com/MarcoAttene/MeshFix-V2.1
.. _tetgen: https://github.com/akaszynski/tetgen


Authors
=======

Please take a look at the `contributors page`_ and the active `list of authors`_
to learn more about the developers of PyVista.

.. _contributors page: https://GitHub.com/pyvista/pyvista/graphs/contributors/
.. _list of authors: http://docs.pyvista.org/authors


Contributing
============

We absolutely welcome contributions and we hope that this guide will facilitate
an understanding of the PyVista code repository. It is important to note that
the  PyVista software package is maintained on a volunteer basis and thus we
need to foster a community that can support user questions and develop new
features to make this software a useful tool for all users. To learn more about
contributing to PyVista, please see the `Contributing Guide`_.

.. _Contributing Guide: http://docs.pyvista.org/dev/contributing.html
