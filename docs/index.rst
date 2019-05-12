.. title:: pyvista

.. raw:: html

    <div class="banner">
        <img src="_static/pyvista_logo.png" alt="pyvista" width="500px">
        <h2>A Streamlined Python Interface for the Visualization Toolkit</h2>
    </div>


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

.. |contributors| image:: https://img.shields.io/github/contributors/pyvista/pyvista.svg?logo=github&logoColor=white
   :target: https://github.com/pyvista/pyvista/graphs/contributors/

.. |stars| image:: https://img.shields.io/github/stars/pyvista/pyvista.svg?style=social&label=Stars
   :target: https://github.com/pyvista/pyvista
   :alt: GitHub


+----------------------+------------------------+
| Deployment           | |pypi| |conda|         |
+----------------------+------------------------+
| Build Status         | |travis| |appveyor|    |
+----------------------+------------------------+
| Metrics              | |codacy| |codecov|     |
+----------------------+------------------------+
| GitHub               | |contributors| |stars| |
+----------------------+------------------------+


About
*****

PyVista is a helper module for the Visualization Toolkit (VTK) that takes a
different approach on interfacing with VTK through NumPy and direct array
access. This package provides a Pythonic, well-documented interface exposing
VTK's powerful visualization backend to facilitate rapid prototyping, analysis,
and visual integration of spatially referenced datasets.

This module can be used for scientific plotting for presentations and research
papers as well as a supporting module for other mesh dependent Python modules.


Want to test-drive PyVista? Check out our live examples on MyBinder:

.. image:: https://mybinder.org/badge_logo.svg
   :target: https://mybinder.org/v2/gh/pyvista/pyvista-examples/master
   :alt: Launch on Binder

.. toctree::
   :hidden:

   self
   why
   authors

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





Getting Started
***************

If you have a working copy of VTK, installation is simply::

    $ pip install pyvista

You can also visit `PyPi <http://pypi.python.org/pypi/pyvista>`_ or
`GitHub <https://github.com/pyvista/pyvista>`_ to download the source.

See :ref:`install_ref` for more details.


.. toctree::
   :maxdepth: 2
   :caption: Getting Started
   :hidden:

   getting-started/installation
   getting-started/simple



Examples
********

Be sure to head over to the `examples gallery <./examples/index.html>`_
to explore different use cases of PyVista and to start visualizing 3D data in
Pyhton! Also, please explore the list of external projects leveraging PyVista
for 3D visualization in our `external examples list <./external_examples.html>`_


.. toctree::
   :maxdepth: 2
   :caption: Examples
   :hidden:

   examples/index
   external_examples



Data Types
**********

The `Visualization Toolkit`_ (VTK), developed by Kitware_, has many mesh data
types that PyVista wraps.
This chapter is intended to describe these different mesh types on the VTK
side to help new users understand which data types to use.

.. _Visualization Toolkit: https://vtk.org
.. _Kitware: https://www.kitware.com

.. toctree::
   :maxdepth: 2
   :caption: Data Types
   :hidden:

   types/common
   types/points
   types/point-grids
   types/grids
   types/container



Tools
*****

* Pythonic interface to VTK's Python-C++ bindings
* Filtering/plotting tools built for interactivity in Jupyter notebooks (see :ref:`ipy_tools_ref`)
* Direct access to common VTK filters (see :ref:`filters_ref`)
* Intuitive plotting routines with ``matplotlib`` similar syntax (see :ref:`plotting_ref`)

.. toctree::
   :maxdepth: 2
   :caption: Tools
   :hidden:

   tools/geometric
   tools/plotting
   tools/filters
   tools/ipy_tools
   tools/qt_plotting
   tools/utilities



Contributing
************

We absolutely welcome contributions and we hope that this guide will facilitate
an understanding of the PyVista code repository. It is important to note that
the  PyVista software package is maintained on a volunteer basis and thus we
need to foster a community that can support user questions and develop new
features to make this software a useful tool for all users. To learn more about
contributing to PyVista, please see :ref:`contributing_ref`.

.. toctree::
   :maxdepth: 2
   :caption: Contributing
   :hidden:

   dev/contributing.rst
   dev/guidelines.rst
   dev/testing.rst




Project Index
*************

* :ref:`genindex`
