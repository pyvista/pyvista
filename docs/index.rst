####
vtki
####

`A Streamlined Python Interface for the Visualization Toolkit`


.. |pypi| image:: https://img.shields.io/pypi/v/vtki.svg?logo=python&logoColor=white
   :target: https://pypi.org/project/vtki/

.. |conda| image:: https://img.shields.io/conda/vn/conda-forge/vtki.svg
   :target: https://anaconda.org/conda-forge/vtki

.. |travis| image:: https://img.shields.io/travis/vtkiorg/vtki/master.svg?label=build&logo=travis
   :target: https://travis-ci.org/vtkiorg/vtki

.. |appveyor| image:: https://img.shields.io/appveyor/ci/banesullivan/vtki.svg?label=AppVeyor&style=flat&logo=appveyor
   :target: https://ci.appveyor.com/project/banesullivan/vtki/history

.. |codecov| image:: https://codecov.io/gh/akaszynski/vtki/branch/master/graph/badge.svg
   :target: https://codecov.io/gh/akaszynski/vtki

.. |codacy| image:: https://api.codacy.com/project/badge/Grade/e927f0afec7e4b51aeb7785847d0fd47
   :target: https://www.codacy.com/app/banesullivan/vtki?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=akaszynski/vtki&amp;utm_campaign=Badge_Grade

.. |contributors| image:: https://img.shields.io/github/contributors/vtkiorg/vtki.svg?logo=github&logoColor=white
   :target: https://github.com/vtkiorg/vtki/graphs/contributors/

.. |stars| image:: https://img.shields.io/github/stars/vtkiorg/vtki.svg?style=social&label=Stars
   :target: https://github.com/vtkiorg/vtki
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

``vtki`` is a helper module for the Visualization Toolkit (VTK) that takes a
different approach on interfacing with VTK through NumPy and direct array
access. This package provides a Pythonic, well-documented interface exposing
VTK's powerful visualization backend to facilitate rapid prototyping, analysis,
and visual integration of spatially referenced datasets.

This module can be used for scientific plotting for presentations and research
papers as well as a supporting module for other mesh dependent Python modules.


Want to test-drive ``vtki``? Check out our live examples on MyBinder:

.. image:: https://mybinder.org/badge_logo.svg
   :target: https://mybinder.org/v2/gh/vtkiorg/vtki-examples/master
   :alt: Launch on Binder

.. toctree::
   :hidden:

   self
   why
   authors

Connections
===========

``vtki`` is a powerful tool that researchers can harness to create compelling,
integrated visualizations of large datasets in an intuitive, Pythonic manner.
Here are a few open-source projects that leverage ``vtki``:

* pyansys_: Pythonic interface to ANSYS result, full, and archive files
* PVGeo_: Python package of VTK-based algorithms to analyze geoscientific data and models. ``vtki`` is used to make the inputs and outputs of PVGeo's algorithms more accessible.
* omfvtk_: 3D visualization for the Open Mining Format (omf). ``vtki`` provides the foundation for this library's visualization.
* discretize_: Discretization tools for finite volume and inverse problems. ``discretize`` provides ``toVTK`` methods that return ``vtki`` versions of their data types for `creating compelling visualizations`_.
* pymeshfix_: Python/Cython wrapper of Marco Attene's wonderful, award-winning MeshFix software.
* tetgen_: Python Interface to Hang Si's C++ TetGen Library


.. _pymeshfix: https://github.com/akaszynski/pymeshfix
.. _pyansys: https://github.com/akaszynski/pyansys
.. _PVGeo: https://github.com/OpenGeoVis/PVGeo
.. _omfvtk: https://github.com/OpenGeoVis/omfvtk
.. _discretize: http://discretize.simpeg.xyz/en/master/
.. _creating compelling visualizations: http://discretize.simpeg.xyz/en/master/content/mixins.html#module-discretize.mixins.vtkModule
.. _pymeshfix: https://github.com/akaszynski/pymeshfix
.. _MeshFix: https://github.com/MarcoAttene/MeshFix-V2.1
.. _tetgen: https://github.com/akaszynski/tetgen





Getting Started
***************

If you have a working copy of VTK, installation is simply::

    $ pip install vtki

You can also visit `PyPi <http://pypi.python.org/pypi/vtki>`_ or
`GitHub <https://github.com/vtkiorg/vtki>`_ to download the source.

See :ref:`install_ref` for more details.


.. toctree::
   :maxdepth: 2
   :caption: Getting Started
   :hidden:

   getting-started/installation
   getting-started/simple



Data Types
**********

The `Visualization Toolkit`_ (VTK), developed by Kitware_, has many mesh data
types that ``vtki`` wraps.
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


Examples
********

Be sure to head over to the `examples gallery <./examples/index.html>`_
to explore different use cases of ``vtki`` and to start visualizing 3D data in
Pyhton!


.. toctree::
   :maxdepth: 2
   :caption: Examples
   :hidden:

   examples/index
   external_examples


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
an understanding of the ``vtki`` code repository. It is important to note that
the  ``vtki`` software package is maintained on a volunteer basis and thus we
need to foster a community that can support user questions and develop new
features to make this software a useful tool for all users. To learn more about
contributing to ``vtki``, please see :ref:`contributing_ref`.

.. toctree::
   :maxdepth: 2
   :caption: Contributing
   :hidden:

   dev/contributing.rst
   dev/guidelines.rst
   dev/testing.rst



License
*******

.. include:: ../LICENSE


Project Index
*************

* :ref:`genindex`
