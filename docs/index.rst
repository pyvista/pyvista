#############
vtki Overview
#############


.. image:: https://img.shields.io/pypi/v/vtki.svg?logo=python&logoColor=white
   :target: https://pypi.org/project/vtki/

.. image:: https://img.shields.io/travis/akaszynski/vtki/master.svg?label=build&logo=travis
   :target: https://travis-ci.org/akaszynski/vtki

.. image:: https://img.shields.io/appveyor/ci/akaszynski/vtkinterface.svg?label=AppVeyor&style=flat&logo=appveyor
   :target: https://ci.appveyor.com/project/akaszynski/vtkinterface/history

.. image:: https://img.shields.io/github/contributors/akaszynski/vtki.svg?logo=github&logoColor=white
   :target: https://GitHub.com/akaszynski/vtki/graphs/contributors/

.. image:: https://img.shields.io/github/stars/akaszynski/vtki.svg?style=social&label=Stars
  :target: https://github.com/akaszynski/vtki
  :alt: GitHub

About
*****

``vtki`` is a helper module for the Visualization Toolkit (VTK) that takes a
different approach on interfacing with VTK through NumPy and direct array access.
This module simplifies mesh creation and plotting by adding functionality to
existing VTK objects.

This module can be used for scientific plotting for presentations and research
papers as well as a supporting module for other mesh dependent Python modules.

.. toctree::
   :hidden:

   self
   why

Connections
===========

``vtki`` is a powerful tool that researchers can harness to create compelling,
integrated visualizations of large datasets in an intuitive, Pythonic manner.
Here are a few open-source projects that leverage ``vtki``:

* PVGeo_: Python package of VTK-based algorithms to analyze geoscientific data and models
* omfvtk_: 3D visualization for the Open Mining Format (omf)


.. _PVGeo: https://github.com/OpenGeoVis/PVGeo
.. _omfvtk: https://github.com/OpenGeoVis/omfvtk




Getting Started
***************

If you have a working copy of VTK, installation is simply::

    $ pip install vtki

You can also visit `PyPi <http://pypi.python.org/pypi/vtki>`_ or
`GitHub <https://github.com/akaszynski/vtki>`_ to download the source.

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

Be sure to head over to the :ref:`examples_ref` chapter to explore different use
cases of ``vtki`` and to start visualizing 3D data in Pyhton!


.. toctree::
   :maxdepth: 2
   :caption: Examples
   :hidden:

   examples/index


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

   tools/plotting
   tools/filters
   tools/ipy_tools
   tools/qt_plotting
   tools/utilities



Project Index & Search
**********************

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
