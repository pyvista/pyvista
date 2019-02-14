vtki
====


.. image:: https://img.shields.io/pypi/v/vtki.svg?logo=python&logoColor=white
   :target: https://pypi.org/project/vtki/

.. image:: https://img.shields.io/travis/akaszynski/vtki/master.svg?label=build&logo=travis
   :target: https://travis-ci.org/akaszynski/vtki

.. image:: https://img.shields.io/appveyor/ci/akaszynski/vtkinterface.svg?label=AppVeyor&style=flat&logo=appveyor
   :target: https://ci.appveyor.com/project/akaszynski/vtkinterface/history

.. image:: https://img.shields.io/readthedocs/vtkinterface.svg?logo=read%20the%20docs&logoColor=white
   :target: http://www.vtki.org/

.. image:: https://img.shields.io/github/contributors/akaszynski/vtki.svg?logo=github&logoColor=white
   :target: https://GitHub.com/akaszynski/vtki/graphs/contributors/

.. image:: https://codecov.io/gh/akaszynski/vtki/branch/master/graph/badge.svg
  :target: https://codecov.io/gh/akaszynski/vtki


``vtki`` is a VTK helper module that takes a different approach on interfacing
with VTK through NumPy and direct array access. This module simplifies mesh
creation and plotting by adding functionality to existing VTK objects.

This module can be used for scientific plotting for presentations and research
papers as well as a supporting module for other mesh dependent Python modules.


Documentation
-------------
Refer to the `Read the Docs <http://www.vtki.org/>`_
documentation for detailed installation and usage details.

Also see the `wiki <https://github.com/akaszynski/vtki/wiki>`_ for brief code
snippets.

Installation
------------
Installation is simply::

    pip install vtki

You can also visit `PyPi <http://pypi.python.org/pypi/vtki>`_ or
`GitHub <https://github.com/akaszynski/vtki>`_ to download the source.

See the `Installation <http://www.vtki.org/en/latest/getting-started/installation.html#install-ref.>`_
for more details if the installation through pip doesn't work out.


Highlights
----------

Head over to the `Quick Examples`_ page in the docs to learn more about using
``vtki``.

.. _Quick Examples: http://www.vtki.org/en/latest/examples/index.html

* Pythonic interface to VTK's Python-C++ bindings
* Filtering/plotting tools built for interactivity in Jupyter notebooks (see `IPython Tools`_)
* Direct access to common VTK filters (see Filters_)
* Intuitive plotting routines with ``matplotlib`` similar syntax (see Plotting_)


.. _IPython Tools: http://www.vtki.org/en/latest/tools/ipy_tools.html
.. _Filters: http://www.vtki.org/en/latest/tools/filters.html
.. _Plotting: http://www.vtki.org/en/latest/tools/plotting.html


Connections
-----------

``vtki`` is a powerful tool that researchers can harness to create compelling,
integrated visualizations of large datasets in an intuitive, Pythonic manner.
Here are a few open-source projects that leverage ``vtki``:

* PVGeo_: Python package of VTK-based algorithms to analyze geoscientific data and models
* omfvtk_: 3D visualization for the Open Mining Format (omf)


.. _PVGeo: https://github.com/OpenGeoVis/PVGeo
.. _omfvtk: https://github.com/OpenGeoVis/omfvtk
