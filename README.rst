vtki
****

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


+----------------------+------------------------+
| Deployment           | |pypi| |conda|         |
+----------------------+------------------------+
| Build Status         | |travis| |appveyor|    |
+----------------------+------------------------+
| Metrics              | |codacy| |codecov|     |
+----------------------+------------------------+


``vtki`` is a helper module for the Visualization Toolkit (VTK) that takes a
different approach on interfacing with VTK through NumPy and direct array
access. This package provides a Pythonic, well-documented interface exposing
VTK's powerful visualization backend to facilitate rapid prototyping, analysis,
and visual integration of spatially referenced datasets.

This module can be used for scientific plotting for presentations and research
papers as well as a supporting module for other mesh dependent Python modules.


Documentation
=============
Refer to the `documentation <http://docs.vtki.org/>`_ for detailed
installation and usage details.

For general questions about the project, its applications, or about software
usage, please do not create an issue but join us on Slack_ or send one
of the developers an email. The project support team can be reached at
`info@vtki.org`_.

.. _Slack: http://slack.opengeovis.org
.. _info@vtki.org: mailto:info@vtki.org


Installation
============
Installation is simply::

    pip install vtki

You can also visit `PyPi <http://pypi.python.org/pypi/vtki>`_,
`Anaconda <https://anaconda.org/conda-forge/vtki>`_, or
`GitHub <https://github.com/vtkiorg/vtki>`_ to download the source.

See the `Installation <http://docs.vtki.org/getting-started/installation.html#install-ref.>`_
for more details if the installation through pip doesn't work out.


Highlights
==========

Head over to the `Quick Examples`_ page in the docs to learn more about using
``vtki``.

.. _Quick Examples: http://docs.vtki.org/examples/index.html

Want to test-drive ``vtki``? Check out our live examples on MyBinder:

.. image:: https://mybinder.org/badge_logo.svg
   :target: https://mybinder.org/v2/gh/vtkiorg/vtki-examples/master
   :alt: Launch on Binder


* Pythonic interface to VTK's Python-C++ bindings
* Filtering/plotting tools built for interactivity in Jupyter notebooks (see `IPython Tools`_)
* Direct access to common VTK filters (see Filters_)
* Intuitive plotting routines with ``matplotlib`` similar syntax (see Plotting_)


.. _IPython Tools: http://docs.vtki.org/tools/ipy_tools.html
.. _Filters: http://docs.vtki.org/tools/filters.html
.. _Plotting: http://docs.vtki.org/tools/plotting.html


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


Authors
=======

Please take a look at the `contributors page`_ and the active `list of authors`_
to learn more about the developers of ``vtki``.

.. _contributors page: https://GitHub.com/akaszynski/vtki/graphs/contributors/
.. _list of authors: http://docs.vtki.org/authors


Contributing
============

We absolutely welcome contributions and we hope that this guide will facilitate
an understanding of the ``vtki`` code repository. It is important to note that
the  ``vtki`` software package is maintained on a volunteer basis and thus we
need to foster a community that can support user questions and develop new
features to make this software a useful tool for all users. To learn more about
contributing to ``vtki``, please see the `Contributing Guide`_.

.. _Contributing Guide: http://docs.vtki.org/dev/contributing.html
