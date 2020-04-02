.. title:: PyVista

.. raw:: html

    <div class="banner">
        <img src="_static/pyvista_logo.png" alt="pyvista" width="500px">
        <h2>3D plotting and mesh analysis through a streamlined interface for the Visualization Toolkit (VTK)</h2>
    </div>


.. |pypi| image:: https://img.shields.io/pypi/v/pyvista.svg?logo=python&logoColor=white
   :target: https://pypi.org/project/pyvista/

.. |conda| image:: https://img.shields.io/conda/vn/conda-forge/pyvista.svg?logo=conda-forge&logoColor=white
   :target: https://anaconda.org/conda-forge/pyvista

.. |azure| image:: https://dev.azure.com/pyvista/PyVista/_apis/build/status/pyvista.pyvista?branchName=master
   :target: https://dev.azure.com/pyvista/PyVista/_build/latest?definitionId=3&branchName=master

.. |codecov| image:: https://codecov.io/gh/pyvista/pyvista/branch/master/graph/badge.svg
   :target: https://codecov.io/gh/pyvista/pyvista

.. |codacy| image:: https://api.codacy.com/project/badge/Grade/e927f0afec7e4b51aeb7785847d0fd47
   :target: https://www.codacy.com/app/banesullivan/pyvista?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=akaszynski/pyvista&amp;utm_campaign=Badge_Grade

.. |contributors| image:: https://img.shields.io/github/contributors/pyvista/pyvista.svg?logo=github&logoColor=white
   :target: https://github.com/pyvista/pyvista/graphs/contributors/

.. |stars| image:: https://img.shields.io/github/stars/pyvista/pyvista.svg?style=social&label=Stars
   :target: https://github.com/pyvista/pyvista
   :alt: GitHub

.. |zenodo| image:: https://zenodo.org/badge/92974124.svg
   :target: https://zenodo.org/badge/latestdoi/92974124

.. |joss| image:: https://joss.theoj.org/papers/78f2901bbdfbd2a6070ec41e8282d978/status.svg
   :target: https://joss.theoj.org/papers/10.21105/joss.01450

.. |MIT| image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :target: https://opensource.org/licenses/MIT

.. |slack| image:: https://img.shields.io/badge/Slack-PyVista-green.svg?logo=slack
   :target: http://slack.pyvista.org

.. |gitter| image:: https://img.shields.io/gitter/room/pyvista/community?color=darkviolet
   :target: https://gitter.im/pyvista/community


+----------------------+------------------------+
| Deployment           | |pypi| |conda|         |
+----------------------+------------------------+
| Build Status         | |azure|                |
+----------------------+------------------------+
| Metrics              | |codacy| |codecov|     |
+----------------------+------------------------+
| GitHub               | |contributors| |stars| |
+----------------------+------------------------+
| Citation             | |joss| |zenodo|        |
+----------------------+------------------------+
| License              | |MIT|                  |
+----------------------+------------------------+
| Community            | |slack| |gitter|       |
+----------------------+------------------------+


About
*****

PyVista is...

* *"VTK for humans"*: a high-level API to the `Visualization Toolkit`_ (VTK)
* mesh data structures and filtering methods for spatial datasets
* 3D plotting made simple and built for large/complex data geometries

.. _Visualization Toolkit: https://vtk.org


PyVista (formerly ``vtki``) is a helper module for the Visualization Toolkit
(VTK) that takes a different approach on interfacing with VTK through NumPy and
direct array access.
This package provides a Pythonic, well-documented interface exposing
VTK's powerful visualization backend to facilitate rapid prototyping, analysis,
and visual integration of spatially referenced datasets.

This module can be used for scientific plotting for presentations and research
papers as well as a supporting module for other mesh dependent Python modules.

.. |tweet| image:: https://img.shields.io/twitter/url.svg?style=social&url=http%3A%2F%2Fshields.io
   :target: https://twitter.com/intent/tweet?text=Check%20out%20this%20project%20for%203D%20visualization%20in%20Python&url=https://github.com/pyvista/pyvista&hashtags=3D,visualization,Python,vtk,mesh,plotting,PyVista

Share this project on Twitter: |tweet|


.. |binder| image:: https://static.mybinder.org/badge_logo.svg
   :target: https://mybinder.org/v2/gh/pyvista/pyvista-examples/master
   :alt: Launch on Binder

Want to test-drive PyVista? Check out our live examples on MyBinder: |binder|


.. toctree::
   :hidden:

   self
   why
   authors


Support
=======

For general questions about the project, its applications, or about software
usage, please create an issue in the `pyvista/pyvista-support`_ repository
where the community can collectively address your questions. You are also
welcome to join us on Slack_ or send one of the developers an email.
The project support team can be reached at `info@pyvista.org`_.

.. _pyvista/pyvista-support: https://github.com/pyvista/pyvista-support
.. _Slack: http://slack.pyvista.org
.. _info@pyvista.org: mailto:info@pyvista.org


Connections
===========

PyVista is a powerful tool that researchers can harness to create compelling,
integrated visualizations of large datasets in an intuitive, Pythonic manner.
Here are a few open-source projects that leverage PyVista:

* itkwidgets_: Interactive Jupyter widgets to visualize images, point sets, and meshes in 2D and 3D. Supports all PyVista mesh types.
* pyansys_: Pythonic interface to ANSYS result, full, and archive files
* PVGeo_: Python package of VTK-based algorithms to analyze geoscientific data and models. PyVista is used to make the inputs and outputs of PVGeo's algorithms more accessible.
* omfvista_: 3D visualization for the Open Mining Format (omf). PyVista provides the foundation for this library's visualization.
* discretize_: Discretization tools for finite volume and inverse problems. ``discretize`` provides ``toVTK`` methods that return PyVista versions of their data types for `creating compelling visualizations`_.
* pymeshfix_: Python/Cython wrapper of Marco Attene's wonderful, award-winning MeshFix software.
* tetgen_: Python Interface to Hang Si's C++ TetGen Library


.. _itkwidgets: https://github.com/InsightSoftwareConsortium/itkwidgets
.. _pyansys: https://github.com/akaszynski/pyansys
.. _PVGeo: https://github.com/OpenGeoVis/PVGeo
.. _omfvista: https://github.com/OpenGeoVis/omfvista
.. _discretize: http://discretize.simpeg.xyz/en/master/
.. _creating compelling visualizations: http://discretize.simpeg.xyz/en/master/api/generated/discretize.mixins.vtkModule.html
.. _pymeshfix: https://github.com/pyvista/pymeshfix
.. _MeshFix: https://github.com/MarcoAttene/MeshFix-V2.1
.. _tetgen: https://github.com/pyvista/tetgen


Citing PyVista
==============

There is a `paper about PyVista <https://doi.org/10.21105/joss.01450>`_!

If you are using PyVista in your scientific research, please help our scientific
visibility by citing our work! Head over to :ref:`citation_ref` to learn more
about citing PyVista.


Getting Started
***************

If you have a working copy of VTK, installation is simply::

    $ pip install pyvista

You can also visit `PyPi <https://pypi.org/project/pyvista/>`_ or
`GitHub <https://github.com/pyvista/pyvista>`_ to download the source.

See :ref:`install_ref` for more details.


Be sure to head over to the `examples gallery <./examples/index.html>`_
to explore different use cases of PyVista and to start visualizing 3D data in
Python! Also, please explore the list of external projects leveraging PyVista
for 3D visualization in our `external examples list <./external_examples.html>`_



.. toctree::
   :maxdepth: 2
   :caption: Getting Started
   :hidden:

   getting-started/installation
   getting-started/what-is-a-mesh
   getting-started/simple
   examples/index
   external_examples


API Reference
*************

In this section, you can learn more about how PyVista wraps different VTK mesh
types and how you can leverage powerful 3D plotting and mesh analysis tools.
Highlights of the API include:

* Pythonic interface to VTK's Python-C++ bindings
* Filtering/plotting tools built for interactivity (see :ref:`widgets`)
* Direct access to common VTK filters (see :ref:`filters_ref`)
* Intuitive plotting routines with ``matplotlib`` similar syntax (see :ref:`plotting_ref`)

.. toctree::
   :maxdepth: 2
   :caption: API Reference
   :hidden:

   core/index
   plotting/index
   utilities/index




Project Index
*************

* :ref:`genindex`
