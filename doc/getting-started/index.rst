Getting Started
***************

.. toctree::
   :hidden:

   why
   authors
   installation
   external_examples


.. panels::
    :column: col-lg-6 col-md-6 col-sm-12 col-xs-12 p-3

    Why PyVista?
    ^^^^^^^^^^^^

    Learn more about why we created PyVista as an interface to the Visualization Toolkit (VTK).

    .. link-button:: why
        :type: ref
        :text: Why PyVista?
        :classes: btn-outline-primary btn-block stretched-link


    ---

    Authors & Citation
    ^^^^^^^^^^^^^^^^^^

    Using PyVista in your research? Please consider citing or
    acknowledging us.  We have a publication!

    .. link-button:: authors
        :type: ref
        :text: Authors & Citation
        :classes: btn-outline-primary btn-block stretched-link

    ---
    :column: col-12 p-3

    See PyVista in External Efforts
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    Take a look at third party projects using PyVista

    .. link-button:: external_examples
        :type: ref
        :text: Learn more
        :classes: btn-outline-primary btn-block stretched-link


Installation
============

.. panels::
    :card: + install-card
    :column: col-lg-6 col-md-6 col-sm-12 col-xs-12 p-3

    Working with conda?
    ^^^^^^^^^^^^^^^^^^^

    PyVista is available on `conda-forge <https://anaconda.org/conda-forge/pyvista>`_

    ++++++++++++++++++++++

    .. code-block:: bash

        conda install -c conda-forge pyvista

    ---

    Prefer pip?
    ^^^^^^^^^^^

    PyVista can be installed via pip from `PyPI <https://pypi.org/project/pyvista>`__.

    ++++

    .. code-block:: bash

        pip install pyvista

    ---
    :column: col-12 p-3

    In-depth instructions?
    ^^^^^^^^^^^^^^^^^^^^^^

    Installing a specific version? Installing from source? Check the advanced
    installation page.

    .. link-button:: installation
        :type: ref
        :text: Installing PyVista
        :classes: btn-outline-primary btn-block stretched-link


Support
=======

For general questions about the project, its applications, or about software
usage, please create a discussion in `pyvista/discussions`_
where the community can collectively address your questions. You are also
welcome to join us on Slack_ or send one of the developers an email.
The project support team can be reached at `info@pyvista.org`_.

.. _pyvista/discussions: https://github.com/pyvista/pyvista/discussions
.. _Slack: http://slack.pyvista.org
.. _info@pyvista.org: mailto:info@pyvista.org


Connections
===========

PyVista is a powerful tool that researchers can harness to create compelling,
integrated visualizations of large datasets in an intuitive, Pythonic manner.
Here are a few open-source projects that leverage PyVista:

* PVGeo_: Python package of VTK-based algorithms to analyze geoscientific data and models. PyVista is used to make the inputs and outputs of PVGeo's algorithms more accessible.
* discretize_: Discretization tools for finite volume and inverse problems. ``discretize`` provides ``toVTK`` methods that return PyVista versions of their data types for `creating compelling visualizations`_.
* gemgis_: Spatial data processing for geomodeling.
* gimly_: Geophysical inversion and modeling library.
* itkwidgets_: Interactive Jupyter widgets to visualize images, point sets, and meshes in 2D and 3D. Supports all PyVista mesh types.
* mne-python_: MNE: Magnetoencephalography (MEG) and Electroencephalography (EEG) in Python.
* omfvista_: 3D visualization for the Open Mining Format (omf). PyVista provides the foundation for this library's visualization.
* pyleecan_: Electrical engineering open-source software providing a user-friendly, unified, flexible simulation framework for the multiphysic design and optimization of electrical machines and drives
* pymapdl_: Pythonic interface to Ansys MAPDL.
* pymeshfix_: Python/Cython wrapper of Marco Attene's wonderful, award-winning MeshFix_ software.
* pyprocar_: A Python library for electronic structure pre/post-processing.
* tetgen_: Python Interface to Hang Si's C++ TetGen Library

.. _MeshFix: https://github.com/MarcoAttene/MeshFix-V2.1
.. _PVGeo: https://github.com/OpenGeoVis/PVGeo
.. _creating compelling visualizations: https://discretize.simpeg.xyz/en/main/api/generated/discretize.mixins.vtk_mod.InterfaceVTK.html
.. _discretize: https://discretize.simpeg.xyz/en/main/
.. _gemgis: https://github.com/cgre-aachen/gemgis
.. _gimly: https://github.com/gimli-org/gimli
.. _itkwidgets: https://github.com/InsightSoftwareConsortium/itkwidgets
.. _mne-python: https://github.com/mne-tools/mne-python
.. _omfvista: https://github.com/OpenGeoVis/omfvista
.. _pyleecan: https://github.com/Eomys/pyleecan
.. _pymapdl: https://github.com/pyansys/pymapdl
.. _pymeshfix: https://github.com/pyvista/pymeshfix
.. _pyprocar: https://github.com/romerogroup/pyprocar
.. _tetgen: https://github.com/pyvista/tetgen


Citing PyVista
==============

There is a `paper about PyVista <https://doi.org/10.21105/joss.01450>`_!

If you are using PyVista in your scientific research, please help our scientific
visibility by citing our work! Head over to :ref:`citation_ref` to learn more
about citing PyVista.
