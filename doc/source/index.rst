.. title:: PyVista

.. raw:: html

    <div class="banner">
        <a href="./examples/index.html"><center><img src="_static/pyvista_logo.svg" alt="pyvista" width="75%"/></a>
        <h3>3D visualization and mesh analysis for science and engineering</h3>
        <a href="./examples/index.html"><img src="_static/pyvista_banner_small.png" alt="pyvista" width="100%"/></a>
    </div>

|


Overview
********

PyVista is the foundational Python library for 3D visualization and mesh
analysis in scientific computing and engineering. PyVista plays the same role
for 3D data that pandas plays for tabular data and xarray plays for labeled
n-dimensional arrays.

* a NumPy-native API for 3D visualization and mesh analysis
* dataset structures and filters for points, surfaces, and volumes
* one plotting framework for notebooks, scripts, CI, and apps
* streamlined 3D for newcomers and graphics experts alike

Use PyVista for figures in papers and presentations, interactive analysis in
notebooks, and as the visualization layer of larger Python tools.


Built for production
====================

3D code has to keep working when the underlying graphics stack changes.
PyVista is image-regression tested on every commit across the supported
matrix of Python and `VTK`_ releases. Public APIs follow a deliberate
deprecation lifecycle (warn, then error, then remove) across at least three
minor versions, and every release ships wheels for every supported Python
version.

The C++ toolkit underneath does not provide the same guarantees. Its Python
bindings shift between releases, large parts of the surface ship without test
coverage, and rendering changes are not gated on visual diffs. PyVista is
what science and engineering teams reach for when code written today has to
still produce the same picture two years from now.


Built to extend
===============

Downstream libraries build on PyVista through a small, lazily evaluated
extension API. Third-party packages attach domain-specific filters and
plotter components via registered accessors, with no subclassing, no
monkey-patching, and no vendoring of upstream algorithms. This is the same
contract pandas exposes for tabular accessors and xarray exposes for labeled
arrays. See :ref:`extending-pyvista` for the full contract.

For a curated, continuously updated list of domain-specific tooling that
interoperates with or is built on PyVista, see
`awesome-pyvista <https://github.com/pyvista/awesome-pyvista>`_.

Reach for the underlying `VTK`_ toolkit directly only when there is genuinely
no PyVista equivalent. When that happens, please file an issue. There should
be one.

.. _VTK: https://vtk.org

.. |binder| image:: https://static.mybinder.org/badge_logo.svg
   :target: https://mybinder.org/v2/gh/pyvista/pyvista-examples/master
   :alt: Launch on Binder

Want to test-drive PyVista? Check out our live examples on MyBinder: |binder|

.. grid::

   .. grid-item-card:: PyVista is a NumFOCUS affiliated project
      :link: https://numfocus.org/sponsored-projects/affiliated-projects
      :class-title: pyvista-card-title

      .. image:: https://raw.githubusercontent.com/numfocus/templates/master/images/numfocus-logo.png
         :target: https://numfocus.org/sponsored-projects/affiliated-projects
         :alt: NumFOCUS affiliated projects
         :height: 60px


.. toctree::
   :hidden:

   getting-started/index
   user-guide/index
   tags/tagsindex
   examples/index
   api/index
   extras/index


Brief Examples
**************
Here are some brief interactive examples that demonstrate how you
might want to use PyVista:


.. pyvista-plot::
   :context:
   :include-source: false
   :force_static:

   import pyvista as pv
   pv.set_jupyter_backend('static')
   pv.global_theme.background = 'white'
   pv.global_theme.window_size = [600, 400]
   pv.global_theme.axes.show = False
   pv.global_theme.smooth_shading = True
   pv.global_theme.anti_aliasing = 'fxaa'


Maps and Geoscience
~~~~~~~~~~~~~~~~~~~
Download the surface elevation map of Mount St. Helens and plot it.

.. pyvista-plot::
    :context:

    from pyvista import examples
    mesh = examples.download_st_helens()
    warped = mesh.warp_by_scalar('Elevation')
    surf = warped.extract_surface().triangulate()
    surf = surf.decimate_pro(0.75)  # reduce the density of the mesh by 75%
    surf.plot(cmap='gist_earth')


Finite Element Analysis
~~~~~~~~~~~~~~~~~~~~~~~
Plot the 'X' component of elastic stress of a 3D notch specimen.


.. pyvista-plot::
   :context:

   from pyvista import examples
   mesh = examples.download_notch_stress()
   mesh.plot(scalars='Nodal Stress', component=0, cmap='turbo', cpos='xy')


Simple Point Cloud with NumPy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Easily integrate with NumPy and create a variety of geometries and plot
them. You could use any geometry to create your glyphs, or even plot
the points directly.


.. pyvista-plot::
    :context:

    import numpy as np
    import pyvista as pv

    rng = np.random.default_rng(seed=0)
    point_cloud = rng.random((100, 3))
    pdata = pv.PolyData(point_cloud)
    pdata['orig_sphere'] = np.arange(100)

    # create many spheres from the point cloud
    sphere = pv.Sphere(radius=0.02, phi_resolution=10, theta_resolution=10)
    pc = pdata.glyph(scale=False, geom=sphere, orient=False)
    pc.plot(cmap='Reds')


Plot a Spline
~~~~~~~~~~~~~
Generate a spline from an array of NumPy points.

.. pyvista-plot::
    :context:

    import numpy as np
    import pyvista as pv

    # Make the xyz points
    theta = np.linspace(-10 * np.pi, 10 * np.pi, 100)
    z = np.linspace(-2, 2, 100)
    r = z**2 + 1
    x = r * np.sin(theta)
    y = r * np.cos(theta)
    points = np.column_stack((x, y, z))

    spline = pv.Spline(points, 500).tube(radius=0.1)
    spline.plot(scalars='arc_length', show_scalar_bar=False)


Boolean Operations on Meshes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Subtract a sphere from a cube mesh.

.. pyvista-plot::
    :context:

    import pyvista as pv
    import numpy as np

    def make_cube():
        x = np.linspace(-0.5, 0.5, 25)
        grid = pv.StructuredGrid(*np.meshgrid(x, x, x))
        surf = grid.extract_surface().triangulate().flip_faces()
        return surf

    # Create example PolyData meshes for boolean operations
    sphere = pv.Sphere(radius=0.65, center=(0, 0, 0))
    cube = make_cube()

    # Perform a boolean difference
    boolean = cube.boolean_difference(sphere)
    boolean.plot(color='darkgrey', smooth_shading=True, split_sharp_edges=True)


Plot Volumetric Data
~~~~~~~~~~~~~~~~~~~~
Plot the :math:`3d_{xy}` orbital of a hydrogen atom.

.. note::
   This example requires `sympy <https://www.sympy.org/>`_.

.. pyvista-plot::
    :context:
    :force_static:

    from pyvista import examples
    grid = examples.load_hydrogen_orbital(3, 2, -2)
    grid.plot(volume=True, opacity=[1, 0, 1], cmap='magma')


Translating
***********
The recommended way for new contributors to translate PyVista's
documentation is to join the translation team on Transifex.

There is a `pyvista translation page`_ for pyvista (main) documentation.

#. Login to transifex_ service.
#. Go to `pyvista translation page`_.
#. Click ``Request language`` and fill form.
#. Wait acceptance by transifex pyvista translation maintainers.
#. (After acceptance) Translate on transifex.
#. We can host the translated document using `atsphinx-mini18n`_.
#. Translation is backed up in `pyvista-doc-translations`_.

Details can be found here: https://help.transifex.com/en/

.. _`pyvista translation page`: https://app.transifex.com/signin/?next=/tkoyama010/pyvista-doc/
.. _Transifex: https://app.transifex.com/signin/?next=/home/
.. _atsphinx-mini18n: https://atsphinx.github.io/mini18n/en/
.. _`pyvista-doc-translations`: https://github.com/pyvista/pyvista-doc-translations


Status
******

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

.. |contributors| image:: https://img.shields.io/github/contributors/pyvista/pyvista.svg?logo=github&logoColor=white
   :target: https://github.com/pyvista/pyvista/graphs/contributors/

.. |stars| image:: https://img.shields.io/github/stars/pyvista/pyvista.svg?style=social&label=Stars
   :target: https://github.com/pyvista/pyvista
   :alt: GitHub

.. |zenodo| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.8415866.svg
   :target: https://zenodo.org/records/8415866

.. |joss| image:: https://joss.theoj.org/papers/78f2901bbdfbd2a6070ec41e8282d978/status.svg
   :target: https://joss.theoj.org/papers/10.21105/joss.01450

.. |MIT| image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :target: https://opensource.org/license/mit/

.. |slack| image:: https://img.shields.io/badge/Slack-PyVista-green.svg?logo=slack
   :target: https://communityinviter.com/apps/pyvista/pyvista

.. |PyPIact| image:: https://img.shields.io/pypi/dm/pyvista.svg?label=PyPI%20downloads
   :target: https://pypi.org/project/pyvista/

.. |condaact| image:: https://img.shields.io/conda/dn/conda-forge/pyvista.svg?label=Conda%20downloads
   :target: https://anaconda.org/conda-forge/pyvista

.. |discuss| image:: https://img.shields.io/badge/GitHub-Discussions-green?logo=github
   :target: https://github.com/pyvista/pyvista/discussions

.. |python| image:: https://img.shields.io/badge/python-3.9+-blue.svg
   :target: https://www.python.org/downloads/

+----------------------+----------------+-------------+
| Deployment           | |pypi|         | |conda|     |
+----------------------+----------------+-------------+
| Build Status         | |GH-CI|        |  |python|   |
+----------------------+----------------+-------------+
| Metrics              | |codacy|       | |codecov|   |
+----------------------+----------------+-------------+
| Activity             | |PyPIact|      | |condaact|  |
+----------------------+----------------+-------------+
| GitHub               | |contributors| | |stars|     |
+----------------------+----------------+-------------+
| Citation             | |joss|         | |zenodo|    |
+----------------------+----------------+-------------+
| License              | |MIT|          |             |
+----------------------+----------------+-------------+
| Community            | |slack|        | |discuss|   |
+----------------------+----------------+-------------+


Professional Support
********************

PyVista is a community-driven Open Source project, but many users and organizations rely on it in production workflows, research pipelines, and custom visualization systems. If you need expert guidance, development help, or guaranteed support, there are several ways to engage with the people who build and maintain PyVista.

For general inquiries, reach out to info@pyvista.org and we can help connect you with the right community experts for your 3D visualization or analysis needs.

If you are looking for professional services (consulting, custom development, feature design, integration support, or training), consider sponsoring PyVista's core developers through the “Sponsor this project” section on GitHub. Sponsorship not only provides direct access to experts but also helps sustain critical maintenance and ongoing feature work that keeps PyVista reliable and modern.

More details can be found in the discussion post: https://github.com/pyvista/pyvista/discussions/4033

Sponsoring a developer supports both your project and the health of the PyVista ecosystem, ensuring continued improvements, long-term stability, and expert help when you need it.


Project Index
*************

* :ref:`genindex`
