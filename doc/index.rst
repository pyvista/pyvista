.. title:: PyVista

.. raw:: html

    <div class="banner">
        <a href="./examples/index.html"><center><img src="_static/pyvista_logo.png" alt="pyvista" width="75%"/></a>
        <h3>3D plotting and mesh analysis through a streamlined interface for the Visualization Toolkit (VTK)</h2>
        <a href="./examples/index.html"><img src="_static/pyvista_banner_small.png" alt="pyvista" width="100%"/></a>
    </div>

|


Overview
********
PyVista is...

* *Pythonic VTK*: a high-level API to the `Visualization Toolkit`_ (VTK)
* mesh data structures and filtering methods for spatial datasets
* 3D plotting made simple and built for large/complex data geometries

.. _Visualization Toolkit: https://vtk.org

PyVista is a helper library for the Visualization Toolkit (VTK) that
takes a different approach on interfacing with VTK through NumPy and
direct array access.  This package provides a Pythonic,
well-documented interface exposing VTK's powerful visualization
backend to facilitate rapid prototyping, analysis, and visual
integration of spatially referenced datasets.

This module can be used for scientific plotting for presentations and
research papers as well as a supporting module for other mesh
dependent Python modules.

.. |tweet| image:: https://img.shields.io/twitter/url.svg?style=social&url=http%3A%2F%2Fshields.io
   :target: https://twitter.com/intent/tweet?text=Check%20out%20this%20project%20for%203D%20visualization%20in%20Python&url=https://github.com/pyvista/pyvista&hashtags=3D,visualization,Python,vtk,mesh,plotting,PyVista

Share this project on Twitter: |tweet|


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
   examples/index
   api/index
   extras/index


Brief Examples
**************
Here are some brief interactive examples that demonstrate how you
might want to use PyVista:


.. jupyter-execute::
   :hide-code:

   # Configure for panel
   import pyvista
   pyvista.set_jupyter_backend('panel')
   pyvista.global_theme.background = 'white'
   pyvista.global_theme.window_size = [600, 400]
   pyvista.global_theme.axes.show = False
   pyvista.global_theme.smooth_shading = True
   pyvista.global_theme.anti_aliasing = 'fxaa'


Maps and Geoscience
~~~~~~~~~~~~~~~~~~~
Download the surface elevation map of Mount St. Helens and plot it.

.. jupyter-execute::

    from pyvista import examples
    mesh = examples.download_st_helens()
    warped = mesh.warp_by_scalar('Elevation')
    surf = warped.extract_surface().triangulate()
    surf = surf.decimate_pro(0.75)  # reduce the density of the mesh by 75%
    surf.plot(cmap='gist_earth')


Finite Element Analysis
~~~~~~~~~~~~~~~~~~~~~~~
Plot the 'X' component of elastic stress of a 3D notch specimen.

.. jupyter-execute::

   from pyvista import examples
   mesh = examples.download_notch_stress()
   mesh.plot(scalars='Nodal Stress', component=0, cmap='turbo', cpos='xy')


Simple Point Cloud with Numpy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Easily integrate with NumPy and create a variety of geometries and plot
them.  You could use any geometry to create your glyphs, or even plot
the points directly.

.. jupyter-execute::

    import numpy as np
    import pyvista

    point_cloud = np.random.random((100, 3))
    pdata = pyvista.PolyData(point_cloud)
    pdata['orig_sphere'] = np.arange(100)

    # create many spheres from the point cloud
    sphere = pyvista.Sphere(radius=0.02, phi_resolution=10, theta_resolution=10)
    pc = pdata.glyph(scale=False, geom=sphere, orient=False)
    pc.plot(cmap='Reds')


Plot a Spline
~~~~~~~~~~~~~
Generate a spline from an array of NumPy points.

.. jupyter-execute::

    import numpy as np
    import pyvista

    # Make the xyz points
    theta = np.linspace(-10 * np.pi, 10 * np.pi, 100)
    z = np.linspace(-2, 2, 100)
    r = z**2 + 1
    x = r * np.sin(theta)
    y = r * np.cos(theta)
    points = np.column_stack((x, y, z))

    spline = pyvista.Spline(points, 500).tube(radius=0.1)
    spline.plot(scalars='arc_length', show_scalar_bar=False)


Boolean Operations on Meshes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Subtract a sphere from a cube mesh.

.. jupyter-execute::

    import pyvista
    import numpy as np

    def make_cube():
        x = np.linspace(-0.5, 0.5, 25)
        grid = pyvista.StructuredGrid(*np.meshgrid(x, x, x))
        surf = grid.extract_surface().triangulate()
        surf.flip_normals()
        return surf

    # Create example PolyData meshes for boolean operations
    sphere = pyvista.Sphere(radius=0.65, center=(0, 0, 0))
    cube = make_cube()

    # Perform a boolean difference
    boolean = cube.boolean_difference(sphere)
    boolean.plot(color='darkgrey', smooth_shading=True, split_sharp_edges=True)


Plot Volumetric Data
~~~~~~~~~~~~~~~~~~~~
Plot the :math:`3d_{xy}` orbital of a hydrogen atom.

.. note::
   This example requires `sympy <https://www.sympy.org/>`_.

.. jupyter-execute::

    from pyvista import examples
    grid = examples.load_hydrogen_orbital(3, 2, -2)
    grid.plot(volume=True, opacity=[1, 0, 1], cmap='magma')


Translating
***********
The recommended way for new contributors to translate PyVista's
documentation is to join the translation team on Transifex.

There is a `pyvista translation page`_ for pyvista (main) documentation.

1. Login to transifex_ service.
2. Go to `pyvista translation page`_.
3. Click ``Request language`` and fill form.
4. Wait acceptance by transifex pyvista translation maintainers.
5. (After acceptance) Translate on transifex.
6. We can host the translated document in `GitHub Pages`_ by creating `GitHub repository`_.
7. Translation is backed up in `pyvista-doc-translations`_.

Details can be found here: https://docs.transifex.com/getting-started-1/translators

.. _`pyvista translation page`: https://www.transifex.com/tkoyama010/pyvista-doc/
.. _Transifex: https://www.transifex.com/
.. _`GitHub Pages`: https://pyvista.github.io/pyvista-docs-dev-ja/index.html
.. _`GitHub repository`: https://github.com/pyvista/pyvista-docs-dev-ja
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
   :target: https://codecov.io/gh/pyvista/pyvista

.. |codacy| image:: https://app.codacy.com/project/badge/Grade/779ac6aed37548839384acfc0c1aab44
   :target: https://www.codacy.com/gh/pyvista/pyvista/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=pyvista/pyvista&amp;utm_campaign=Badge_Grade

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

.. |PyPIact| image:: https://img.shields.io/pypi/dm/pyvista.svg?label=PyPI%20downloads
   :target: https://pypi.org/project/pyvista/

.. |condaact| image:: https://img.shields.io/conda/dn/conda-forge/pyvista.svg?label=Conda%20downloads
   :target: https://anaconda.org/conda-forge/pyvista

.. |discuss| image:: https://img.shields.io/badge/GitHub-Discussions-green?logo=github
   :target: https://github.com/pyvista/pyvista/discussions

.. |python| image:: https://img.shields.io/badge/python-3.7+-blue.svg
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


Project Index
*************

* :ref:`genindex`
