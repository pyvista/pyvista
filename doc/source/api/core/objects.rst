DataObject
==========

The :class:`pyvista.DataObject` class is a set of common methods and attributes
for all PyVista types. These objects have no spatial reference, but simply
hold data.

See :ref:`pyvista_data_model` for further details.

.. autosummary::
   :toctree: _autosummary

   pyvista.DataObject


Table
-----

The table class is a non-spatially referenced data object that can be used on
VTK pipelines and holds arrays of data.

.. autosummary::
   :toctree: _autosummary

   pyvista.Table


Textures
--------

The :class:`pyvista.Texture` class is used to load and represent images that
can be placed on the surface of :class:`pyvista.DataSet` that have texture
coordinates.

.. autosummary::
   :toctree: _autosummary

   pyvista.Texture

These functions create a :class:`pyvista.Texture` from image data or arrays.
To load a texture from an image file, see :func:`pyvista.read_texture`.

.. seealso::

   :ref:`texture_example`
      Apply textures to meshes.

   :ref:`pbr_example`
      Use a cubemap as an environment texture.

   :meth:`~pyvista.DataSetFilters.texture_map_to_plane`
      Generate texture coordinates for a dataset.

.. autosummary::
   :toctree: _autosummary

   pyvista.image_to_texture
   pyvista.numpy_to_texture
   pyvista.cubemap
   pyvista.cubemap_from_filenames
