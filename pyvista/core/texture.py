"""This module provides a wrapper vtkTexture."""
import warnings

import numpy as np

import pyvista
from pyvista import _vtk
from pyvista.utilities import PyVistaDeprecationWarning

from .dataset import DataObject


class Texture(_vtk.vtkTexture, DataObject):
    """Wrap vtk.Texture.

    Textures can be used to apply images to surfaces, as in the case of
    :ref:`ref_texture_example`.

    They can also be used for environment textures to affect the lighting of
    the scene, or even as a environment cubemap as in the case of
    :ref:`pbr_example` and :ref:`planets_example`.

    Parameters
    ----------
    uinput : str, vtkImageData, vtkTexture, numpy.ndarray, list, optional
        Filename, ``vtkImagedata``, ``vtkTexture``, :class:`numpy.ndarray` or a
        list of images to create a cubemap. If a list of images, must be of the
        same size and in the following order:

        * +X
        * -X
        * +Y
        * -Y
        * +Z
        * -Z

    **kwargs : dict, optional
        Optional arguments when reading from a file. Generally unused.

    Examples
    --------
    Load a texture from file. Files should be "image" or "image-like" files.

    >>> import os
    >>> import pyvista
    >>> from pyvista import examples
    >>> path = examples.download_masonry_texture(load=False)
    >>> os.path.basename(path)
    'masonry.bmp'
    >>> texture = pyvista.Texture(path)
    >>> texture  # doctest:+SKIP
    Texture (0x7f3131807fa0)
      Components:      3
      Cube Map:        False
      Dimensions:      256, 256

    Create a texture from an RGB array. Note how this is colored per "point"
    rather than per "pixel".

    >>> import numpy as np
    >>> arr = np.array(
    ...     [
    ...         [255, 255, 255],
    ...         [255, 0, 0],
    ...         [0, 255, 0],
    ...         [0, 0, 255],
    ...     ], dtype=np.uint8
    ... )
    >>> arr = arr.reshape((2, 2, 3))
    >>> texture = pyvista.Texture(arr)
    >>> texture.plot()

    Create a cubemap from 6 images.

    >>> px = examples.download_sky(direction='posx')  # doctest:+SKIP
    >>> nx = examples.download_sky(direction='negx')  # doctest:+SKIP
    >>> py = examples.download_sky(direction='posy')  # doctest:+SKIP
    >>> ny = examples.download_sky(direction='negy')  # doctest:+SKIP
    >>> pz = examples.download_sky(direction='posz')  # doctest:+SKIP
    >>> nz = examples.download_sky(direction='negz')  # doctest:+SKIP
    >>> texture = pyvista.Texture([px, nx, py, ny, pz, nz])  # doctest:+SKIP
    >>> texture.cube_map  # doctest:+SKIP
    True

    """

    def __init__(self, uinput=None, **kwargs):
        """Initialize the texture."""
        super().__init__(uinput, **kwargs)

        if isinstance(uinput, _vtk.vtkTexture):
            self._from_texture(uinput)
        elif isinstance(uinput, np.ndarray):
            self._from_array(uinput)
        elif isinstance(uinput, _vtk.vtkImageData):
            self._from_image_data(uinput)
        elif isinstance(uinput, str):
            self._from_file(filename=uinput, **kwargs)
        elif isinstance(uinput, list):
            # Create a cubemap

            if sum([isinstance(item, pyvista.UniformGrid) for item in uinput]) != 6:
                raise ValueError('`uinput` must consist of 6 pyvista.UniformGrid images.')

            self.SetMipmap(True)
            self.SetInterpolate(True)
            self.cube_map = True  # Must be set prior to setting images

            # add each image to the cubemap
            for i, image in enumerate(uinput):
                self.SetInputDataObject(i, image.flip_y())
        elif uinput is None:
            pass
        else:
            raise TypeError(f'Cannot create a pyvista.Texture from ({type(uinput)})')

    def __repr__(self):
        """Return the object representation."""
        return pyvista.DataSet.__repr__(self)

    def _get_attrs(self):
        """Return the representation methods (internal helper)."""
        attrs = []
        attrs.append(("Components", self.n_components, "{:d}"))
        attrs.append(("Cube Map", self.cube_map, "{:}"))
        attrs.append(("Dimensions", self.dimensions, "{:d}, {:d}"))
        return attrs

    @property
    def dimensions(self) -> tuple:
        """Dimensions of the texture.

        Examples
        --------
        >>> from pyvista import examples
        >>> texture = examples.download_masonry_texture()
        >>> texture.dimensions
        (256, 256)

        """
        input_data = self.GetInput()
        if input_data is None:
            return (0, 0)
        return input_data.GetDimensions()[:2]

    def _from_file(self, filename, **kwargs):
        try:
            image = pyvista.read(filename, **kwargs)
            if image.n_points < 2:  # pragma: no cover
                raise ValueError("Problem reading the image with VTK.")

            # to have textures match imageio's format, flip_x
            self._from_image_data(image.flip_y())
        except (KeyError, ValueError):
            from imageio import imread

            self._from_array(imread(filename))

    def _from_texture(self, texture):
        """Initialize or update from a pyvista.Texture."""
        self._from_image_data(texture.GetInput())

    def _from_image_data(self, image):
        """Initialize or update from a UniformGrid."""
        if not isinstance(image, pyvista.UniformGrid):
            image = pyvista.UniformGrid(image)
        self.SetInputDataObject(image)
        self.Update()

    def _from_array(self, arr):
        """Create a texture from a np.ndarray."""
        if not isinstance(arr, np.ndarray):
            raise TypeError(f"Invalid type {type(arr)}. Type must be numpy.ndarray")

        if arr.ndim not in [2, 3]:
            # we support 2 [single component image] or 3 [e.g. rgb or rgba] dims
            raise ValueError('Input image must be nn by nm by RGB[A]')

        if arr.ndim == 3:
            if arr.shape[2] not in [1, 3, 4]:
                raise ValueError(
                    'Third dimension of the array must be of size 1 (grayscale), 3 (RGB), or 4 (RGBA)'
                )
            n_components = arr.shape[2]
        elif arr.ndim == 2:
            n_components = 1

        grid = pyvista.UniformGrid(dims=(arr.shape[1], arr.shape[0], 1))
        grid.point_data['Image'] = arr.reshape(-1, n_components)
        self._from_image_data(grid)

    @property
    def repeat(self) -> bool:
        """Enable or disable repeating the texture."""
        return bool(self.GetRepeat())

    @repeat.setter
    def repeat(self, flag: bool):
        self.SetRepeat(flag)

    @property
    def n_components(self) -> int:
        """Components in the image.

        Single component textures are grayscale, while 3 or 4 component are
        used for representing RGB and RGBA images.

        Examples
        --------
        >>> from pyvista import examples
        >>> texture = examples.download_masonry_texture()
        >>> texture.n_components
        3

        """
        input_data = self.GetInput()
        if input_data is None:
            return 0
        return input_data.GetPointData().GetScalars().GetNumberOfComponents()

    def rotate_cw(self, inplace=False):
        """Rotate this texture 90 degrees clockwise.

        Parameters
        ----------
        inplace : bool, default: False
            Operate on this texture in-situ.

        Returns
        -------
        pyvista.Texture
            Rotated texture. This texture if ``inplace=True``.

        Examples
        --------
        >>> from pyvista import examples
        >>> texture = examples.download_puppy_texture()
        >>> rotated = texture.rotate_cw()
        >>> rotated.plot()

        """
        data = self.image_data[:, ::-1, :]
        if inplace:
            self._from_array(data)
            return self
        return Texture(data)

    def rotate_ccw(self, inplace=False):
        """Rotate this texture 90 degrees counter-clockwise.

        Parameters
        ----------
        inplace : bool, default: False
            Operate on this texture in-situ.

        Returns
        -------
        pyvista.Texture
            Rotated texture. This texture if ``inplace=True``.

        Examples
        --------
        >>> from pyvista import examples
        >>> texture = examples.download_puppy_texture()
        >>> rotated = texture.rotate_ccw()
        >>> rotated.plot()

        """
        data = self.image_data[::-1, :, :]
        if inplace:
            self._from_array(data)
            return self
        return Texture(data)

    def to_image(self) -> 'pyvista.UniformGrid':
        """Return the texture as an image (:class:`pyvista.UniformGrid`).

        Returns
        -------
        pyvista.UniformGrid or None
            Texture represented as a :class:`pyvista.UniformGrid` if set.

        Examples
        --------
        >>> from pyvista import examples
        >>> texture = examples.download_masonry_texture()
        >>> texture.to_image()  # doctest:+SKIP
        UniformGrid (0x7f313126afa0)
          N Cells:      65025
          N Points:     65536
          X Bounds:     0.000e+00, 2.550e+02
          Y Bounds:     0.000e+00, 2.550e+02
          Z Bounds:     0.000e+00, 0.000e+00
          Dimensions:   256, 256, 1
          Spacing:      1.000e+00, 1.000e+00, 1.000e+00
          N Arrays:     1

        """
        return self.GetInput()

    def to_array(self):
        """Return the texture as an array.

        .. deprecated:: 0.37.0
           ``to_array`` is deprecated. Use :attr:`Texture.image_data` instead.

        Returns
        -------
        pyvista.pyvista_ndarray
            Texture data as an array.

        """
        # Deprecated on v0.35.0, estimated removal on v0.37.0
        warnings.warn(  # pragma: no cover
            '`Texture.to_array` is deprecated. Use `image_data` instead', PyVistaDeprecationWarning
        )
        return self.image_data

    def plot(self, **kwargs):
        """Plot the texture as an image.

        If the texture is a cubemap, it will be displayed as a skybox.

        Parameters
        ----------
        **kwargs : dict, optional
            Optional keyworld arguments. See :func:`pyvista.plot`.

        Returns
        -------
        various or None
            See the returns section of :func:`pyvista.plot`.

        Examples
        --------
        >>> from pyvista import examples
        >>> texture = examples.download_masonry_texture()
        >>> texture.plot()

        """
        if self.cube_map:
            return self._plot_skybox(**kwargs)
        kwargs.setdefault('zoom', 'tight')
        kwargs.setdefault('lighting', False)
        kwargs.setdefault('rgba', self.n_components > 1)
        kwargs.setdefault('show_axes', False)
        kwargs.setdefault('show_scalar_bar', False)
        if self.n_components == 1:
            kwargs.setdefault('cmap', 'gray')
        return self.to_image().plot(**kwargs)

    def _plot_skybox(self, **kwargs):
        """Plot this texture as a skybox."""
        cpos = kwargs.pop('cpos', 'xy')
        zoom = kwargs.pop('zoom', 0.5)
        pl = pyvista.Plotter(**kwargs)
        pl.add_actor(self.to_skybox())
        pl.camera_position = cpos
        pl.camera.zoom(zoom)
        pl.show()

    @property
    def cube_map(self):
        """Return ``True`` if cube mapping is enabled and ``False`` otherwise."""
        return self.GetCubeMap()

    @cube_map.setter
    def cube_map(self, flag):
        self.SetCubeMap(flag)

    def copy(self, deep=True):
        """Make a copy of this texture.

        Parameters
        ----------
        deep : bool, optional
            Perform a deep copy when ``True``. Shallow copy when ``False``.

        Returns
        -------
        pyvista.Texture
            Copied texture.
        """
        return Texture(self.to_image().copy(deep=deep))

    def to_skybox(self):
        """Return the texture as a ``vtk.vtkSkybox`` if cube mapping is enabled.

        Returns
        -------
        vtk.vtkSkybox
            Skybox if cube mapping is enabled.  Otherwise, raises an exception.

        Examples
        --------
        Add a skybox to a plotter scene.

        Note how this texture is intentionally not mapped onto the sphere despite
        using physically based rendering. For this to be the case texture would
        have to also be added to the :class:`pyvista.Plotter` with
        :func`:pyvista.Plotter.set_environment_texture`.

        >>> import pyvista
        >>> from pyvista import examples
        >>> texture = examples.download_sky_cubemap()
        >>> skybox = texture.to_skybox()
        >>> pl = pv.Plotter()
        >>> pl.add_actor(skybox)
        >>> pl.add_mesh(pyvista.Sphere(), pbr=True, metallic=1.0)
        >>> pl.camera_position = 'xy'
        >>> pl.camera.zoom(0.5)
        >>> pl.show()

        """
        if not self.cube_map:
            raise ValueError('This texture is not a cube map.')
        skybox = _vtk.vtkSkybox()
        skybox.SetTexture(self)
        return skybox

    @property
    def image_data(self):
        """Return the underlying image data associated with this texture.

        Returns
        -------
        pyvista.pyvista_ndarray
            Array of the image data.

        Examples
        --------
        >>> from pyvista import examples
        >>> texture = examples.download_masonry_texture()
        >>> texture.image_data.shape
        (256, 256, 3)
        >>> texture.image_data.dtype
        dtype('uint8')

        """
        return self.to_image().active_scalars.squeeze()

    @image_data.setter
    def image_data(self, image_data):
        """Set the image data."""
        self._from_array(image_data)

    def to_grayscale(self):
        """Convert this texture as a single component (grayscale) texture.

        Returns
        -------
        pyvista.Texture
            Texture converted to grayscale. If already grayscale, the original
            texture itself is returned.

        Notes
        -----
        The transparency channel (if available) will be dropped.

        Follows the `CCIR 601 <https://en.wikipedia.org/wiki/Rec._601>`_ luma
        calculation equation of ``Y = 0.299*R + 0.587*G + 0.114*B``.

        Examples
        --------
        >>> from pyvista import examples
        >>> texture = examples.download_masonry_texture()
        >>> bw_texture = texture.to_grayscale()
        >>> bw_texture.plot()

        """
        if self.n_components == 1:
            return self

        data = np.asarray(self.image_data)  # use asarray decouple VTK data
        r, g, b = data[..., 0], data[..., 1], data[..., 2]
        grayscale = (0.299 * r + 0.587 * g + 0.114 * b).round().astype(np.uint8)
        return Texture(grayscale.T)

    def flip_x(self, inplace=False):
        """Flip the texture in the x direction.

        Parameters
        ----------
        inplace : bool, default: False
            Operate on this texture in-situ.

        Returns
        -------
        pyvista.Texture
            Flipped texture. This texture if ``inplace=True``.

        Examples
        --------
        >>> from pyvista import examples
        >>> texture = examples.download_puppy_texture()
        >>> flipped = texture.flip_x()
        >>> flipped.plot()

        """
        if inplace:
            self._from_image_data(self.to_image().flip_x(inplace))
            return self
        return Texture(self.to_image().flip_x(inplace))

    def flip_y(self, inplace=False):
        """Flip the texture in the y direction.

        Parameters
        ----------
        inplace : bool, default: False
            Operate on this texture in-situ.

        Returns
        -------
        pyvista.Texture
            Flipped texture. This texture if ``inplace=True``.

        Examples
        --------
        >>> from pyvista import examples
        >>> texture = examples.download_puppy_texture()
        >>> flipped = texture.flip_y()
        >>> flipped.plot()

        """
        if inplace:
            self._from_image_data(self.to_image().flip_y(inplace))
            return self
        return Texture(self.to_image().flip_y(inplace))
