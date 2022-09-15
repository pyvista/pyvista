"""An internal module for wrapping the use of mappers."""
import logging
import sys
from typing import Optional

import numpy as np

import pyvista as pv
from pyvista import _vtk
from pyvista.utilities import (
    abstract_class,
    convert_array,
    convert_string_array,
    raise_not_matching,
)
from pyvista.utilities.misc import has_module

from .colors import Color, get_cmap_safe
from .tools import normalize


@abstract_class
class _BaseMapper(_vtk.vtkAbstractMapper):
    """Base Mapper with methods common to other mappers."""

    def __init__(self, theme, **kwargs):
        self._theme = theme

    def copy(self, deep=True):
        """Create a copy of this mapper."""
        new_mapper = type(self)()
        if deep:
            new_mapper.dataset = self.dataset
            new_mapper.scalar_map_mode = self.scalar_map_mode
            new_mapper.scalar_range = self.scalar_range
            new_mapper.scalar_visibility = self.scalar_visibility
        else:
            new_mapper.ShallowCopy(self)
        return new_mapper

    @property
    def scalar_range(self) -> tuple:
        """Return or set the scalar range.

        Examples
        --------
        Return the scalar range of a mapper.

        >>> import pyvista as pv
        >>> mesh = pv.Sphere()
        >>> pl = pv.Plotter()
        >>> actor = pl.add_mesh(mesh, scalars=mesh.points[:, 2])
        >>> actor.mapper.scalar_range
        (-0.5, 0.5)
        >>> pl.close()

        Return the scalar range of a composite dataset. In this example it's
        set to its default value of ``(0.0, 1.0)``.

        >>> import pyvista as pv
        >>> dataset = pv.MultiBlock([pv.Cube(), pv.Sphere(center=(0, 0, 1))])
        >>> pl = pv.Plotter()
        >>> actor, mapper = pl.add_composite(dataset)
        >>> mapper.scalar_range
        (0.0, 1.0)
        >>> pl.close()

        """
        return self.GetScalarRange()

    @scalar_range.setter
    def scalar_range(self, clim):
        self.SetScalarRange(*clim)
        if self.lookup_table is not None:
            self.lookup_table.SetRange(*clim)

    @property
    def lookup_table(self) -> pv._vtk.vtkLookupTable:
        """Return or set the lookup table.

        Examples
        --------
        Return the lookup table of a composite dataset mapper.

        >>> import pyvista as pv
        >>> dataset = pv.MultiBlock([pv.Cube(), pv.Sphere(center=(0, 0, 1))])
        >>> pl = pv.Plotter()
        >>> actor, mapper = pl.add_composite(dataset)
        >>> mapper.lookup_table  # doctest:+SKIP
        <vtkmodules.vtkCommonCore.vtkLookupTable(0x2d4c6e0) at 0x7fce74a89fa0>

        """
        return self.GetLookupTable()

    @lookup_table.setter
    def lookup_table(self, table):
        return self.SetLookupTable(table)

    @property
    def color_mode(self) -> str:
        """Return or set the color mode.

        Either ``'direct'``, or ``'map'``.

        * ``'direct'`` - All integer types are treated as colors with values in
          the range 0-255 and floating types are treated as colors with values
          in the range 0.0-1.0
        * ``'map'`` - All scalar data will be mapped through the lookup table.

        """
        mode = self.GetColorModeAsString().lower()
        if mode == 'mapscalars':
            return 'map'
        return 'direct'

    @color_mode.setter
    def color_mode(self, value: str):
        if value == 'direct':
            self.SetColorModeToDirectScalars()
        elif value == 'map':
            self.SetColorModeToMapScalars()
        else:
            raise ValueError('color mode must be either "default", "direct" or "map"')

    @property
    def interpolate_before_map(self) -> bool:
        """Return or set the interpolation of scalars before mapping.

        Enabling makes for a smoother scalars display.  When ``False``,
        OpenGL will interpolate the mapped colors which can result in
        showing colors that are not present in the color map.

        Examples
        --------
        Disable interpolation before mapping.

        >>> import pyvista as pv
        >>> dataset = pv.MultiBlock([pv.Cube(), pv.Sphere(center=(0, 0, 1))])
        >>> dataset[0].point_data['data'] = dataset[0].points[:, 2]
        >>> dataset[1].point_data['data'] = dataset[1].points[:, 2]
        >>> pl = pv.Plotter()
        >>> actor, mapper = pl.add_composite(
        ...     dataset, show_scalar_bar=False, n_colors=3, cmap='bwr',
        ... )
        >>> mapper.interpolate_before_map = False
        >>> pl.show()

        Enable interpolation before mapping.

        >>> pl = pv.Plotter()
        >>> actor, mapper = pl.add_composite(
        ...     dataset, show_scalar_bar=False, n_colors=3, cmap='bwr',
        ... )
        >>> mapper.interpolate_before_map = True
        >>> pl.show()

        See :ref:`interpolate_before_mapping_example` for additional
        explanation regarding this attribute.

        """
        return bool(self.GetInterpolateScalarsBeforeMapping())

    @interpolate_before_map.setter
    def interpolate_before_map(self, value: bool):
        self.SetInterpolateScalarsBeforeMapping(value)

    @property
    def array_name(self) -> str:
        """Return or set the array name or number and component to color by.

        Examples
        --------
        Show the name of the active scalars in the mapper.

        >>> import pyvista as pv
        >>> mesh = pv.Sphere()
        >>> mesh['my_scalars'] = mesh.points[:, 2]
        >>> pl = pv.Plotter()
        >>> actor = pl.add_mesh(mesh, scalars='my_scalars')
        >>> actor.mapper.array_name
        'my_scalars'
        >>> pl.close()

        """
        return self.GetArrayName()

    @array_name.setter
    def array_name(self, name: str):
        """Return or set the array name or number and component to color by."""
        self.SetArrayName(name)

    @property
    def scalar_map_mode(self) -> str:
        """Return or set the scalar map mode.

        Examples
        --------
        Show that the scalar map mode is set to ``'point'`` when setting the
        active scalars to point data.

        >>> import pyvista as pv
        >>> dataset = pv.MultiBlock([pv.Cube(), pv.Sphere(center=(0, 0, 1))])
        >>> dataset[0].point_data['data'] = dataset[0].points[:, 2]
        >>> dataset[1].point_data['data'] = dataset[1].points[:, 2]
        >>> pl = pv.Plotter()
        >>> actor, mapper = pl.add_composite(dataset, scalars='data', show_scalar_bar=False)
        >>> mapper.scalar_map_mode
        'point'
        >>> pl.close()

        """
        # map vtk strings to more sensible strings
        vtk_to_pv = {
            'Default': 'default',
            'UsePointData': 'point',
            'UseCellData': 'cell',
            'UsePointFieldData': 'point_field',
            'UseCellFieldData': 'cell_field',
            'UseFieldData': 'field',
        }
        return vtk_to_pv[self.GetScalarModeAsString()]

    @scalar_map_mode.setter
    def scalar_map_mode(self, scalar_mode: str):
        if scalar_mode == 'default':
            self.SetScalarModeToDefault()
        elif scalar_mode == 'point':
            self.SetScalarModeToUsePointData()
        elif scalar_mode == 'cell':
            self.SetScalarModeToUseCellData()
        elif scalar_mode == 'point_field':
            self.SetScalarModeToUsePointFieldData()
        elif scalar_mode == 'cell_field':
            self.SetScalarModeToUseCellFieldData()
        elif scalar_mode == 'field':
            self.SetScalarModeToUseFieldData()
        else:
            raise ValueError(
                f'Invalid `scalar_map_mode` "{scalar_mode}". Should be either '
                '"default", "point", "cell", "point_field", "cell_field" or "field".'
            )

    @property
    def scalar_visibility(self) -> bool:
        """Return or set the scalar visibility.

        Examples
        --------
        Show that scalar visibility is ``False``.

        >>> import pyvista as pv
        >>> mesh = pv.Sphere()
        >>> pl = pv.Plotter()
        >>> actor = pl.add_mesh(mesh)
        >>> actor.mapper.scalar_visibility
        False
        >>> pl.close()

        Show that scalar visibility is ``True``.

        >>> import pyvista as pv
        >>> dataset = pv.MultiBlock([pv.Cube(), pv.Sphere(center=(0, 0, 1))])
        >>> dataset[0].point_data['data'] = dataset[0].points[:, 2]
        >>> dataset[1].point_data['data'] = dataset[1].points[:, 2]
        >>> pl = pv.Plotter()
        >>> actor, mapper = pl.add_composite(dataset, scalars='data')
        >>> mapper.scalar_visibility
        True
        >>> pl.close()

        """
        return bool(self.GetScalarVisibility())

    @scalar_visibility.setter
    def scalar_visibility(self, value: bool):
        return self.SetScalarVisibility(value)

    def update(self):
        """Update this mapper."""
        self.Update()


class DataSetMapper(_vtk.vtkDataSetMapper, _BaseMapper):
    """Wrap _vtk.vtkDataSetMapper.

    Parameters
    ----------
    dataset : pyvista.DataSet, optional
        Dataset to assign to this mapper.

    theme : pyvista.themes.DefaultTheme, optional
        Plot-specific theme.

    Examples
    --------
    Create a mapper outside :class:`pyvista.Plotter` and assign it to an
    actor.

    >>> import pyvista as pv
    >>> mesh = pv.Cube()
    >>> mapper = pv.DataSetMapper(dataset=mesh)
    >>> actor = pv.Actor(mapper=mapper)
    >>> actor.plot()

    """

    def __init__(
        self,
        dataset: Optional['pv.DataSet'] = None,
        theme: Optional['pv.themes.DefaultTheme'] = None,
    ):
        """Initialize this class."""
        super().__init__(theme)
        if dataset is not None:
            self.dataset = dataset

    @property
    def dataset(self) -> Optional['pv.DataSet']:
        """Return or set the dataset assigned to this mapper."""
        return self.GetInputAsDataSet()

    @dataset.setter
    def dataset(self, obj: 'pv.core.dataset.DataSet'):
        return self.SetInputData(obj)

    def _configure_scalars_mode(
        self,
        scalars,
        scalars_name,
        n_colors,
        preference,
        direct_scalars_color_mode,
    ):
        """Configure scalar mode.

        Parameters
        ----------
        scalars : numpy.ndarray
            Array of scalars to assign to the mapper.

        scalars_name : str
            If the name of this array exists, scalars is ignored. Otherwise,
            the scalars will be added to the existing dataset and this
            parameter is the name to assign the scalars.

        n_colors : int
            Number of colors.

        preference : str
            Either ``'point'`` or ``'cell'``.

        direct_scalars_color_mode : bool
            When ``True``, scalars are treated as RGB colors. When
            ``False``, scalars are mapped to the color table.

        """
        if scalars.shape[0] == self.dataset.n_points and scalars.shape[0] == self.dataset.n_cells:
            use_points = preference == 'point'
            use_cells = not use_points
        else:
            use_points = scalars.shape[0] == self.dataset.n_points
            use_cells = scalars.shape[0] == self.dataset.n_cells

        # Scalars interpolation approach
        if use_points:
            if scalars_name not in self.dataset.point_data:
                self.dataset.point_data.set_array(scalars, scalars_name, False)
            self.dataset.active_scalars_name = scalars_name
            self.scalar_map_mode = 'point'
        elif use_cells:
            if scalars_name not in self.dataset.cell_data:
                self.dataset.cell_data.set_array(scalars, scalars_name, False)
            self.dataset.active_scalars_name = scalars_name
            self.scalar_map_mode = 'cell'
        else:
            raise_not_matching(scalars, self.dataset)

        self.lookup_table.SetNumberOfTableValues(n_colors)
        if direct_scalars_color_mode:
            self.color_mode = 'direct'
        else:
            self.color_mode = 'map'

    def set_scalars(
        self,
        scalars,
        scalars_name,
        n_colors=256,
        scalar_bar_args=None,
        rgb=None,
        component=None,
        preference='point',
        custom_opac=False,
        annotations=None,
        log_scale=False,
        nan_color=None,
        above_color=None,
        below_color=None,
        cmap=None,
        flip_scalars=False,
        opacity=None,
        categories=False,
        clim=None,
    ):
        """Set the scalars on this mapper.

        Parameters
        ----------
        scalars : numpy.ndarray
            Array of scalars to assign to the mapper.

        scalars_name : str
            If the name of this array exists, scalars is ignored. Otherwise,
            the scalars will be added to the existing dataset and this
            parameter is the name to assign the scalars.

        n_colors : int, default: 256
            Number of colors to use when displaying scalars.

        scalar_bar_args : dict, optional
            Dictionary of keyword arguments to pass when adding the
            scalar bar to the scene. For options, see
            :func:`pyvista.BasePlotter.add_scalar_bar`.

        rgb : bool, default: False
            If an 2 dimensional array is passed as the scalars, plot
            those values as RGB(A) colors. ``rgba`` is also an
            accepted alias for this.  Opacity (the A) is optional.  If
            a scalars array ending with ``"_rgba"`` is passed, the default
            becomes ``True``.  This can be overridden by setting this
            parameter to ``False``.

        component : int, optional
            Set component of vector valued scalars to plot.  Must be
            nonnegative, if supplied. If ``None``, the magnitude of
            the vector is plotted.

        preference : str, default: 'Point'
            When ``dataset.n_points == dataset.n_cells`` and setting scalars,
            this parameter sets how the scalars will be mapped to the mesh.
            Can be either ``'point'`` or ``'cell'``.

        custom_opac : bool, default: False
            Use custom opacity.

        annotations : dict, optional
            Pass a dictionary of annotations. Keys are the float
            values in the scalars range to annotate on the scalar bar
            and the values are the the string annotations.

        log_scale : bool, default: False
            Use log scale when mapping data to colors. Scalars less
            than zero are mapped to the smallest representable
            positive float.

        nan_color : color_like
            The color to use for all ``NaN`` values in the plotted
            scalar array.

        above_color : color_like, optional
            Solid color for values below the scalars range
            (``clim``). This will automatically set the scalar bar
            ``above_label`` to ``'Above'``.

        below_color : color_like, optional
            Solid color for values below the scalars range
            (``clim``). This will automatically set the scalar bar
            ``below_label`` to ``'Below'``.

        cmap : str, list, optional
            Name of the Matplotlib colormap to use when mapping the
            ``scalars``.  See available Matplotlib colormaps.  Only
            applicable for when displaying ``scalars``. Requires
            Matplotlib to be installed.  ``colormap`` is also an
            accepted alias for this. If ``colorcet`` or ``cmocean``
            are installed, their colormaps can be specified by name.

            You can also specify a list of colors to override an
            existing colormap with a custom one.  For example, to
            create a three color colormap you might specify
            ``['green', 'red', 'blue']``.

        flip_scalars : bool, default: False
            Flip direction of cmap. Most colormaps allow ``*_r`` suffix to do
            this as well.

        opacity : str or numpy.ndarray, optional
            Opacity mapping for the scalars array.
            A string can also be specified to map the scalars range to a
            predefined opacity transfer function (options include: 'linear',
            'linear_r', 'geom', 'geom_r'). Or you can pass a custom made
            transfer function that is an array either ``n_colors`` in length or
            shorter.

        categories : bool, default: False
            If set to ``True``, then the number of unique values in the scalar
            array will be used as the ``n_colors`` argument.

        clim : 2 item list, optional
            Color bar range for scalars.  Defaults to minimum and
            maximum of scalars array.  Example: ``(-1, 2)``.

        """
        if scalar_bar_args is None:
            scalar_bar_args = {'n_colors': n_colors}

        if cmap is None:  # Set default map if matplotlib is available
            if has_module('matplotlib'):
                if self._theme is None:
                    cmap = pv.global_theme.cmap
                else:
                    cmap = self._theme.cmap

        if not isinstance(scalars, np.ndarray):
            scalars = np.asarray(scalars)

        # Set the array title for when it is added back to the mesh
        if custom_opac:
            scalars_name = '__custom_rgba'

        _using_labels = False
        if not np.issubdtype(scalars.dtype, np.number):

            # we can rapidly handle bools
            if scalars.dtype == np.bool_:
                cats = np.array([b'False', b'True'], dtype='|S5')
                values = np.array([0, 1])
                clim = [-0.5, 1.5]
            else:
                # If str array, digitive and annotate
                cats, scalars = np.unique(scalars.astype('|S'), return_inverse=True)
                values = np.unique(scalars)
                clim = [np.min(values) - 0.5, np.max(values) + 0.5]
                scalars_name = f'{scalars_name}-digitized'

            n_colors = len(cats)
            scalar_bar_args.setdefault('n_labels', 0)
            _using_labels = True

        # Use only the real component if an array is complex
        if np.issubdtype(scalars.dtype, np.complexfloating):
            scalars = scalars.astype(float)
            scalars_name = f'{scalars_name}-real'

        if scalars.ndim != 1:
            if rgb:
                pass
            elif scalars.ndim == 2 and (
                scalars.shape[0] == self.dataset.n_points
                or scalars.shape[0] == self.dataset.n_cells
            ):
                if not isinstance(component, (int, type(None))):
                    raise TypeError('component must be either None or an integer')
                if component is None:
                    scalars = np.linalg.norm(scalars.copy(), axis=1)
                    scalars_name = f'{scalars_name}-normed'
                elif component < scalars.shape[1] and component >= 0:
                    scalars = np.array(scalars[:, component]).copy()
                    scalars_name = f'{scalars_name}-{component}'
                else:
                    raise ValueError(
                        'Component must be nonnegative and less than the '
                        f'dimensionality of the scalars array: {scalars.shape[1]}'
                    )
            else:
                scalars = scalars.ravel()

        if scalars.dtype == np.bool_:
            scalars = scalars.astype(np.float_)

        table = self.lookup_table
        if _using_labels:
            table.SetAnnotations(convert_array(values), convert_string_array(cats))

        if isinstance(annotations, dict):
            for val, anno in annotations.items():
                table.SetAnnotation(float(val), str(anno))

        # Set scalars range
        if clim is None:
            clim = [np.nanmin(scalars), np.nanmax(scalars)]
        elif isinstance(clim, (int, float)):
            clim = [-clim, clim]

        if log_scale:
            if clim[0] <= 0:
                clim = [sys.float_info.min, clim[1]]
            table.SetScaleToLog10()

        if np.any(clim) and not rgb:
            self.scalar_range = clim[0], clim[1]

        table.SetNanColor(Color(nan_color).float_rgba)
        if above_color:
            table.SetUseAboveRangeColor(True)
            table.SetAboveRangeColor(*Color(above_color).float_rgba)
            scalar_bar_args.setdefault('above_label', 'Above')
        if below_color:
            table.SetUseBelowRangeColor(True)
            table.SetBelowRangeColor(*Color(below_color).float_rgba)
            scalar_bar_args.setdefault('below_label', 'Below')

        if cmap is not None:
            # have to add the attribute to pass it onward to some classes
            if isinstance(cmap, str):
                self.cmap = cmap
            # ipygany uses different colormaps
            if self._theme is None:
                jupyter_backend = pv.global_theme.jupyter_backend
            else:
                jupyter_backend = self._theme.jupyter_backend
            if jupyter_backend == 'ipygany':
                from ..jupyter.pv_ipygany import check_colormap

                check_colormap(cmap)
            else:
                if not has_module('matplotlib'):
                    cmap = None
                    logging.warning('Please install matplotlib for color maps.')

                cmap = get_cmap_safe(cmap)
                if categories:
                    if categories is True:
                        n_colors = len(np.unique(scalars))
                    elif isinstance(categories, int):
                        n_colors = categories
                ctable = cmap(np.linspace(0, 1, n_colors)) * 255
                ctable = ctable.astype(np.uint8)
                # Set opactities
                if isinstance(opacity, np.ndarray) and not custom_opac:
                    ctable[:, -1] = opacity
                if flip_scalars:
                    ctable = np.ascontiguousarray(ctable[::-1])
                table.SetTable(_vtk.numpy_to_vtk(ctable))
                if custom_opac:
                    # need to round the colors here since we're
                    # directly displaying the colors
                    hue = normalize(scalars, minimum=clim[0], maximum=clim[1])
                    scalars = np.round(hue * n_colors) / n_colors
                    scalars = cmap(scalars) * 255
                    scalars[:, -1] *= opacity
                    scalars = scalars.astype(np.uint8)

        else:  # no cmap specified
            if flip_scalars:
                table.SetHueRange(0.0, 0.66667)
            else:
                table.SetHueRange(0.66667, 0.0)

        self._configure_scalars_mode(
            scalars,
            scalars_name,
            n_colors,
            preference,
            rgb or custom_opac,
        )

    def set_custom_opacity(self, opacity, color, n_colors, preference='point'):
        """Set custom opacity.

        Parameters
        ----------
        opacity : numpy.ndarray
            Opacity array to color the dataset. Array length must match either
            the number of points or cells.

        color : color_like
            The color to use with the opacity array.

        n_colors : int
            Number of colors to use.

        preference : str, default: 'point'
            Either ``'point'`` or ``'cell'``. Used when the number of cells
            matches the number of points.

        """
        # Create a custom RGBA array to supply our opacity to
        if opacity.size == self.dataset.n_points:
            rgba = np.empty((self.dataset.n_points, 4), np.uint8)
        elif opacity.size == self.dataset.n_cells:
            rgba = np.empty((self.dataset.n_cells, 4), np.uint8)
        else:  # pragma: no cover
            raise ValueError(
                f"Opacity array size ({opacity.size}) does not equal "
                f"the number of points ({self.dataset.n_points}) or the "
                f"number of cells ({self.dataset.n_cells})."
            )

        if self._theme is not None:
            default_color = self._theme.color
        else:
            default_color = pv.global_theme.color

        rgba[:, :-1] = Color(color, default_color=default_color).int_rgb
        rgba[:, -1] = np.around(opacity * 255)

        self.color_mode = 'direct'
        self._configure_scalars_mode(rgba, '', n_colors, preference, True)

    def __repr__(self):
        """Representation of the mapper."""
        mapper_attr = [
            f'{type(self).__name__} ({hex(id(self))})',
            f'  Scalar visibility:           {self.scalar_visibility}',
            f'  Scalar range:                {self.scalar_range}',
            f'  Interpolate before mapping:  {self.interpolate_before_map}',
            f'  Scalar map mode:             {self.scalar_map_mode}',
            f'  Color mode:                  {self.color_mode}',
            '',
        ]

        mapper_attr.append('Attached dataset:')
        mapper_attr.append(str(self.dataset))

        return '\n'.join(mapper_attr)


@abstract_class
class _BaseVolumeMapper(_BaseMapper):
    """Volume mapper class to override methods and attributes for to volume mappers."""

    def __init__(self, theme=None):
        """Initialize this class."""
        super().__init__(theme)
        self._lut = None
        self._scalar_range = None

    @property
    def dataset(self):
        """Return or set the dataset assigned to this mapper."""
        return self.GetInputAsDataSet()

    @dataset.setter
    def dataset(self, new_dataset: 'pv.core.dataset.DataSet'):
        return self.SetInputData(new_dataset)

    @property
    def lookup_table(self):
        return self._lut

    @lookup_table.setter
    def lookup_table(self, lut):
        self._lut = lut

    @property
    def scalar_range(self):
        """Return or set the scalar range."""
        return self._scalar_range

    @scalar_range.setter
    def scalar_range(self, clim):
        if self.lookup_table is not None:
            self.lookup_table.SetRange(*clim)
        self._scalar_range = clim

    def __del__(self):
        self._lut = None


class FixedPointVolumeRayCastMapper(_vtk.vtkFixedPointVolumeRayCastMapper, _BaseVolumeMapper):
    """Wrap _vtk.vtkFixedPointVolumeRayCastMapper."""

    def __init__(self, theme=None):
        """Initialize this class."""
        super().__init__(theme)


class GPUVolumeRayCastMapper(_vtk.vtkGPUVolumeRayCastMapper, _BaseVolumeMapper):
    """Wrap _vtk.vtkGPUVolumeRayCastMapper."""

    def __init__(self, theme=None):
        """Initialize this class."""
        super().__init__(theme)


class OpenGLGPUVolumeRayCastMapper(_vtk.vtkOpenGLGPUVolumeRayCastMapper, _BaseVolumeMapper):
    """Wrap _vtk.vtkOpenGLGPUVolumeRayCastMapper."""

    def __init__(self, theme=None):
        """Initialize this class."""
        super().__init__(theme)


class SmartVolumeMapper(_vtk.vtkSmartVolumeMapper, _BaseVolumeMapper):
    """Wrap _vtk.vtkSmartVolumeMapper."""

    def __init__(self, theme=None):
        """Initialize this class."""
        super().__init__(theme)
