"""An internal module for wrapping the use of mappers."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING
from typing import cast

import numpy as np

import pyvista as pv
from pyvista._deprecate_positional_args import _deprecate_positional_args
from pyvista.core._typing_core import BoundsTuple
from pyvista.core.utilities.arrays import FieldAssociation
from pyvista.core.utilities.arrays import convert_array
from pyvista.core.utilities.arrays import convert_string_array
from pyvista.core.utilities.arrays import raise_not_matching
from pyvista.core.utilities.helpers import wrap
from pyvista.core.utilities.misc import _BoundsSizeMixin
from pyvista.core.utilities.misc import _NoNewAttrMixin
from pyvista.core.utilities.misc import abstract_class
from pyvista.core.utilities.state_manager import _update_alg

from . import _vtk
from .colors import Color
from .colors import get_cmap_safe
from .lookup_table import LookupTable
from .tools import normalize
from .utilities.algorithms import set_algorithm_input

if TYPE_CHECKING:
    from pyvista import DataSet
    from pyvista.themes import Theme


@abstract_class
class _BaseMapper(
    _NoNewAttrMixin, _BoundsSizeMixin, _vtk.DisableVtkSnakeCase, _vtk.vtkAbstractMapper
):
    """Base Mapper with methods common to other mappers."""

    def __init__(self, theme=None, **kwargs) -> None:
        self._theme = pv.themes.Theme()
        if theme is None:
            # copy global theme to ensure local property theme is fixed
            # after creation.
            self._theme.load_theme(pv.global_theme)
        else:
            self._theme.load_theme(theme)
        self.lookup_table = LookupTable()

        self.interpolate_before_map = kwargs.get(
            'interpolate_before_map',
            self._theme.interpolate_before_map,
        )

    @property
    def bounds(self) -> BoundsTuple:  # numpydoc ignore=RT01
        """Return the bounds of this mapper.

        Examples
        --------
        >>> import pyvista as pv
        >>> mapper = pv.DataSetMapper(dataset=pv.Cube())
        >>> mapper.bounds
        BoundsTuple(x_min = -0.5,
                    x_max =  0.5,
                    y_min = -0.5,
                    y_max =  0.5,
                    z_min = -0.5,
                    z_max =  0.5)

        """
        if self.dataset is None:
            with pv.vtk_verbosity('off'):
                return BoundsTuple(*self.GetBounds())
        return BoundsTuple(*self.GetBounds())

    @property
    def center(self) -> tuple[float, float, float]:
        """Return the center of mapper.

        Returns
        -------
        tuple[float, float, float]
            Center of the active renderer.

        """
        if self.dataset is None:
            with pv.vtk_verbosity('off'):
                return self.GetCenter()
        return self.GetCenter()

    def copy(self) -> _BaseMapper:
        """Create a copy of this mapper.

        Returns
        -------
        pyvista.DataSetMapper
            A copy of this dataset mapper.

        Examples
        --------
        >>> import pyvista as pv
        >>> mapper = pv.DataSetMapper(dataset=pv.Cube())
        >>> mapper_copy = mapper.copy()

        """
        new_mapper = type(self)(theme=self._theme)
        # even though this uses ShallowCopy, the new mapper no longer retains
        # any connection with the original
        new_mapper.ShallowCopy(self)
        if hasattr(self, 'dataset'):
            new_mapper.dataset = self.dataset
        return new_mapper

    @property
    def scalar_range(self) -> tuple[float, float]:  # numpydoc ignore=RT01
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
    def scalar_range(self, clim) -> None:
        self.SetScalarRange(*clim)

    @property
    def lookup_table(self) -> LookupTable:  # numpydoc ignore=RT01
        """Return or set the lookup table.

        Examples
        --------
        Return the lookup table of a dataset mapper.

        >>> import pyvista as pv
        >>> mesh = pv.Sphere()
        >>> pl = pv.Plotter()
        >>> actor = pl.add_mesh(mesh, scalars=mesh.points[:, 2], cmap='bwr')
        >>> actor.mapper.lookup_table
        LookupTable (...)
          Table Range:                (-0.5, 0.5)
          N Values:                   256
          Above Range Color:          None
          Below Range Color:          None
          NAN Color:                  Color(name='darkgray', hex='#a9a9a9ff', opacity=255)
          Log Scale:                  False
          Color Map:                  "bwr"

        Return the lookup table of a composite dataset mapper.

        >>> import pyvista as pv
        >>> dataset = pv.MultiBlock([pv.Cube(), pv.Sphere(center=(0, 0, 1))])
        >>> pl = pv.Plotter()
        >>> actor, mapper = pl.add_composite(dataset)
        >>> mapper.lookup_table  # doctest:+SKIP
        <vtkmodules.vtkCommonCore.vtkLookupTable(...) at ...>

        """
        return self.GetLookupTable()

    @lookup_table.setter
    def lookup_table(self, table) -> None:
        self.SetLookupTable(table)

    @property
    def color_mode(self) -> str:  # numpydoc ignore=RT01
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
            msg = 'Color mode must be either "default", "direct" or "map"'
            raise ValueError(msg)

    @property
    def interpolate_before_map(self) -> bool | None:  # numpydoc ignore=RT01
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
        ...     dataset,
        ...     show_scalar_bar=False,
        ...     n_colors=3,
        ...     cmap='bwr',
        ... )
        >>> mapper.interpolate_before_map = False
        >>> pl.show()

        Enable interpolation before mapping.

        >>> pl = pv.Plotter()
        >>> actor, mapper = pl.add_composite(
        ...     dataset,
        ...     show_scalar_bar=False,
        ...     n_colors=3,
        ...     cmap='bwr',
        ... )
        >>> mapper.interpolate_before_map = True
        >>> pl.show()

        See :ref:`interpolate_before_map_example` for additional
        explanation regarding this attribute.

        """
        return bool(self.GetInterpolateScalarsBeforeMapping())

    @interpolate_before_map.setter
    def interpolate_before_map(self, value: bool) -> None:
        self.SetInterpolateScalarsBeforeMapping(value)

    @property
    def array_name(self) -> str:  # numpydoc ignore=RT01
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
    def array_name(self, name: str) -> None:
        """Return or set the array name or number and component to color by."""
        self.SetArrayName(name)

    @property
    def scalar_map_mode(self) -> str:  # numpydoc ignore=RT01
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
        >>> actor, mapper = pl.add_composite(
        ...     dataset, scalars='data', show_scalar_bar=False
        ... )
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
    def scalar_map_mode(self, scalar_mode: str | FieldAssociation):
        if isinstance(scalar_mode, FieldAssociation):
            scalar_mode = scalar_mode.name
        scalar_mode = scalar_mode.lower()
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
            msg = (
                f'Invalid `scalar_map_mode` "{scalar_mode}". Should be either '
                '"default", "point", "cell", "point_field", "cell_field" or "field".'
            )
            raise ValueError(msg)

    @property
    def scalar_visibility(self) -> bool:  # numpydoc ignore=RT01
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
    def scalar_visibility(self, value: bool) -> None:
        self.SetScalarVisibility(value)

    def update(self) -> None:
        """Update this mapper."""
        _update_alg(self)


class _DataSetMapper(_BaseMapper):
    """Base wrapper for :vtk:`vtkDataSetMapper`.

    Parameters
    ----------
    dataset : pyvista.DataSet, optional
        Dataset to assign to this mapper.

    theme : pyvista.plotting.themes.Theme, optional
        Plot-specific theme.

    """

    _cmap = None

    def __init__(
        self,
        dataset: DataSet | None = None,
        theme: Theme | None = None,
    ) -> None:
        """Initialize this class."""
        super().__init__(theme=theme)
        if dataset is not None:
            self.dataset = dataset

    @property
    def dataset(self) -> DataSet | None:  # numpydoc ignore=RT01
        """Return or set the dataset assigned to this mapper."""
        return cast('pv.DataSet | None', wrap(_mapper_get_data_set_input(self)))

    @dataset.setter
    def dataset(
        self,
        obj: DataSet | _vtk.vtkAlgorithm | _vtk.vtkAlgorithmOutput,
    ) -> None:
        set_algorithm_input(self, obj)

    def as_rgba(self) -> None:
        """Convert the active scalars to RGBA.

        This method is used to convert the active scalars to a fixed RGBA array
        and is used for certain mappers that do not support the "map" color
        mode.

        """
        if self.color_mode == 'direct':
            return

        if self.dataset is not None:
            self.dataset.point_data.pop('__rgba__', None)
            self._configure_scalars_mode(
                scalars=self.lookup_table(self.dataset.active_scalars),
                scalars_name='__rgba__',
                preference=self.scalar_map_mode,
                direct_scalars_color_mode=True,
            )

    def _configure_scalars_mode(
        self,
        *,
        scalars,
        scalars_name,
        preference,
        direct_scalars_color_mode,
    ) -> None:
        """Configure scalar mode.

        Parameters
        ----------
        scalars : numpy.ndarray
            Array of scalars to assign to the mapper.

        scalars_name : str
            If the name of this array exists, scalars is ignored. Otherwise,
            the scalars will be added to the existing dataset and this
            parameter is the name to assign the scalars.

        preference : str
            Either ``'point'`` or ``'cell'``.

        direct_scalars_color_mode : bool
            When ``True``, scalars are treated as RGB colors. When
            ``False``, scalars are mapped to the color table.

        """
        if self.dataset is not None:
            if (
                scalars.shape[0] == self.dataset.n_points
                and scalars.shape[0] == self.dataset.n_cells
            ):
                use_points = preference == 'point'
                use_cells = not use_points
            else:
                use_points = scalars.shape[0] == self.dataset.n_points
                use_cells = scalars.shape[0] == self.dataset.n_cells

            # Scalars interpolation approach
            if use_points:
                if (
                    scalars_name not in self.dataset.point_data
                    or scalars_name == pv.DEFAULT_SCALARS_NAME
                ):
                    self.dataset.point_data.set_array(scalars, scalars_name, deep_copy=False)
                self.dataset.active_scalars_name = scalars_name
                self.scalar_map_mode = 'point'
            elif use_cells:
                if (
                    scalars_name not in self.dataset.cell_data
                    or scalars_name == pv.DEFAULT_SCALARS_NAME
                ):
                    self.dataset.cell_data.set_array(scalars, scalars_name, deep_copy=False)
                self.dataset.active_scalars_name = scalars_name
                self.scalar_map_mode = 'cell'
            else:
                raise_not_matching(scalars, self.dataset)

            self.color_mode = 'direct' if direct_scalars_color_mode else 'map'

    @_deprecate_positional_args(allowed=['scalars', 'scalars_name'])
    def set_scalars(  # noqa: PLR0917
        self,
        scalars,
        scalars_name,
        n_colors=256,
        scalar_bar_args=None,
        rgb=None,
        component=None,
        preference='point',
        custom_opac: bool = False,  # noqa: FBT001, FBT002
        annotations=None,
        log_scale: bool = False,  # noqa: FBT001, FBT002
        nan_color=None,
        above_color=None,
        below_color=None,
        cmap=None,
        flip_scalars: bool = False,  # noqa: FBT001, FBT002
        opacity=None,
        categories: bool | int = False,  # noqa: FBT001, FBT002
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
            :func:`pyvista.Plotter.add_scalar_bar`.

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
            and the values are the string annotations.

        log_scale : bool, default: False
            Use log scale when mapping data to colors. Scalars less
            than zero are mapped to the smallest representable
            positive float.

        nan_color : pyvista.ColorLike, optional
            The color to use for all ``NaN`` values in the plotted
            scalar array.

        above_color : pyvista.ColorLike, optional
            Solid color for values below the scalars range
            (``clim``). This will automatically set the scalar bar
            ``above_label`` to ``'above'``.

        below_color : pyvista.ColorLike, optional
            Solid color for values below the scalars range
            (``clim``). This will automatically set the scalar bar
            ``below_label`` to ``'below'``.

        cmap : str, list, or pyvista.LookupTable
            Name of the Matplotlib colormap to use when mapping the
            ``scalars``.  See available Matplotlib colormaps.  Only applicable
            for when displaying ``scalars``.
            ``colormap`` is also an accepted alias for this. If
            ``colorcet`` or ``cmocean`` are installed, their colormaps can be
            specified by name.

            You can also specify a list of colors to override an existing
            colormap with a custom one.  For example, to create a three color
            colormap you might specify ``['green', 'red', 'blue']``.

            This parameter also accepts a :class:`pyvista.LookupTable`. If this
            is set, all parameters controlling the color map like ``n_colors``
            will be ignored.

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

        clim : Sequence, optional
            Color bar range for scalars.  Defaults to minimum and
            maximum of scalars array.  Example: ``(-1, 2)``.

        """
        if scalar_bar_args is None:
            scalar_bar_args = {'n_colors': n_colors}

        if not isinstance(scalars, np.ndarray):
            scalars = np.asarray(scalars)

        # Set the array title for when it is added back to the mesh
        if custom_opac:
            scalars_name = '__custom_rgba'

        if not np.issubdtype(scalars.dtype, np.number) and not isinstance(
            cmap,
            pv.LookupTable,
        ):
            # we can rapidly handle bools
            if scalars.dtype == np.bool_:
                cats = np.array([b'False', b'True'], dtype='|S5')
                values = np.array([0, 1])
                clim = [-0.5, 1.5]
            else:
                # If str array, digitize and annotate
                cats, scalars = np.unique(scalars.astype('|S'), return_inverse=True)
                values = np.unique(scalars)
                clim = [np.min(values) - 0.5, np.max(values) + 0.5]
                scalars_name = f'{scalars_name}-digitized'

            n_colors = len(cats)
            scalar_bar_args.setdefault('n_labels', 0)

            self.lookup_table.SetAnnotations(convert_array(values), convert_string_array(cats))

        # Use only the real component if an array is complex
        if np.issubdtype(scalars.dtype, np.complexfloating):
            scalars = scalars.astype(float)
            scalars_name = f'{scalars_name}-real'

        if scalars.ndim != 1:
            if rgb:
                pass
            elif scalars.ndim == 2 and (
                scalars.shape[0] == self.dataset.n_points  # type: ignore[union-attr]
                or scalars.shape[0] == self.dataset.n_cells  # type: ignore[union-attr]
            ):
                if not isinstance(component, (int, type(None))):
                    msg = 'component must be either None or an integer'
                    raise TypeError(msg)
                if component is None:
                    scalars = np.linalg.norm(scalars.copy(), axis=1)
                    scalars_name = f'{scalars_name}-normed'
                elif component < scalars.shape[1] and component >= 0:
                    scalars = np.array(scalars[:, component]).copy()
                    scalars_name = f'{scalars_name}-{component}'
                else:
                    msg = (
                        'Component must be nonnegative and less than the '
                        f'dimensionality of the scalars array: {scalars.shape[1]}'
                    )
                    raise ValueError(msg)
            else:
                scalars = scalars.ravel()

        if scalars.dtype == np.bool_:
            scalars = scalars.astype(np.float64)

        # Set scalars range
        if clim is None:
            clim = [np.nanmin(scalars), np.nanmax(scalars)]
        elif isinstance(clim, (int, float)):
            clim = [-clim, clim]

        if log_scale and clim[0] <= 0:
            clim = [sys.float_info.min, clim[1]]

        if np.any(clim) and not rgb:
            self.scalar_range = clim[0], clim[1]

        if isinstance(cmap, pv.LookupTable):
            self.lookup_table = cmap
            self.scalar_range = self.lookup_table.scalar_range
        else:
            self.lookup_table.scalar_range = self.scalar_range
            # Set default map
            if cmap is None:
                cmap = pv.global_theme.cmap if self._theme is None else self._theme.cmap

            # have to add the attribute to pass it onward to some classes
            if isinstance(cmap, str):
                self._cmap = cmap
            if categories:
                if categories is True:
                    n_colors = len(np.unique(scalars))
                elif isinstance(categories, int):
                    n_colors = categories

            self.lookup_table.apply_cmap(cmap, n_colors)

            # Set opactities
            if isinstance(opacity, np.ndarray) and not custom_opac:
                self.lookup_table.apply_opacity(opacity)

            if flip_scalars:
                self.lookup_table.values[:] = self.lookup_table.values[::-1]

            if custom_opac:
                # need to round the colors here since we're
                # directly displaying the colors
                hue = normalize(scalars, minimum=clim[0], maximum=clim[1])
                scalars = np.round(hue * n_colors) / n_colors
                scalars = get_cmap_safe(cmap)(scalars) * 255
                scalars[:, -1] *= opacity
                scalars = scalars.astype(np.uint8)

            # configure the lookup table
            if nan_color:
                self.lookup_table.nan_color = nan_color
            if above_color:
                self.lookup_table.above_range_color = above_color
                scalar_bar_args.setdefault('above_label', 'above')
            if below_color:
                self.lookup_table.below_range_color = below_color
                scalar_bar_args.setdefault('below_label', 'below')
            if isinstance(annotations, dict):
                self.lookup_table.annotations = annotations
            self.lookup_table.log_scale = log_scale

        self._configure_scalars_mode(
            scalars=scalars,
            scalars_name=scalars_name,
            preference=preference,
            direct_scalars_color_mode=rgb or custom_opac,
        )

        if isinstance(self, PointGaussianMapper):
            self.as_rgba()

    @property
    def cmap(self) -> str | None:  # numpydoc ignore=RT01
        """Colormap assigned to this mapper.

        See :ref:`named_colormaps` for supported colormaps.
        """
        return self._cmap

    @property
    def resolve(self) -> str:
        """Set or return the global flag to avoid z-buffer resolution.

        A global flag that controls whether the coincident topology
        (e.g., a line on top of a polygon) is shifted to avoid
        z-buffer resolution (and hence rendering problems).

        If not off, there are two methods to choose from.
        `polygon_offset` uses graphics systems calls to shift polygons,
        lines, and points from each other.
        `shift_zbuffer` is a legacy method that is used to remap the z-buffer
        to distinguish vertices, lines, and polygons,
        but does not always produce acceptable results.
        You should only use the polygon_offset method (or none) at this point.

        Returns
        -------
        str
            Global flag to avoid z-buffer resolution.
            Must be either `off`, `polygon_offset` or `shift_zbuffer`.

        Examples
        --------
        >>> import pyvista as pv
        >>> from pyvista import examples

        >>> mesh = examples.download_tri_quadratic_hexahedron()
        >>> surface_sep = mesh.separate_cells().extract_surface(
        ...     nonlinear_subdivision=4
        ... )
        >>> edges = surface_sep.extract_feature_edges()
        >>> surface = mesh.extract_surface(nonlinear_subdivision=4)

        >>> pl = pv.Plotter()
        >>> _ = pl.add_mesh(surface, smooth_shading=True, split_sharp_edges=True)
        >>> actor = pl.add_mesh(edges, color='k', line_width=3)
        >>> actor.mapper.resolve = 'polygon_offset'
        >>> pl.show()

        """
        vtk_to_pv = {
            _vtk.VTK_RESOLVE_OFF: 'off',
            _vtk.VTK_RESOLVE_POLYGON_OFFSET: 'polygon_offset',
            _vtk.VTK_RESOLVE_SHIFT_ZBUFFER: 'shift_zbuffer',
        }
        return vtk_to_pv[self.GetResolveCoincidentTopology()]

    @resolve.setter
    def resolve(self, resolve):
        if resolve == 'off':
            self.SetResolveCoincidentTopologyToOff()
        elif resolve == 'polygon_offset':
            self.SetResolveCoincidentTopologyToPolygonOffset()
        elif resolve == 'shift_zbuffer':
            self.SetResolveCoincidentTopologyToShiftZBuffer()
        else:
            msg = 'Resolve must be either "off", "polygon_offset" or "shift_zbuffer"'
            raise ValueError(msg)

    @_deprecate_positional_args(allowed=['opacity'])
    def set_custom_opacity(  # noqa: PLR0917
        self, opacity, color, n_colors, preference='point'
    ):
        """Set custom opacity.

        Parameters
        ----------
        opacity : numpy.ndarray
            Opacity array to color the dataset. Array length must match either
            the number of points or cells.

        color : pyvista.ColorLike
            The color to use with the opacity array.

        n_colors : int
            Number of colors to use.

        preference : str, default: 'point'
            Either ``'point'`` or ``'cell'``. Used when the number of cells
            matches the number of points.

        """
        # Create a custom RGBA array to supply our opacity to
        if opacity.size == self.dataset.n_points:  # type: ignore[union-attr]
            rgba = np.empty((self.dataset.n_points, 4), np.uint8)  # type: ignore[union-attr]
        elif opacity.size == self.dataset.n_cells:  # type: ignore[union-attr]
            rgba = np.empty((self.dataset.n_cells, 4), np.uint8)  # type: ignore[union-attr]
        else:  # pragma: no cover
            msg = (
                f'Opacity array size ({opacity.size}) does not equal '
                f'the number of points ({self.dataset.n_points}) or the '  # type: ignore[union-attr]
                f'number of cells ({self.dataset.n_cells}).',  # type: ignore[union-attr]
            )
            raise ValueError(msg)

        default_color = self._theme.color if self._theme is not None else pv.global_theme.color

        rgba[:, :-1] = Color(color, default_color=default_color).int_rgb
        rgba[:, -1] = np.around(opacity * 255)

        self.color_mode = 'direct'
        self.lookup_table.n_values = n_colors
        self._configure_scalars_mode(
            scalars=rgba, scalars_name='', preference=preference, direct_scalars_color_mode=True
        )

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


class DataSetMapper(_DataSetMapper, _vtk.vtkDataSetMapper):
    """Wrap :vtk:`vtkDataSetMapper`.

    Parameters
    ----------
    dataset : pyvista.DataSet, optional
        Dataset to assign to this mapper.

    theme : pyvista.plotting.themes.Theme, optional
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
        dataset: DataSet | None = None,
        theme: Theme | None = None,
    ) -> None:
        """Initialize this class."""
        super().__init__(dataset=dataset, theme=theme)


class PointGaussianMapper(_DataSetMapper, _vtk.vtkPointGaussianMapper):
    """Wrap :vtk:`vtkPointGaussianMapper`.

    Parameters
    ----------
    theme : pyvista.Theme, optional
        The theme to be used.
    emissive : bool, optional
        Whether or not the point should appear emissive. Default is set by the
        theme's ``lighting_params.emissive``.
    scale_factor : float, default: 1.0
        Scale factor applied to the point size.

    """

    def __init__(self, theme=None, emissive=None, scale_factor=1.0) -> None:
        super().__init__(theme=theme)
        if emissive is None:
            emissive = self._theme.lighting_params.emissive
        self.emissive = emissive
        self.scale_factor = scale_factor

    @property
    def emissive(self) -> bool:  # numpydoc ignore=RT01
        """Set or return emissive.

        This treats points as emissive light sources. Two points that overlap
        will have their brightness combined.
        """
        return bool(self.GetEmissive())

    @emissive.setter
    def emissive(self, value: bool) -> None:
        self.SetEmissive(value)

    @property
    def scale_factor(self) -> float:  # numpydoc ignore=RT01
        """Set or return the scale factor.

        Ranges from 0 to 1. A value of 0 will cause the splats to be rendered
        as simple points. Defaults to 1.0.

        """
        return self.GetScaleFactor()

    @scale_factor.setter
    def scale_factor(self, value: float) -> None:
        self.SetScaleFactor(value)

    @property
    def scale_array(self) -> str:  # numpydoc ignore=RT01
        """Set or return the name of the array used to scale the splats.

        Scalars used to scale the gaussian points. Accepts a string
        name of an array that is present on the mesh.

        Notes
        -----
        Setting this automatically sets ``scale_factor = 1.0``.

        Examples
        --------
        Plot spheres using `style='points_gaussian'` style and scale them by
        radius.

        >>> import numpy as np
        >>> import pyvista as pv
        >>> n_spheres = 1_000
        >>> pos = np.random.random((n_spheres, 3))
        >>> rad = np.random.random(n_spheres) * 0.01
        >>> pdata = pv.PolyData(pos)
        >>> pdata['radius'] = rad
        >>> pl = pv.Plotter()
        >>> actor = pl.add_mesh(
        ...     pdata,
        ...     style='points_gaussian',
        ...     emissive=False,
        ...     render_points_as_spheres=True,
        ... )
        >>> actor.mapper.scale_array = 'radius'
        >>> pl.show()

        """
        return self.GetScaleArray()

    @scale_array.setter
    def scale_array(self, name: str):
        if not self.dataset:  # pragma: no cover
            msg = 'Missing dataset.'
            raise RuntimeError(msg)
        if name not in self.dataset.point_data:
            available_arrays = ', '.join(self.dataset.point_data.keys())
            msg = (
                f'Point array "{name}" does not exist. '
                f'Available point arrays are: {available_arrays}'
            )
            raise KeyError(msg)

        self.scale_factor = 1.0
        self.SetScaleArray(name)

    def use_circular_splat(self, opacity: float = 1.0) -> None:
        """Set the fragment shader code to create a circular splat.

        Parameters
        ----------
        opacity : float, default: 1.0
            Desired opacity between 0 and 1.

        Notes
        -----
        This very close to ParaView's PointGaussianMapper, but uses opacity to
        modify the scale as the opacity cannot be set from the actor's property.

        """
        self.SetSplatShaderCode(
            '//VTK::Color::Impl\n'
            'float dist = dot(offsetVCVSOutput.xy,offsetVCVSOutput.xy);\n'
            'if (dist > 1.0) {\n'
            '  discard;\n'
            '} else {\n'
            f'  float scale = ({opacity} - dist);\n'
            '  ambientColor *= scale;\n'
            '  diffuseColor *= scale;\n'
            '}\n',
        )
        # maintain consistency with the default style
        self.scale_factor *= 1.5

    def use_default_splat(self) -> None:
        """Clear the fragment shader and use the default splat."""
        self.SetSplatShaderCode(None)  # type: ignore[arg-type]
        self.scale_factor /= 1.5

    def __repr__(self):
        """Representation of the Gaussian mapper."""
        mapper_attr = [
            f'{type(self).__name__} ({hex(id(self))})',
            f'  Scalar visibility:           {self.scalar_visibility}',
            f'  Scalar range:                {self.scalar_range}',
            f'  Emissive:                    {self.emissive}',
            f'  Scale Factor:                {self.scale_factor}',
            f'  Using custom splat:          {self.GetSplatShaderCode() is None}',
            '',
        ]

        mapper_attr.append('Attached dataset:')
        mapper_attr.append(str(self.dataset))

        return '\n'.join(mapper_attr)


@abstract_class
class _BaseVolumeMapper(_BaseMapper):
    """Volume mapper class to override methods and attributes for to volume mappers."""

    def __init__(self, theme=None) -> None:
        """Initialize this class."""
        super().__init__(theme=theme)
        self._lut = LookupTable()
        self._scalar_range = (0.0, 256.0)

    @property
    def interpolate_before_map(self) -> bool | None:  # numpydoc ignore=RT01
        """Interpolate before map is not supported with volume mappers."""
        return None

    @interpolate_before_map.setter
    def interpolate_before_map(self, *args) -> None:
        pass

    @property
    def dataset(self):  # numpydoc ignore=RT01
        """Return or set the dataset assigned to this mapper."""
        return wrap(_mapper_get_data_set_input(self))

    @dataset.setter
    def dataset(
        self,
        obj: DataSet | _vtk.vtkAlgorithm | _vtk.vtkAlgorithmOutput,
    ) -> None:
        set_algorithm_input(self, obj)

    @property
    def lookup_table(self):  # numpydoc ignore=RT01
        return self._lut

    @lookup_table.setter
    def lookup_table(self, lut) -> None:
        self._lut = lut

    @property
    def scalar_range(self) -> tuple[float, float]:  # numpydoc ignore=RT01
        """Return or set the scalar range."""
        return self._scalar_range

    @scalar_range.setter
    def scalar_range(self, clim) -> None:
        if self.lookup_table is not None:
            self.lookup_table.SetRange(*clim)
        self._scalar_range = tuple(clim)

    @property
    def blend_mode(self) -> str:  # numpydoc ignore=RT01
        """Return or set the blend mode.

        One of the following:

        * ``"composite"``
        * ``"maximum"``
        * ``"minimum"``
        * ``"average"``
        * ``"additive"``

        Also accepts integer values corresponding to
        :vtk:`vtkVolumeMapper.BlendModes`. For example
        :vtk:`vtkVolumeMapper.COMPOSITE_BLEND`.

        """
        value = self.GetBlendMode()
        mode = {
            0: 'composite',
            1: 'maximum',
            2: 'minimum',
            3: 'average',
            4: 'additive',
        }.get(value)
        if mode is None:  # pragma: no cover
            msg = f'Unsupported blend mode return value {value}'
            raise NotImplementedError(msg)
        return mode

    @blend_mode.setter
    def blend_mode(self, value: str | int):
        if isinstance(value, int):
            self.SetBlendMode(value)
        elif isinstance(value, str):
            value = value.lower()
            if value in ['additive', 'add', 'sum']:
                self.SetBlendModeToAdditive()
            elif value in ['average', 'avg', 'average_intensity']:
                self.SetBlendModeToAverageIntensity()
            elif value in ['composite', 'comp']:
                self.SetBlendModeToComposite()
            elif value in ['maximum', 'max', 'maximum_intensity']:
                self.SetBlendModeToMaximumIntensity()
            elif value in ['minimum', 'min', 'minimum_intensity']:
                self.SetBlendModeToMinimumIntensity()
            else:
                msg = (
                    f'Blending mode {value!r} invalid. '
                    'Please choose either "additive", '
                    '"composite", "minimum" or "maximum".'
                )
                raise ValueError(msg)
        else:
            msg = f'`blend_mode` should be either an int or str, not `{type(value)}`'  # type: ignore[unreachable]
            raise TypeError(msg)

    def __del__(self) -> None:
        if hasattr(self, '_lut'):
            del self._lut


class FixedPointVolumeRayCastMapper(_BaseVolumeMapper, _vtk.vtkFixedPointVolumeRayCastMapper):
    """Wrap :vtk:`vtkFixedPointVolumeRayCastMapper`."""

    def __init__(self, theme=None) -> None:
        """Initialize this class."""
        super().__init__(theme=theme)
        self.AutoAdjustSampleDistancesOff()


class GPUVolumeRayCastMapper(_BaseVolumeMapper, _vtk.vtkGPUVolumeRayCastMapper):
    """Wrap :vtk:`vtkGPUVolumeRayCastMapper`."""

    def __init__(self, theme=None) -> None:
        """Initialize this class."""
        super().__init__(theme=theme)
        self.AutoAdjustSampleDistancesOff()


class OpenGLGPUVolumeRayCastMapper(_BaseVolumeMapper, _vtk.vtkOpenGLGPUVolumeRayCastMapper):
    """Wrap :vtk:`vtkOpenGLGPUVolumeRayCastMapper`."""

    def __init__(self, theme=None) -> None:
        """Initialize this class."""
        super().__init__(theme=theme)
        self.AutoAdjustSampleDistancesOff()


class SmartVolumeMapper(_BaseVolumeMapper, _vtk.vtkSmartVolumeMapper):
    """Wrap :vtk:`vtkSmartVolumeMapper`."""

    def __init__(self, theme=None) -> None:
        """Initialize this class."""
        super().__init__(theme=theme)
        self.AutoAdjustSampleDistancesOff()
        self.InteractiveAdjustSampleDistancesOff()


class UnstructuredGridVolumeRayCastMapper(
    _BaseVolumeMapper, _vtk.vtkUnstructuredGridVolumeRayCastMapper
):
    """Wrap :vtk:`vtkUnstructuredGridVolumeMapper`."""

    def __init__(self, theme=None) -> None:
        """Initialize this class."""
        super().__init__(theme=theme)
        self.AutoAdjustSampleDistancesOff()


def _mapper_has_data_set_input(mapper):
    """Check if mapper has a data set input using the appropriate method.

    Some mappers use 'GetDataSetInput', others use 'GetInputAsDataSet'. This has
    been standardized to 'GetDataSetInput' in VTK >= 9.5.
    """
    return hasattr(mapper, 'GetDataSetInput') or hasattr(mapper, 'GetInputAsDataSet')


def _mapper_get_data_set_input(mapper) -> _vtk.vtkDataSet:
    """Get data set input from mapper using the appropriate method.

    Some mappers use 'GetDataSetInput', others use 'GetInputAsDataSet'. This has
    been standardized to 'GetDataSetInput' in VTK >= 9.5.
    """
    return (
        mapper.GetDataSetInput()
        if hasattr(mapper, 'GetDataSetInput')
        else mapper.GetInputAsDataSet()
    )
