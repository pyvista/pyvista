"""Wrap vtkLookupTable."""
from typing import Any, Optional, Tuple, Union

import numpy as np

import pyvista as pv
from pyvista import _vtk
from pyvista.utilities.helpers import convert_array
from pyvista.utilities.misc import has_module, no_new_attr

from .._typing import color_like
from .colors import Color, get_cmap_safe
from .tools import opacity_transfer_function

RAMP_MAP = {0: 'linear', 1: 's-curve', 2: 'sqrt'}
RAMP_MAP_INV = {k: v for v, k in RAMP_MAP.items()}


class lookup_table_ndarray(np.ndarray):
    """An ndarray which references the owning table and the underlying vtkArray.

    This class is used to ensure that the internal vtkLookupTable updates when
    the values array is updated.

    """

    def __new__(
        cls,
        array,
        table=None,
    ):
        """Allocate the array."""
        obj = convert_array(array).view(cls)
        obj.VTKObject = array

        obj.table = _vtk.vtkWeakReference()
        obj.table.Set(table)

        return obj

    def __array_finalize__(self, obj):
        """Finalize array (associate with parent metadata)."""
        _vtk.VTKArray.__array_finalize__(self, obj)
        if np.shares_memory(self, obj):
            self.table = getattr(obj, 'table', None)
            self.VTKObject = getattr(obj, 'VTKObject', None)
        else:
            self.table = None
            self.VTKObject = None

    def __setitem__(self, key, value):
        """Implement [] set operator.

        When the array is changed it triggers "Modified()" which updates
        all upstream objects, including any render windows holding the
        object.
        """
        super().__setitem__(key, value)
        if self.VTKObject is not None:
            self.VTKObject.Modified()

        # the associated dataset should also be marked as modified
        if self.table is not None and self.table.Get():
            # this creates a new shallow copy and is necessary to update the
            # internal VTK array
            self.table.Get().values = self

    def __array_wrap__(self, out_arr, context=None):
        """Return a numpy scalar if array is 0d.

        See https://github.com/numpy/numpy/issues/5819

        """
        if out_arr.ndim:
            return np.ndarray.__array_wrap__(self, out_arr, context)

        # Match numpy's behavior and return a numpy dtype scalar
        return out_arr[()]

    __getattr__ = _vtk.VTKArray.__getattr__


@no_new_attr
class LookupTable(_vtk.vtkLookupTable):
    """Scalar to RGBA mapping table.

    A lookup table is an array that maps input values to output values. When
    plotting data over a dataset, it is necessary to map those scalars to
    colors (in the RGBA format), and this class provides the functionality to
    do so.

    See `vtkLookupTable
    <https://vtk.org/doc/nightly/html/classvtkLookupTable.html>`_ for more
    details regarding the underlying VTK API.

    Parameters
    ----------
    cmap : str, colors.Colormap, optional
        Color map from ``matplotlib``, ``colorcet``, or ``cmocean``. Either
        ``cmap`` or ``values`` can be set, but not both.

    n_values : int, default: 256
        Number of colors in the color map.

    flip : bool, default: False
        Flip the direction of cmap. Most colormaps allow ``*_r`` suffix to do this
        as well.

    values : numpy.ndarray, optional
        Lookup table values. Either ``values`` or ``cmap`` can be set, but not
        both.

    value_range : tuple, optional
        The range of the brightness of the mapped lookup table. This range is
        only used when creating custom color maps and will be ignored if
        ``cmap`` is set.

    hue_range : tuple, optional
        Lookup table hue range. This range is only used when creating custom
        color maps and will be ignored if ``cmap`` is set.

    alpha_range : tuple, optional
        Lookup table alpha (transparency) range. This range is only used when
        creating custom color maps and will be ignored if ``cmap`` is set.

    scalar_range : tuple, optional
        The range of scalars which will be mapped to colors. Values outside of
        this range will be colored according to
        :attr`LookupTable.below_range_color` and
        :attr`LookupTable.above_range_color`.

    log_scale : bool, optional
        Use a log scale when mapping scalar values.

    nan_color : color_like, optional
        Color to render any values that are NANs.

    above_range_color : color_like, optional
        Color to render any values above :attr:`LookupTable.scalar_range`.

    below_range_color : color_like, optional
        Color to render any values below :attr:`LookupTable.scalar_range`.

    ramp : str, optional
        The shape of the table ramp. This range is only used when creating
        custom color maps and will be ignored if ``cmap`` is set.

    annotations : dict, optional
        A dictionary of annotations. Keys are the float values in the scalars
        range to annotate on the scalar bar and the values are the the string
        annotations.

    Examples
    --------
    Plot the lookup table with the default VTK color map.

    >>> import pyvista as pv
    >>> lut = pv.LookupTable()
    >>> lut  # doctest:+SKIP
    LookupTable (0x7ff3de60d580)
      Table Range:                (0.0, 1.0)
      N Values:                   256
      Above Range Color:          None
      Below Range Color:          None
      NAN Color:                  Color(name='maroon', hex='#800000ff')
      Log Scale:                  False
      Color Map:                  "VTK lookup table"
        Alpha Range:              (1.0, 1.0)
        Hue Range:                (0.0, 0.66667)
        Saturation Range          (1.0, 1.0)
        Value Range               (1.0, 1.0)
        Ramp                      s-curve
        Is Opaque                 True
    >>> lut.plot()

    Plot the lookup table with the ``'inferno'`` color map.

    >>> import pyvista as pv
    >>> lut = pv.LookupTable('inferno', n_values=32)
    >>> lut  # doctest:+SKIP
    LookupTable (0x7ff3c053f3a0)
      Table Range:                (0.0, 1.0)
      N Values:                   32
      Above Range Color:          None
      Below Range Color:          None
      NAN Color:                  Color(name='maroon', hex='#800000ff')
      Log Scale:                  False
      Color Map:                  "inferno"
    >>> lut.plot()

    """

    _nan_color_set = False
    _cmap = None
    _values_manual = False
    _opacity_parm: Tuple[Any, bool, str] = (None, False, 'quadratic')

    def __init__(
        self,
        cmap=None,
        n_values=256,
        flip=False,
        values=None,
        value_range=None,
        hue_range=None,
        alpha_range=None,
        scalar_range=None,
        log_scale=None,
        nan_color=None,
        above_range_color=None,
        below_range_color=None,
        ramp=None,
        annotations=None,
    ):
        """Initialize the lookup table."""
        if cmap is not None and values is not None:
            raise ValueError('Cannot set both `cmap` and `values`.')

        if cmap is not None:
            self.apply_cmap(cmap, n_values=n_values, flip=flip)
        elif values is not None:
            self.values = values
        else:
            self.n_values = n_values
            if value_range is not None:
                self.value_range = value_range
            if hue_range is not None:
                self.hue_range = hue_range
            if alpha_range is not None:
                self.alpha_range = alpha_range
            if ramp is not None:
                self.ramp = ramp

        if nan_color is not None:
            self.nan_color = nan_color
        if above_range_color is not None:
            self.above_range_color = above_range_color
        if below_range_color is not None:
            self.below_range_color = below_range_color
        if scalar_range is not None:
            self.scalar_range = scalar_range
        if log_scale is not None:
            self.log_scale = log_scale
        if annotations is not None:
            self.annotations = annotations

    @property
    def value_range(self) -> Optional[tuple]:
        """Return or set the brightness of the mapped lookup table.

        This range is only used when creating custom color maps and will return
        ``None`` when a color map has been set with :attr:`LookupTable.cmap`.

        This will clear any existing color map and create new values for the
        lookup table when set.

        Examples
        --------
        Show the effect of setting the value range on the default color
        map.

        >>> import pyvista as pv
        >>> lut = pv.LookupTable()
        >>> lut.value_range = (0, 1.0)
        >>> lut.plot()

        Demonstrate a different value range.

        >>> import pyvista as pv
        >>> lut = pv.LookupTable()
        >>> lut.value_range = (0.5, 0.8)
        >>> lut.plot()

        """
        if self._cmap:
            return
        return self.GetValueRange()

    @value_range.setter
    def value_range(self, value: tuple):
        self.SetValueRange(value)
        self.rebuild()

    @property
    def hue_range(self) -> Optional[tuple]:
        """Return or set the hue range.

        This range is only used when creating custom color maps and will return
        ``None`` when a color map has been set with :attr:`LookupTable.cmap`.

        This will clear any existing color map and create new values for the
        lookup table when set.

        Examples
        --------
        Set the hue range. This allows you to create a lookup table
        without setting a color map.

        >>> import pyvista as pv
        >>> lut = pv.LookupTable()
        >>> lut.hue_range = (0, 0.1)
        >>> lut.plot()

        Create a different color map.

        >>> import pyvista as pv
        >>> lut = pv.LookupTable()
        >>> lut.hue_range = (0.5, 0.8)
        >>> lut.plot()

        """
        if self._cmap:
            return
        return self.GetHueRange()

    @hue_range.setter
    def hue_range(self, value: tuple):
        self.SetHueRange(value)
        self.rebuild()

    @property
    def cmap(self) -> Optional[str]:
        """Return or set the color map used by this lookup table.

        Examples
        --------
        Apply the single Matplotlib color map ``"Oranges"``.

        >>> import pyvista as pv
        >>> lut = pv.LookupTable()
        >>> lut.cmap = 'Oranges'
        >>> lut.plot()

        Apply a list of colors as a colormap.

        >>> import pyvista as pv
        >>> lut = pv.LookupTable()
        >>> lut.cmap = ['black', 'red', 'orange']
        >>> lut.plot()

        """
        return self._cmap

    @cmap.setter
    def cmap(self, value):
        self.apply_cmap(value, self.n_values)

    @property
    def log_scale(self) -> bool:
        """Use log scale.

        When ``True`` the lookup table is a log scale to
        :attr:`LookupTable.scalar_range`. Otherwise, it is linear scale.

        Examples
        --------
        Use log scale for the lookup table.

        >>> import pyvista as pv
        >>> lut = pv.LookupTable()
        >>> lut.log_scale = True
        >>> lut.scalar_range = (1, 100)
        >>> lut.plot()

        """
        return bool(self.GetScale())

    @log_scale.setter
    def log_scale(self, value: bool):
        self.SetScale(value)

    def __repr__(self):
        """Return the representation."""
        lines = [f'{type(self).__name__} ({hex(id(self))})']
        lines.append(f'  Table Range:                {self.scalar_range}')
        lines.append(f'  N Values:                   {self.n_values}')
        lines.append(f'  Above Range Color:          {self.above_range_color}')
        lines.append(f'  Below Range Color:          {self.below_range_color}')
        lines.append(f'  NAN Color:                  {self.nan_color}')
        lines.append(f'  Log Scale:                  {self.log_scale}')

        lines.append(f'  Color Map:                  "{self._lookup_type}"')
        if not (self.cmap or self._values_manual):
            lines.append(f'    Alpha Range:              {self.alpha_range}')
            lines.append(f'    Hue Range:                {self.hue_range}')
            lines.append(f'    Saturation Range          {self.saturation_range}')
            lines.append(f'    Value Range               {self.value_range}')
            lines.append(f'    Ramp                      {self.ramp}')

        return '\n'.join(lines)

    @property
    def scalar_range(self) -> tuple:
        """Return or set the table range.

        This is the range of scalars which will be mapped to colors. Values
        outside of this range will be colored according to
        :attr`LookupTable.below_range_color` and
        :attr`LookupTable.above_range_color`.

        Examples
        --------
        >>> import pyvista as pv
        >>> lut = pv.LookupTable()
        >>> lut.scalar_range = (0, 10)
        >>> lut.scalar_range
        (0.0, 10.0)

        """
        return self.GetTableRange()

    @scalar_range.setter
    def scalar_range(self, value: tuple):
        self.SetTableRange(value)

    @property
    def alpha_range(self) -> tuple:
        """Return or set the alpha range.

        This range is only used when creating custom color maps and will return
        ``None`` when a color map has been set with :attr:`LookupTable.cmap`.

        This will clear any existing color map and create new values for the
        lookup table when set.

        Examples
        --------
        Create a custom "blues" lookup table that decreases in opacity.

        >>> import pyvista as pv
        >>> lut = pv.LookupTable()
        >>> lut.hue_range = (0.7, 0.7)
        >>> lut.alpha_range = (1.0, 0.0)
        >>> lut.plot(background='grey')

        """
        if self._cmap:
            return None
        return self.GetAlphaRange()

    @alpha_range.setter
    def alpha_range(self, value: tuple):
        self.SetAlphaRange(value)
        self.rebuild()

    @property
    def saturation_range(self) -> tuple:
        """Return or set the saturation range.

        This range is only used when creating custom color maps and will return
        ``None`` when a color map has been set with :attr:`LookupTable.cmap`.

        This will clear any existing color map and create new values for the
        lookup table when set.

        Examples
        --------
        Create a custom "blues" lookup table that increases in saturation.

        >>> import pyvista as pv
        >>> lut = pv.LookupTable()
        >>> lut.hue_range = (0.7, 0.7)
        >>> lut.saturation_range = (0.0, 1.0)
        >>> lut.plot(background='grey')

        """
        if self._cmap:
            return None
        return self.GetSaturationRange()

    @saturation_range.setter
    def saturation_range(self, value: tuple):
        self.SetSaturationRange(value)
        self.rebuild()

    def rebuild(self):
        """Clear the color map and recompute the values table.

        This is called automatically when setting values like
        :attr:`LookupTable.value_range`.

        Notes
        -----
        This will reset any colormap set with :func:`LookupTable.apply_cmap` or
        :attr:`LookupTable.values`.

        """
        self._cmap = None
        self._values_manual = False
        self.ForceBuild()

    @property
    def nan_color(self) -> Optional[Color]:
        """Return or set the not a number (NAN) color.

        Any values that are NANs will be rendered with this color.

        Examples
        --------
        Set the NAN color to ``'grey'``.

        >>> import pyvista as pv
        >>> lut = pv.LookupTable()
        >>> lut.nan_color = 'grey'
        >>> lut.plot()

        """
        return Color(self.GetNanColor())

    @nan_color.setter
    def nan_color(self, value):
        # NAN value is always set, but make it explicit for example plotting
        self._nan_color_set = True
        self.SetNanColor(*Color(value).float_rgba)

    @property
    def ramp(self) -> str:
        """Set the shape of the table ramp.

        This attribute is only used when creating custom color maps and will
        return ``None`` when a color map has been set with
        :attr:`LookupTable.cmap`. This will clear any existing color map and
        create new values for the lookup table when set.

        This value may be either ``"s-curve"``, ``"linear"``, or ``"sqrt"``.

        * The default is S-curve, which tails off gradually at either end.
        * The equation used for ``"s-curve"`` is ``y = (sin((x - 1/2)*pi) +
          1)/2``, For an S-curve greyscale ramp, you should set
          :attr:`LookupTable.n_values`` to 402 (which is ``256*pi/2``) to provide
          room for the tails of the ramp.

        * The equation for the ``"linear"`` is simply ``y = x``.
        * The equation for the ``"sqrt"`` is ``y = sqrt(x)``.

        Examples
        --------
        Show the default s-curve ramp.

        >>> import pyvista as pv
        >>> lut = pv.LookupTable()
        >>> lut.hue_range = (0.0, 0.33)
        >>> lut.ramp = 's-curve'
        >>> lut.plot()

        Plot the linear ramp.

        >>> import pyvista as pv
        >>> lut = pv.LookupTable()
        >>> lut.hue_range = (0.0, 0.33)
        >>> lut.ramp = 'linear'
        >>> lut.plot()

        Plot the ``"sqrt"`` ramp.

        >>> import pyvista as pv
        >>> lut = pv.LookupTable()
        >>> lut.hue_range = (0.0, 0.33)
        >>> lut.ramp = 'sqrt'
        >>> lut.plot()

        """
        return RAMP_MAP[self.GetRamp()]

    @ramp.setter
    def ramp(self, value: str):
        try:
            self.SetRamp(RAMP_MAP_INV[value])
        except KeyError:
            raise ValueError(f'`ramp` must be one of the following:\n{list(RAMP_MAP_INV.keys())}')
        self.rebuild()

    @property
    def above_range_color(self) -> Optional[Color]:
        """Return or set the above range color.

        Any values above :attr:`LookupTable.scalar_range` will be rendered with this
        color.

        Examples
        --------
        Enable the usage of the above range color.

        >>> import pyvista as pv
        >>> lut = pv.LookupTable()
        >>> lut.above_range_color = 'blue'
        >>> lut.plot()

        Disable the usage of the above range color.

        >>> import pyvista as pv
        >>> lut = pv.LookupTable()
        >>> lut.above_range_color = None
        >>> lut.plot()

        """
        if self.GetUseAboveRangeColor():
            return Color(self.GetAboveRangeColor())
        return None

    @above_range_color.setter
    def above_range_color(self, value: Union[bool, color_like]):
        if value in (None, False):
            self.SetUseAboveRangeColor(False)
        elif value is True:
            self.SetAboveRangeColor(*Color(pv.global_theme.above_range_color).float_rgba)
            self.SetUseAboveRangeColor(True)
        else:
            self.SetAboveRangeColor(*Color(value).float_rgba)
            self.SetUseAboveRangeColor(True)

    @property
    def below_range_color(self) -> Optional[Color]:
        """Return or set the below range color.

        Any values below :attr:`LookupTable.scalar_range` will be rendered with this
        color.

        Examples
        --------
        Enable the usage of the below range color.

        >>> import pyvista as pv
        >>> lut = pv.LookupTable()
        >>> lut.below_range_color = 'blue'
        >>> lut.plot()

        Disable the usage of the below range color.

        >>> import pyvista as pv
        >>> lut = pv.LookupTable()
        >>> lut.below_range_color = None
        >>> lut.plot()

        """
        if self.GetUseBelowRangeColor():
            return Color(self.GetBelowRangeColor())
        return None

    @below_range_color.setter
    def below_range_color(self, value: Union[bool, color_like]):
        if value in (None, False):
            self.SetUseBelowRangeColor(False)
        elif value is True:
            self.SetBelowRangeColor(*Color(pv.global_theme.above_range_color).float_rgba)
            self.SetUseBelowRangeColor(True)
        else:
            self.SetBelowRangeColor(*Color(value).float_rgba)
            self.SetUseBelowRangeColor(True)

    def apply_cmap(self, cmap, n_values: int = 256, flip: bool = False):
        """Assign a colormap to this lookup table.

        This can be used instead of :attr:`LookupTable.cmap` when you need to
        set the number of values at the same time as the color map.

        Parameters
        ----------
        cmap : str, list, colors.Colormap
            Colormap from Matplotlib, colorcet, or cmocean.

        n_values : int, default: 256
            Number of colors in the color map.

        flip : bool, default: False
            Flip direction of cmap. Most colormaps allow ``*_r`` suffix to do
            this as well.

        Examples
        --------
        Apply ``matplotlib``'s ``'cividis'`` color map.

        >>> import pyvista as pv
        >>> lut = pv.LookupTable()
        >>> lut.apply_cmap('cividis', n_values=32)
        >>> lut.plot()

        """
        if not has_module('matplotlib'):  # pragma: no cover
            raise ModuleNotFoundError('Install Matplotlib to use color maps.')

        if isinstance(cmap, list):
            n_values = len(cmap)

        cmap = get_cmap_safe(cmap)
        values = cmap(np.linspace(0, 1, n_values)) * 255

        if flip:
            values = values[::-1]

        self.values = values
        self._values_manual = False

        # reapply the opacity
        if self._opacity_parm[0] is not None:
            self.apply_opacity(*self._opacity_parm)

        self._cmap = cmap

    def apply_opacity(self, opacity, interpolate: bool = True, kind: str = 'quadratic'):
        """Assign custom opacity to this lookup table.

        Parameters
        ----------
        opacity : float, list(float), str
            The opacity mapping to use. Can be a ``str`` name of a predefined
            mapping including ``'linear'``, ``'geom'``, ``'sigmoid'``,
            ``'sigmoid_3-10'``.  Append an ``'_r'`` to any of those names to
            reverse that mapping.  This can also be a custom array or list of
            values that will be interpolated across the ``n_color`` range for
            user defined mappings. Values must be between 0 and 1.

            If a ``float``, simply applies the same opacity across the entire
            colormap and must be between 0 and 1. Note that ``int`` values are
            interpreted as if they were floats.

        interpolate : bool, default: True
            Flag on whether or not to interpolate the opacity mapping for all
            colors.

        kind : str, default: 'quadratic'
            The interpolation kind if ``interpolate`` is ``True`` and ``scipy``
            is available. See :class:`scipy.interpolate.interp1d` for the
            available interpolation kinds.

            If ``scipy`` is not available, ``'linear'`` interpolation is used.

        Examples
        --------
        Apply a user defined custom opacity to a lookup table and plot the
        random hills example.

        >>> import pyvista as pv
        >>> from pyvista import examples
        >>> mesh = examples.load_random_hills()
        >>> lut = pv.LookupTable(cmap='viridis')
        >>> lut.apply_opacity([1.0, 0.4, 0.0, 0.4, 0.9])
        >>> lut.scalar_range = (mesh.active_scalars.min(), mesh.active_scalars.max())
        >>> pl = pv.Plotter()
        >>> _ = pl.add_mesh(mesh, cmap=lut)
        >>> pl.show()

        """
        if isinstance(opacity, (float, int)):
            if not 0 <= opacity <= 1:
                raise ValueError(f'Opacity must be between 0 and 1, got {opacity}')
            self.values[:, -1] = opacity * 255
        elif len(opacity) == self.n_values:
            # no interpolation is necessary
            self.values[:, -1] = np.array(opacity)
        else:
            self.values[:, -1] = opacity_transfer_function(
                opacity, self.n_values, interpolate=interpolate, kind=kind
            )
        self._opacity_parm = (opacity, interpolate, kind)

    @property
    def values(self) -> lookup_table_ndarray:
        """Return or set the lookup table values.

        This attribute is used when creating a custom lookup table. The table
        must be a RGBA array shaped ``(n, 4)``.

        Examples
        --------
        Create a simple four value lookup table ranging from black to red.

        >>> import pyvista as pv
        >>> lut = pv.LookupTable()
        >>> lut.values = [
        ...     [0, 0, 0, 255],
        ...     [85, 0, 0, 255],
        ...     [170, 0, 0, 255],
        ...     [255, 0, 0, 255],
        ... ]
        >>> lut.values
        lookup_table_ndarray([[  0,   0,   0, 255],
                              [ 85,   0,   0, 255],
                              [170,   0,   0, 255],
                              [255,   0,   0, 255]], dtype=uint8)
        >>> lut.plot()

        """
        return lookup_table_ndarray(self.GetTable(), table=self)

    @values.setter
    def values(self, new_values):
        self._values_manual = True
        self._cmap = None
        new_values = np.array(new_values, copy=False).astype(np.uint8, copy=False)
        self.SetTable(_vtk.numpy_to_vtk(new_values))

    @property
    def n_values(self) -> int:
        """Return or set the number of values in the lookup table.

        Examples
        --------
        Plot the ``"reds"`` colormap with 10 values.

        >>> import pyvista as pv
        >>> lut = pv.LookupTable('reds')
        >>> lut.n_values = 10
        >>> lut.plot()

        Plot the default colormap with 1024 values.

        >>> import pyvista as pv
        >>> lut = pv.LookupTable()
        >>> lut.n_values = 1024
        >>> lut.plot()

        """
        return self.GetNumberOfColors()

    @n_values.setter
    def n_values(self, value: int):
        if self._cmap is not None:
            self.apply_cmap(self._cmap, value)
            self.SetNumberOfTableValues(value)
        elif self._values_manual:
            raise RuntimeError(
                'Number of values cannot be set when the values array has been manually set. Reassign the values array if you wish to change the number of values.'
            )
        else:
            self.SetNumberOfColors(value)
            self.ForceBuild()

    @property
    def annotations(self) -> dict:
        """Return or set annotations.

        Pass a dictionary of annotations. Keys are the float values in the
        scalars range to annotate on the scalar bar and the values are the the
        string annotations.

        Examples
        --------
        Assign annotations to the lookup table.

        >>> import pyvista as pv
        >>> lut = pv.LookupTable('magma')
        >>> lut.annotations = {0: 'low', 0.5: 'medium', 1: 'high'}
        >>> lut.plot()

        """
        vtk_values = self.GetAnnotatedValues()
        if vtk_values is None:
            return {}
        n_items = vtk_values.GetSize()
        keys = [vtk_values.GetValue(ii).ToFloat() for ii in range(n_items)]

        vtk_str = self.GetAnnotations()
        values = [str(vtk_str.GetValue(ii)) for ii in range(n_items)]
        return dict(zip(keys, values))

    @annotations.setter
    def annotations(self, values: Optional[dict]):
        self.ResetAnnotations()
        if values is not None:
            for val, anno in values.items():
                self.SetAnnotation(float(val), str(anno))

    @property
    def _lookup_type(self) -> str:
        """Return the lookup type."""
        if self.cmap:
            if hasattr(self.cmap, 'name'):
                return f'{self.cmap.name}'  # type: ignore
            else:  # pragma: no cover
                return f'{self.cmap}'
        elif self._values_manual:
            return 'From values array'
        else:
            return 'PyVista Lookup Table'

    def plot(self, **kwargs):
        """Plot this lookup table.

        Parameters
        ----------
        **kwargs : dict, optional
            Optional keyword arguments passed to :func:`pyvista.Plotter.show`.

        Examples
        --------
        Plot the ``"viridis"`` colormap with the below and above colors.

        >>> import pyvista as pv
        >>> lut = pv.LookupTable('viridis', n_values=8)
        >>> lut.below_range_color = 'black'
        >>> lut.above_range_color = 'grey'
        >>> lut.nan_color = 'r'
        >>> lut.plot()

        Plot only ``"blues"`` colormap.

        >>> import pyvista as pv
        >>> lut = pv.LookupTable('blues', n_values=1024)
        >>> lut.plot()

        """
        # need a trivial polydata for this
        mesh = pv.PolyData(np.zeros((2, 3)))
        mesh['Lookup Table'] = self.scalar_range

        pl = pv.Plotter(window_size=(800, 230), off_screen=kwargs.pop('off_screen', None))
        actor = pl.add_mesh(mesh, scalars=None, show_scalar_bar=False)
        actor.mapper.lookup_table = self
        actor.visibility = False

        scalar_bar_kwargs = {
            'color': 'k',
            'title': self._lookup_type + '\n',
            'outline': False,
            'title_font_size': 40,
        }
        label_level = 0
        if self.below_range_color:
            scalar_bar_kwargs['below_label'] = 'below'
            label_level = 1
        if self.above_range_color:
            scalar_bar_kwargs['above_label'] = 'above'
            label_level = 1

        label_level += self._nan_color_set

        scalar_bar = pl.add_scalar_bar(**scalar_bar_kwargs)
        scalar_bar.SetLookupTable(self)
        scalar_bar.SetMaximumNumberOfColors(self.n_values)
        scalar_bar.SetPosition(0.03, 0.1 + label_level * 0.1)
        scalar_bar.SetPosition2(0.95, 0.9 - label_level * 0.1)
        # scalar_bar.SetTextPad(-10)
        scalar_bar.SetDrawNanAnnotation(self._nan_color_set)

        pl.background_color = kwargs.pop('background', 'w')
        pl.show(**kwargs)

    def to_color_tf(self) -> _vtk.vtkColorTransferFunction:
        """Return the VTK color transfer function of this table.

        Returns
        -------
        vtk.vtkColorTransferFunction
            VTK color transfer function.

        Examples
        --------
        >>> import pyvista
        >>> lut = pyvista.LookupTable()
        >>> tf = lut.to_color_tf()
        >>> tf  # doctest:+SKIP
        <vtkmodules.vtkRenderingCore.vtkColorTransferFunction(0x339bd40) at 0x7ffabf634700>

        """
        color_tf = _vtk.vtkColorTransferFunction()
        mn, mx = self.scalar_range
        for ii, value in enumerate(np.linspace(mn, mx, self.n_values)):
            color_tf.AddRGBPoint(ii, *self.map_value(value, False))
        return color_tf

    def to_opacity_tf(self):
        """Return the opacity transfer function of this table.

        Returns
        -------
        vtk.vtkPiecewiseFunction
            Piecewise function of the opacity of this color table.

        Examples
        --------
        >>> import pyvista
        >>> lut = pyvista.LookupTable()
        >>> tf = lut.to_opacity_tf()
        >>> tf  # doctest:+SKIP
        <vtkmodules.vtkCommonDataModel.vtkPiecewiseFunction(0x32fa410) at 0x7fe963d6d5e0>

        """
        opacity_tf = _vtk.vtkPiecewiseFunction()
        for ii, value in enumerate(self.values[:, 3]):
            opacity_tf.AddPoint(ii, value / self.n_values)
        return opacity_tf

    def map_value(self, value: float, opacity: bool = True) -> tuple:
        """Map a single value through the lookup table, returning an RBG(A) color.

        Parameters
        ----------
        value : float
            Scalar value to map to an RGB(A) color.

        opacity : bool, default: True
            Map the opacity as well.

        Returns
        -------
        tuple
            Mapped RGB(A) color.

        Examples
        --------
        >>> import pyvista
        >>> lut = pyvista.LookupTable()
        >>> rgba_color = lut.map_value(0.0)
        >>> rgba_color
        (1.0, 0.0, 0.0, 1.0)

        """
        color = [0.0, 0.0, 0.0]
        self.GetColor(value, color)
        if opacity:
            color.append(self.GetOpacity(value))
        return tuple(color)

    def __call__(self, value):
        """Implement a Matplotlib colormap-like call."""
        if isinstance(value, (int, float)):
            return self.map_value(value)
        else:
            try:
                return np.array([self.map_value(item) for item in value])
            except:
                raise TypeError('LookupTable __call__ expects a single value or an iterable.')
