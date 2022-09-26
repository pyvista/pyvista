"""Wrap vtkLookupTable."""
from typing import Optional

import numpy as np

import pyvista as pv
from pyvista import _vtk
from pyvista.plotting.colors import Color, get_cmap_safe
from pyvista.utilities.helpers import convert_array
from pyvista.utilities.misc import has_module, no_new_attr

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
        Color map from ``matplotlib``, ``colorcet``, or ``cmocean``.

    n_values : int, default: 256
        Number of colors in the color map.

    flip : bool, default: False
        Flip the direction of cmap. Most colormaps allow ``*_r`` suffix to do this
        as well.

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

    def __init__(self, cmap=None, n_values=256, flip=False):
        """Initialize the lookup table."""
        if cmap is not None:
            self._apply_cmap(cmap, n_values=n_values, flip=flip)
        else:
            self.n_values = 256

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
        self._rebuild()

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
        self._rebuild()

    @property
    def cmap(self) -> Optional[str]:
        """Return or set the color map used by this lookup table.

        Examples
        --------
        Apply the single matplotlib color map ``"Oranges"``.

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
        self._apply_cmap(value, self.n_values)

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
        self._rebuild()

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
        self._rebuild()

    def _rebuild(self):
        """Clear the color map and recompute the values table."""
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
        self._rebuild()

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
    def above_range_color(self, value):
        if value is None:
            self.SetUseAboveRangeColor(False)
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
    def below_range_color(self, value):
        if value is None:
            self.SetUseBelowRangeColor(False)
        else:
            self.SetBelowRangeColor(*Color(value).float_rgba)
            self.SetUseBelowRangeColor(True)

    def _apply_cmap(self, cmap, n_values=256, flip=False):
        """Assign a colormap to this lookup table.

        Parameters
        ----------
        cmap : str, list, colors.Colormap
            Colormap from matplotlib, colorcet, or cmocean.

        n_values : int, default: 256
            Number of colors in the color map.

        flip : bool, default: False
            Flip direction of cmap. Most colormaps allow ``*_r`` suffix to do
            this as well.

        """
        if not has_module('matplotlib'):  # pragma: no cover
            raise ModuleNotFoundError('Install matplotlib to use color maps.')

        if isinstance(cmap, list):
            n_values = len(cmap)

        cmap = get_cmap_safe(cmap)
        values = cmap(np.linspace(0, 1, n_values)) * 255
        if flip:
            values = values[::-1]
        self.values = values
        self._values_manual = False
        self._cmap = cmap

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
            self._apply_cmap(self._cmap, value)
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

        pl = pv.Plotter(window_size=(800, 230))
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
