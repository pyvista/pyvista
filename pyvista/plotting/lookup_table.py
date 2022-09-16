import numpy as np

import pyvista as pv
from pyvista import _vtk
from pyvista.plotting.colors import Color, get_cmap_safe
from pyvista.utilities.misc import has_module, no_new_attr


@no_new_attr
class LookupTable(_vtk.vtkLookupTable):
    """Wrap vtk.vtkLookupTable

    Parameters
    ----------
    cmap : str, colors.Colormap
        Colormap from matplotlib, colorcet, or cmocean.

    n_values : int, default: 256
        Number of colors in the color map.

    flip : bool, default: False
        Flip direction of cmap. Most colormaps allow ``*_r`` suffix to do this
        as well.

    Examples
    --------
    Plot the lookup table with the default VTK color map.

    >>> import pyvista as pv
    >>> lut = pv.LookupTable()
    >>> lut.plot()

    Plot the lookup table with the ``'inferno'`` color map.

    >>> import pyvista as pv
    >>> lut = pv.LookupTable('inferno', n_values=32)
    >>> lut.plot()

    """

    def __init__(self, cmap=None, n_values=256, flip=False):
        """Initialize the lookup table."""
        if cmap is not None:
            self.apply_cmap(cmap, n_values=n_values, flip=flip)
        else:
            self.n_values = 256

    @property
    def hue_range(self):
        """Return or set the hue range.

        Examples
        --------
        Set the hue range. This allows you to create a lookup table
        without setting a color map.

        >>> import pyvista as pv
        >>> lut = pv.LookupTable()
        >>> lut.hue_range = (0, 0.9)
        (0.0, 0.9)
        >>> lut.plot()

        """

        return self.GetHueRange()

    @hue_range.setter
    def hue_range(self, value: tuple):
        self.SetHueRange(value)

    @property
    def scale(self) -> str:
        """Return or set the scale.

        Scale is either linear or log.

        Examples
        --------
        Use log scale for the lookup table.

        >>> import pyvista as pv
        >>> lut = pv.LookupTable()
        >>> lut.scale = 'log'
        >>> lut.rng = (1, 100)
        >>> lut.plot()

        """
        return 'log' if self.GetScale() else 'linear'

    @scale.setter
    def scale(self, value: str):
        if value not in ['linear', 'log']:
            raise ValueError('`scale` must be either "linear" or "log"')
        self.SetScale(value == 'log')

    def __repr__(self):
        """Return the representation."""
        attr = [
            f'{type(self).__name__} ({hex(id(self))})',
            f'  Values:                     {self.center}',
            f'  Number of Colors            {self.n_values}',
        ]
        return '\n'.join(attr)

    @property
    def rng(self) -> tuple:
        """Return or set the scalar range.

        Examples
        --------
        >>> import pyvista as pv
        >>> lut = pv.LookupTable()
        >>> lut.rng = (0, 10)
        >>> lug.rng
        (0, 10)

        """
        return self.GetRange()

    @rng.setter
    def rng(self, value: tuple):
        self.SetRange(value)

    @property
    def above_range_color(self):
        """Return or set the above range color."""
        return Color(self.GetAboveRangeColor())

    @above_range_color.setter
    def above_range_color(self, value):
        self.SetAboveRangeColor(*Color(value).float_rgba)

    @property
    def use_above_range_color(self) -> bool:
        """Use above range color.

        Examples
        --------
        Demonstrate the usage of the above range color.

        >>> import pyvista as pv
        >>> lut = pyvista.pv()
        >>> lut.use_above_range_color = True
        >>> lut.above_range_color = 'blue'

        """
        return self.GetUseAboveRangeColor()

    @use_above_range_color.setter
    def use_above_range_color(self, value: bool):
        """Use above range color."""
        self.SetUseAboveRangeColor(value)

    @property
    def below_range_color(self):
        """Return or set the below range color."""
        return Color(self.GetBelowRangeColor())

    @below_range_color.setter
    def below_range_color(self, value):
        self.SetBelowRangeColor(*Color(value).float_rgba)

    @property
    def use_below_range_color(self) -> bool:
        """Use below range color.

        Examples
        --------
        Demonstrate the usage of the below range color.

        >>> import pyvista as pv
        >>> lut = pyvista.pv()
        >>> lut.use_below_range_color = True
        >>> lut.below_range_color = 'blue'

        """
        return self.GetUseBelowRangeColor()

    @use_below_range_color.setter
    def use_below_range_color(self, value: bool):
        """Use below range color."""
        self.SetUseBelowRangeColor(value)

    def apply_cmap(self, cmap, n_values=256, flip=False):
        """Assign a colormap to this lookup table.

        Parameters
        ----------
        cmap : str, colors.Colormap
            Colormap from matplotlib, colorcet, or cmocean.

        n_values : int, default: 256
            Number of colors in the color map.

        flip : bool, default: False
            Flip direction of cmap. Most colormaps allow ``*_r`` suffix to do
            this as well.

        Examples
        --------
        >>> import pyvista as pv
        >>> lut = pv.LookupTable()
        >>> lut.apply_cmap('inferno', n_values=32)
        >>> lut.plot()

        """
        if not has_module('matplotlib'):  # pragma: no cover
            raise ModuleNotFoundError('Install matplotlib to use color maps.')

        cmap = get_cmap_safe(cmap)
        values = cmap(np.linspace(0, 1, n_values)) * 255
        if flip:
            values = values[::-1]
        self.values = values

    @property
    def values(self) -> np.ndarray:
        """Return or set the lookup table values.

        This is useful when creating a custom lookup table. The table must be a
        RGBA array shaped ``(n, 4)``.

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
        array([[  0,   0,   0, 255],
               [ 85,   0,   0, 255],
               [170,   0,   0, 255],
               [255,   0,   0, 255]], dtype=uint8)

        """
        return _vtk.vtk_to_numpy(self.GetTable())

    @values.setter
    def values(self, new_values):
        new_values = np.array(new_values, copy=False).astype(np.uint8, copy=False)
        self.SetTable(_vtk.numpy_to_vtk(new_values))

    @property
    def n_values(self) -> int:
        """Return the number of values in the lookup table.

        Examples
        --------
        Plot the ``"blues"`` colormap.

        >>> import pyvista as pv
        >>> lut = pv.LookupTable('reds', n_values=10)
        >>> lut.n_values
        10

        """
        return self.GetNumberOfColors()

    @n_values.setter
    def n_values(self, value: int):
        self.SetNumberOfColors(value)

    def plot(self, **kwargs):
        """Plot this lookup table.

        Parameters
        ----------
        **kwargs : dict, optional
            Optional keyword arguments passed to :func:`pyvista.Plotter.show`.

        Examples
        --------
        Plot the ``"blues"`` colormap with the below and above ranges.

        >>> import pyvista as pv
        >>> lut = pv.LookupTable('blues')
        >>> lut.below_range_color = 'black'
        >>> lut.use_below_range_color = True
        >>> lut.above_range_color = 'red'
        >>> lut.use_above_range_color = True
        >>> lut.plot()

        """
        # need a trivial polydata for this
        mesh = pv.PolyData(np.zeros((2, 3)))
        mesh['Lookup Table'] = self.rng

        # hide the actor with opacity rather than visibility to get the scalar
        # bar to render correctly
        pl = pv.Plotter(window_size=(800, 180))
        actor = pl.add_mesh(mesh, scalars=None, show_scalar_bar=False, opacity=0.0)
        actor.mapper.lookup_table = self
        scalar_bar_kwargs = {'color': 'k', 'title': 'Lookup Table', 'outline': False}
        if self.use_below_range_color:
            scalar_bar_kwargs['below_label'] = 'below'
        if self.use_above_range_color:
            scalar_bar_kwargs['above_label'] = 'above'

        scalar_bar = pl.add_scalar_bar(**scalar_bar_kwargs)
        scalar_bar.SetLookupTable(self)
        scalar_bar.SetMaximumNumberOfColors(self.n_values)
        scalar_bar.SetPosition(0.03, 0.1)
        scalar_bar.SetPosition2(0.95, 0.9)
        scalar_bar.SetTextPad(8)
        pl.background_color = 'w'
        pl.show(**kwargs)
