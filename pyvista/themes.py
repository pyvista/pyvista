"""API description for managing plotting theme parameters in pyvista.

Examples
--------
Apply a built-in theme

>>> import pyvista
>>> pyvista.set_plot_theme('default')
>>> pyvista.set_plot_theme('document')
>>> pyvista.set_plot_theme('dark')
>>> pyvista.set_plot_theme('paraview')

Load a theme into pyvista

>>> theme = pyvista.themes.DefaultTheme()
>>> theme.save('my_theme.json')  # doctest:+SKIP
>>> loaded_theme = pyvista.load_theme('my_theme.json')  # doctest:+SKIP

Create a custom theme from the default theme and load it into
pyvista.

>>> my_theme = pyvista.themes.DefaultTheme()
>>> my_theme.font.size = 20
>>> my_theme.font.title_size = 40
>>> my_theme.cmap = 'jet'
...
>>> pyvista.global_theme.load_theme(my_theme)
>>> pyvista.global_theme.font.size
20

"""

import json
from typing import Union, List
import warnings
from enum import Enum
import os

from .plotting.colors import PARAVIEW_BACKGROUND, get_cmap_safe
from .plotting.tools import parse_color, parse_font_family
from .utilities.misc import PyvistaDeprecationWarning
from .core.errors import DeprecationError


class _rcParams(dict):  # pragma: no cover
    """Reference to the deprecated rcParams dictionary."""

    def __getitem__(self, key):
        import pyvista  # avoids circular import
        warnings.warn('rcParams is deprecated.  Please use ``pyvista.global_theme``.',
                      DeprecationWarning)
        return getattr(pyvista.global_theme, key)

    def __setitem__(self, key, value):
        import pyvista  # avoids circular import
        warnings.warn('rcParams is deprecated.  Please use ``pyvista.global_theme``.',
                      DeprecationWarning)
        setattr(pyvista.global_theme, key, value)

    def __repr__(self):
        """Use the repr of global_theme."""
        import pyvista  # avoids circular import
        warnings.warn('rcParams is deprecated.  Please use ``pyvista.global_theme``',
                      DeprecationWarning)
        return repr(pyvista.global_theme)


def _check_between_zero_and_one(value: float, value_name: str = 'value'):
    """Check if a value is between zero and one."""
    if value < 0 or value > 1:
        raise ValueError('{value_name} must be between 0 and 1.')


def load_theme(filename):
    """Load a theme from a file.

    Examples
    --------
    >>> import pyvista
    >>> theme = pyvista.themes.DefaultTheme()
    >>> theme.save('my_theme.json')  # doctest:+SKIP
    >>> loaded_theme = pyvista.load_theme('my_theme.json')  # doctest:+SKIP

    """
    with open(filename) as f:
        theme_dict = json.load(f)
    return DefaultTheme.from_dict(theme_dict)


def set_plot_theme(theme):
    """Set the plotting parameters to a predefined theme using a string.

    Parameters
    ----------
    theme : str
        Theme name.  Either ``'default'``, ``'document'``, ``'dark'``,
        or ``'paraview'``.

    Examples
    --------
    Set to the default theme.

    >>> import pyvista
    >>> pyvista.set_plot_theme('default')

    Set to the document theme.

    >>> pyvista.set_plot_theme('document')

    Set to the dark theme.

    >>> pyvista.set_plot_theme('dark')

    Set to the ParaView theme.

    >>> pyvista.set_plot_theme('paraview')

    """
    import pyvista
    if isinstance(theme, str):
        theme = theme.lower()
        if theme == 'night':  # pragma: no cover
            warnings.warn('use "dark" instead of "night" theme', PyvistaDeprecationWarning)
            theme = 'dark'
        new_theme_type = _ALLOWED_THEMES[theme].value
        pyvista.global_theme.load_theme(new_theme_type())
    elif isinstance(theme, DefaultTheme):
        pyvista.global_theme.load_theme(theme)
    else:
        raise TypeError(f'Expected a ``pyvista.themes.DefaultTheme`` or ``str``, not '
                        f'a {type(theme).__name__}')


class _ThemeConfig():
    """Provide common methods for theme configuration classes."""

    __slots__: List[str] = []

    @classmethod
    def from_dict(cls, dict_):
        """Create from a dictionary."""
        inst = cls()
        for key, value in dict_.items():
            attr = getattr(inst, key)
            if hasattr(attr, 'from_dict'):
                setattr(inst, key, attr.from_dict(value))
            else:
                setattr(inst, key, value)
        return inst

    def to_dict(self) -> dict:
        """Return theme config parameters as a dictionary."""
        # remove the first underscore in each entry
        dict_ = {}
        for key in self.__slots__:
            value = getattr(self, key)
            key = key[1:]
            if hasattr(value, 'to_dict'):
                dict_[key] = value.to_dict()
            else:
                dict_[key] = value
        return dict_

    def __eq__(self, other):
        if not isinstance(self, type(other)):
            return False

        for attr_name in other.__slots__:
            attr = getattr(self, attr_name)
            other_attr = getattr(other, attr_name)
            if isinstance(attr, (tuple, list)):
                if tuple(attr) != tuple(other_attr):
                    return False
            else:
                if not attr == other_attr:
                    return False

        return True

    def __getitem__(self, key):
        """Get a value via a key.

        Implemented here for backwards compatibility.
        """
        return getattr(self, key)

    def __setitem__(self, key, value):
        """Set a value via a key.

        Implemented here for backwards compatibility.
        """
        setattr(self, key, value)


class _DepthPeelingConfig(_ThemeConfig):
    """PyVista depth peeling configuration.

    Examples
    --------
    Set global depth peeling parameters.

    >>> import pyvista
    >>> pyvista.global_theme.depth_peeling.number_of_peels = 1
    >>> pyvista.global_theme.depth_peeling.occlusion_ratio = 0.0
    >>> pyvista.global_theme.depth_peeling.enabled = True

    """

    __slots__ = ['_number_of_peels',
                 '_occlusion_ratio',
                 '_enabled']

    def __init__(self):
        self._number_of_peels = 4
        self._occlusion_ratio = 0.0
        self._enabled = False

    @property
    def number_of_peels(self) -> int:
        """Return or set the number of peels in depth peeling.

        Examples
        --------
        >>> import pyvista
        >>> pyvista.global_theme.depth_peeling.number_of_peels = 1

        """
        return self._number_of_peels

    @number_of_peels.setter
    def number_of_peels(self, number_of_peels: int):
        self._number_of_peels = int(number_of_peels)

    @property
    def occlusion_ratio(self) -> float:
        """Return or set the occlusion ratio in depth peeling.

        Examples
        --------
        >>> import pyvista
        >>> pyvista.global_theme.depth_peeling.occlusion_ratio = 0.0

        """
        return self._occlusion_ratio

    @occlusion_ratio.setter
    def occlusion_ratio(self, occlusion_ratio: float):
        self._occlusion_ratio = float(occlusion_ratio)

    @property
    def enabled(self) -> bool:
        """Return or set if depth peeling is enabled.

        Examples
        --------
        >>> import pyvista
        >>> pyvista.global_theme.depth_peeling.enabled = True

        """
        return self._enabled

    @enabled.setter
    def enabled(self, enabled: bool):
        self._enabled = bool(enabled)

    def __repr__(self):
        txt = ['']
        parm = {
            'Number': 'number_of_peels',
            'Occlusion ratio': 'occlusion_ratio',
            'Enabled': 'enabled',
        }
        for name, attr in parm.items():
            setting = getattr(self, attr)
            txt.append(f'    {name:<21}: {setting}')
        return '\n'.join(txt)


class _SilhouetteConfig(_ThemeConfig):
    """PyVista silhouette configuration.

    Examples
    --------
    Set global silhouette parameters.

    >>> import pyvista
    >>> pyvista.global_theme.silhouette.color = 'grey'
    >>> pyvista.global_theme.silhouette.line_width = 2
    >>> pyvista.global_theme.silhouette.feature_angle = 20

    """

    __slots__ = ['_color',
                 '_line_width',
                 '_opacity',
                 '_feature_angle',
                 '_decimate']

    def __init__(self):
        self._color = parse_color('black')
        self._line_width = 2
        self._opacity = 1.0
        self._feature_angle = None
        self._decimate = 0.9

    @property
    def color(self) -> tuple:
        """Return or set the silhouette color.

        Examples
        --------
        >>> import pyvista
        >>> pyvista.global_theme.silhouette.color = 'red'

        """
        return self._color

    @color.setter
    def color(self, color: Union[tuple, str]):
        self._color = parse_color(color)

    @property
    def line_width(self) -> float:
        """Return or set the silhouette line width.

        Examples
        --------
        >>> import pyvista
        >>> pyvista.global_theme.silhouette.line_width = 2.0

        """
        return self._line_width

    @line_width.setter
    def line_width(self, line_width: float):
        self._line_width = float(line_width)

    @property
    def opacity(self) -> float:
        """Return or set the silhouette opacity.

        Examples
        --------
        >>> import pyvista
        >>> pyvista.global_theme.silhouette.opacity = 1.0

        """
        return self._opacity

    @opacity.setter
    def opacity(self, opacity: float):
        _check_between_zero_and_one(opacity, 'opacity')
        self._opacity = float(opacity)

    @property
    def feature_angle(self) -> Union[float, None]:
        """Return or set the silhouette feature angle.

        Examples
        --------
        >>> import pyvista
        >>> pyvista.global_theme.silhouette.feature_angle = 20.0

        """
        return self._feature_angle

    @feature_angle.setter
    def feature_angle(self, feature_angle: Union[float, None]):
        self._feature_angle = feature_angle

    @property
    def decimate(self) -> float:
        """Return or set the amount to decimate the silhouette.

        Parameter must be between 0 and 1.

        Examples
        --------
        >>> import pyvista
        >>> pyvista.global_theme.silhouette.decimate = 0.9

        """
        return self._decimate

    @decimate.setter
    def decimate(self, decimate: float):
        _check_between_zero_and_one(decimate, 'decimate')
        self._decimate = float(decimate)

    def __repr__(self):
        txt = ['']
        parm = {
            'Color': 'color',
            'Line width': 'line_width',
            'Opacity': 'opacity',
            'Feature angle': 'feature_angle',
            'Decimate': 'decimate',
        }
        for name, attr in parm.items():
            setting = getattr(self, attr)
            txt.append(f'    {name:<21}: {setting}')
        return '\n'.join(txt)


class _ColorbarConfig(_ThemeConfig):
    """PyVista colorbar configuration.

    Examples
    --------
    Set the colorbar width.

    >>> import pyvista
    >>> pyvista.global_theme.colorbar_horizontal.width = 0.2

    """

    __slots__ = ['_width',
                 '_height',
                 '_position_x',
                 '_position_y']

    def __init__(self):
        self._width = None
        self._height = None
        self._position_x = None
        self._position_y = None

    @property
    def width(self) -> float:
        """Return or set colorbar width.

        Examples
        --------
        >>> import pyvista
        >>> pyvista.global_theme.colorbar_horizontal.width = 0.2

        """
        return self._width

    @width.setter
    def width(self, width: float):
        self._width = float(width)

    @property
    def height(self) -> float:
        """Return or set colorbar height.

        Examples
        --------
        >>> import pyvista
        >>> pyvista.global_theme.colorbar_horizontal.height = 0.2

        """
        return self._height

    @height.setter
    def height(self, height: float):
        self._height = float(height)

    @property
    def position_x(self) -> float:
        """Return or set colorbar x position.

        Examples
        --------
        >>> import pyvista
        >>> pyvista.global_theme.colorbar_horizontal.position_x = 0.2

        """
        return self._position_x

    @position_x.setter
    def position_x(self, position_x: float):
        self._position_x = float(position_x)

    @property
    def position_y(self) -> float:
        """Return or set colorbar y position.

        Examples
        --------
        >>> import pyvista
        >>> pyvista.global_theme.colorbar_horizontal.position_y = 0.2

        """
        return self._position_y

    @position_y.setter
    def position_y(self, position_y: float):
        self._position_y = float(position_y)

    def __repr__(self):
        txt = ['']
        parm = {
            'Width': 'width',
            'Height': 'height',
            'X Position': 'position_x',
            'Y Position': 'position_y',
        }
        for name, attr in parm.items():
            setting = getattr(self, attr)
            txt.append(f'    {name:<21}: {setting}')

        return '\n'.join(txt)


class _AxesConfig(_ThemeConfig):
    """PyVista axes configuration.

    Examples
    --------
    Set the x axis color to black.

    >>> import pyvista
    >>> pyvista.global_theme.axes.x_color = 'black'

    Show axes by default.

    >>> pyvista.global_theme.axes.show = True

    Use the ``vtk.vtkCubeAxesActor``.

    >>> pyvista.global_theme.axes.box = True

    """

    __slots__ = ['_x_color',
                 '_y_color',
                 '_z_color',
                 '_box',
                 '_show']

    def __init__(self):
        self._x_color = parse_color('tomato')
        self._y_color = parse_color('seagreen')
        self._z_color = parse_color('mediumblue')
        self._box = False
        self._show = True

    def __repr__(self):
        txt = ['Axes configuration']
        parm = {
            'X Color': 'x_color',
            'Y Color': 'y_color',
            'Z Color': 'z_color',
            'Use Box': 'box',
            'Show': 'show',
        }
        for name, attr in parm.items():
            setting = getattr(self, attr)
            txt.append(f'    {name:<21}: {setting}')

        return '\n'.join(txt)

    @property
    def x_color(self) -> tuple:
        """Return or set x axis color.

        Examples
        --------
        >>> import pyvista
        >>> pyvista.global_theme.axes.x_color = 'red'
        """
        return self._x_color

    @x_color.setter
    def x_color(self, color: Union[tuple, str]):
        self._x_color = parse_color(color)

    @property
    def y_color(self) -> tuple:
        """Return or set y axis color.

        Examples
        --------
        >>> import pyvista
        >>> pyvista.global_theme.axes.y_color = 'red'
        """
        return self._y_color

    @y_color.setter
    def y_color(self, color: Union[tuple, str]):
        self._y_color = parse_color(color)

    @property
    def z_color(self) -> tuple:
        """Return or set z axis color.

        Examples
        --------
        >>> import pyvista
        >>> pyvista.global_theme.axes.z_color = 'red'
        """
        return self._z_color

    @z_color.setter
    def z_color(self, color: Union[tuple, str]):
        self._z_color = parse_color(color)

    @property
    def box(self) -> bool:
        """Use the ``vtk.vtkCubeAxesActor`` instead of the default ``vtk.vtkAxesActor``.

        Examples
        --------
        >>> import pyvista
        >>> pyvista.global_theme.axes.box = True

        """
        return self._box

    @box.setter
    def box(self, box: bool):
        self._box = bool(box)

    @property
    def show(self) -> bool:
        """Show or hide the axes actor.

        Examples
        --------
        Hide the axes by default.

        >>> import pyvista
        >>> pyvista.global_theme.axes.show = False

        """
        return self._show

    @show.setter
    def show(self, show: bool):
        self._show = bool(show)


class _Font(_ThemeConfig):
    """PyVista plotter font configuration.

    Examples
    --------
    Set the default font family to 'arial'.  Must be either
    'arial', 'courier', or 'times'.

    >>> import pyvista
    >>> pyvista.global_theme.font.family = 'arial'

    Set the default font size to 20.

    >>> pyvista.global_theme.font.size = 20

    Set the default title size to 40

    >>> pyvista.global_theme.font.title_size = 40

    Set the default label size to 10

    >>> pyvista.global_theme.font.label_size = 10

    Set the default text color to 'grey'

    >>> pyvista.global_theme.font.color = 'grey'

    Set the string formatter used to format numerical data to '%.6e'

    >>> pyvista.global_theme.font.fmt = '%.6e'

    """

    __slots__ = ['_family',
                 '_size',
                 '_title_size',
                 '_label_size',
                 '_color',
                 '_fmt']

    def __init__(self):
        self._family = 'arial'
        self._size = 12
        self._title_size = None
        self._label_size = None
        self._color = [1, 1, 1]
        self._fmt = None

    def __repr__(self):
        txt = ['']
        parm = {
            'Family': 'family',
            'Size': 'size',
            'Title size': 'title_size',
            'Label size': 'label_size',
            'Color': 'color',
            'Float format': 'fmt',
        }
        for name, attr in parm.items():
            setting = getattr(self, attr)
            txt.append(f'    {name:<21}: {setting}')

        return '\n'.join(txt)

    @property
    def family(self) -> str:
        """Return or set the font family.

        Must be one of the following:

        * ``"arial"``
        * ``"courier"``
        * ``"times"``

        Examples
        --------
        Set the default global font family to 'courier'.

        >>> import pyvista
        >>> pyvista.global_theme.font.family = 'courier'

        """
        return self._family

    @family.setter
    def family(self, family: str):
        parse_font_family(family)  # check valid font
        self._family = family

    @property
    def size(self) -> int:
        """Return or set the font size.

        Examples
        --------
        >>> import pyvista
        >>> pyvista.global_theme.font.size = 20

        """
        return self._size

    @size.setter
    def size(self, size: int):
        self._size = int(size)

    @property
    def title_size(self) -> int:
        """Return or set the title size.

        If ``None``, then VTK uses ``UnconstrainedFontSizeOn`` for titles.

        Examples
        --------
        >>> import pyvista
        >>> pyvista.global_theme.font.title_size = 20
        """
        return self._title_size

    @title_size.setter
    def title_size(self, title_size: int):
        if title_size is None:
            self._title_size = None
        else:
            self._title_size = int(title_size)

    @property
    def label_size(self) -> int:
        """Return or set the label size.

        If ``None``, then VTK uses ``UnconstrainedFontSizeOn`` for labels.

        Examples
        --------
        >>> import pyvista
        >>> pyvista.global_theme.font.label_size = 20
        """
        return self._label_size

    @label_size.setter
    def label_size(self, label_size: int):
        if label_size is None:
            self._label_size = None
        else:
            self._label_size = int(label_size)

    @property
    def color(self) -> tuple:
        """Return or set the font color.

        Examples
        --------
        >>> import pyvista
        >>> pyvista.global_theme.font.color = 'black'
        """
        return self._color

    @color.setter
    def color(self, color: Union[tuple, str]):
        self._color = parse_color(color)

    @property
    def fmt(self) -> str:
        """Return or set the string formatter used to format numerical data.

        Examples
        --------
        Set the string formatter used to format numerical data to '%.6e'.

        >>> import pyvista
        >>> pyvista.global_theme.font.fmt = '%.6e'

        """
        return self._fmt

    @fmt.setter
    def fmt(self, fmt: str):
        self._fmt = fmt


class _SliderStyleConfig(_ThemeConfig):
    """PyVista configuration for a single slider style."""

    __slots__ = ['_name',
                 '_slider_length',
                 '_slider_width',
                 '_slider_color',
                 '_tube_width',
                 '_tube_color',
                 '_cap_opacity',
                 '_cap_length',
                 '_cap_width']

    def __init__(self):
        """Initialize the slider style configuration."""
        self._name = None
        self._slider_length = None
        self._slider_width = None
        self._slider_color = None
        self._tube_width = None
        self._tube_color = None
        self._cap_opacity = None
        self._cap_length = None
        self._cap_width = None

    @property
    def name(self) -> str:
        """Return the name of the slider style configuration."""
        return self._name

    @name.setter
    def name(self, name: str):
        self._name = name

    @property
    def cap_width(self) -> float:
        """Return or set the cap width.

        Examples
        --------
        >>> import pyvista
        >>> pyvista.global_theme.slider_styles.modern.cap_width = 0.02

        """
        return self._cap_width

    @cap_width.setter
    def cap_width(self, cap_width: float):
        self._cap_width = float(cap_width)

    @property
    def cap_length(self) -> float:
        """Return or set the cap length.

        Examples
        --------
        >>> import pyvista
        >>> pyvista.global_theme.slider_styles.modern.cap_length = 0.01

        """
        return self._cap_length

    @cap_length.setter
    def cap_length(self, cap_length: float):
        self._cap_length = float(cap_length)

    @property
    def cap_opacity(self) -> float:
        """Return or set the cap opacity.

        Examples
        --------
        >>> import pyvista
        >>> pyvista.global_theme.slider_styles.modern.cap_opacity = 1.0

        """
        return self._cap_opacity

    @cap_opacity.setter
    def cap_opacity(self, cap_opacity: float):
        self._cap_opacity = float(cap_opacity)

    @property
    def tube_color(self) -> tuple:
        """Return or set the tube color.

        Examples
        --------
        >>> import pyvista
        >>> pyvista.global_theme.slider_styles.modern.tube_color = 'black'
        """
        return self._tube_color

    @tube_color.setter
    def tube_color(self, tube_color: Union[tuple, str]):
        self._tube_color = parse_color(tube_color)

    @property
    def tube_width(self) -> float:
        """Return or set the tube_width.

        Examples
        --------
        >>> import pyvista
        >>> pyvista.global_theme.slider_styles.modern.tube_width = 0.005

        """
        return self._tube_width

    @tube_width.setter
    def tube_width(self, tube_width: float):
        self._tube_width = float(tube_width)

    @property
    def slider_color(self) -> tuple:
        """Return or set the slider color.

        Examples
        --------
        >>> import pyvista
        >>> pyvista.global_theme.slider_styles.modern.slider_color = 'grey'

        """
        return self._slider_color

    @slider_color.setter
    def slider_color(self, slider_color: Union[tuple, str]):
        self._slider_color = parse_color(slider_color)

    @property
    def slider_width(self) -> float:
        """Return or set the slider width.

        Examples
        --------
        >>> import pyvista
        >>> pyvista.global_theme.slider_styles.modern.slider_width = 0.04

        """
        return self._slider_width

    @slider_width.setter
    def slider_width(self, slider_width: float):
        self._slider_width = float(slider_width)

    @property
    def slider_length(self) -> float:
        """Return or set the slider_length.

        Examples
        --------
        >>> import pyvista
        >>> pyvista.global_theme.slider_styles.modern.slider_length = 0.02

        """
        return self._slider_length

    @slider_length.setter
    def slider_length(self, slider_length: float):
        self._slider_length = float(slider_length)

    def __repr__(self):
        txt = ['']
        parm = {
            'Slider length': 'slider_length',
            'Slider width': 'slider_width',
            'Slider color': 'slider_color',
            'Tube width': 'tube_width',
            'Tube color': 'tube_color',
            'Cap opacity': 'cap_opacity',
            'Cap length': 'cap_length',
            'Cap width': 'cap_width',
        }
        for name, attr in parm.items():
            setting = getattr(self, attr)
            txt.append(f'        {name:<17}: {setting}')
        return '\n'.join(txt)


class _SliderConfig(_ThemeConfig):
    """PyVista configuration encompassing all slider styles.

    Examples
    --------
    Set the classic slider configuration.

    >>> import pyvista
    >>> pyvista.global_theme.slider_styles.classic.slider_length = 0.02
    >>> pyvista.global_theme.slider_styles.classic.slider_width = 0.04
    >>> pyvista.global_theme.slider_styles.classic.slider_color = (0.5, 0.5, 0.5)
    >>> pyvista.global_theme.slider_styles.classic.tube_width = 0.005
    >>> pyvista.global_theme.slider_styles.classic.tube_color = (1, 1, 1)
    >>> pyvista.global_theme.slider_styles.classic.cap_opacity = 1
    >>> pyvista.global_theme.slider_styles.classic.cap_length = 0.01
    >>> pyvista.global_theme.slider_styles.classic.cap_width = 0.02

    Set the modern slider configuration.

    >>> pyvista.global_theme.slider_styles.modern.slider_length = 0.02
    >>> pyvista.global_theme.slider_styles.modern.slider_width = 0.04
    >>> pyvista.global_theme.slider_styles.modern.slider_color = (0.43, 0.44, 0.45)
    >>> pyvista.global_theme.slider_styles.modern.tube_width = 0.04
    >>> pyvista.global_theme.slider_styles.modern.tube_color = (0.69, 0.70, 0.709)
    >>> pyvista.global_theme.slider_styles.modern.cap_opacity = 0
    >>> pyvista.global_theme.slider_styles.modern.cap_length = 0.01
    >>> pyvista.global_theme.slider_styles.modern.cap_width = 0.02

    """

    __slots__ = ['_classic',
                 '_modern']

    def __init__(self):
        """Initialize the slider configuration."""
        self._classic = _SliderStyleConfig()
        self._classic.name = 'classic'
        self._classic.slider_length = 0.02
        self._classic.slider_width = 0.04
        self._classic.slider_color = (0.5, 0.5, 0.5)
        self._classic.tube_width = 0.005
        self._classic.tube_color = (1, 1, 1)
        self._classic.cap_opacity = 1
        self._classic.cap_length = 0.01
        self._classic.cap_width = 0.02

        self._modern = _SliderStyleConfig()
        self._modern.name = 'modern'
        self._modern.slider_length = 0.02
        self._modern.slider_width = 0.04
        self._modern.slider_color = (0.43137255, 0.44313725, 0.45882353)
        self._modern.tube_width = 0.04
        self._modern.tube_color = (0.69803922, 0.70196078, 0.70980392)
        self._modern.cap_opacity = 0
        self._modern.cap_length = 0.01
        self._modern.cap_width = 0.02

    @property
    def classic(self) -> _SliderStyleConfig:
        """Return the Classic slider configuration."""
        return self._classic

    @classic.setter
    def classic(self, config: _SliderStyleConfig):
        if not isinstance(config, _SliderStyleConfig):
            raise TypeError('Configuration type must be `_SliderStyleConfig`')
        self._classic = config

    @property
    def modern(self) -> _SliderStyleConfig:
        """Return the Modern slider configuration."""
        return self._modern

    @modern.setter
    def modern(self, config: _SliderStyleConfig):
        if not isinstance(config, _SliderStyleConfig):
            raise TypeError('Configuration type must be `_SliderStyleConfig`')
        self._modern = config

    def __repr__(self):
        txt = ['']
        parm = {
            'Classic': 'classic',
            'Modern': 'modern',
        }
        for name, attr in parm.items():
            setting = getattr(self, attr)
            txt.append(f'    {name:<21}: {setting}')
        return '\n'.join(txt)

    def __iter__(self):
        for style in [self._classic, self._modern]:
            yield style.name


class DefaultTheme(_ThemeConfig):
    """PyVista default theme.

    Examples
    --------
    Change the global default background color to white.

    >>> import pyvista
    >>> pyvista.global_theme.color = 'white'

    Show edges by default.

    >>> pyvista.global_theme.show_edges = True

    Create a new theme from the default theme and apply it globally.

    >>> my_theme = pyvista.themes.DefaultTheme()
    >>> my_theme.color = 'red'
    >>> my_theme.background = 'white'
    >>> pyvista.global_theme.load_theme(my_theme)

    """

    __slots__ = ['_name',
                 '_background',
                 '_jupyter_backend',
                 '_full_screen',
                 '_window_size',
                 '_camera',
                 '_notebook',
                 '_font',
                 '_auto_close',
                 '_cmap',
                 '_color',
                 '_nan_color',
                 '_edge_color',
                 '_outline_color',
                 '_floor_color',
                 '_colorbar_orientation',
                 '_colorbar_horizontal',
                 '_colorbar_vertical',
                 '_show_scalar_bar',
                 '_show_edges',
                 '_lighting',
                 '_interactive',
                 '_render_points_as_spheres',
                 '_transparent_background',
                 '_title',
                 '_axes',
                 '_multi_samples',
                 '_multi_rendering_splitting_position',
                 '_volume_mapper',
                 '_smooth_shading',
                 '_depth_peeling',
                 '_silhouette',
                 '_slider_styles']

    def __init__(self):
        """Initialize the theme."""
        self._name = 'default'
        self._background = parse_color([0.3, 0.3, 0.3])
        self._full_screen = False
        self._camera = {
            'position': [1, 1, 1],
            'viewup': [0, 0, 1],
        }

        self._notebook = None
        self._window_size = [1024, 768]
        self._font = _Font()
        self._cmap = 'viridis'
        self._color = parse_color('white')
        self._nan_color = parse_color('darkgray')
        self._edge_color = parse_color('black')
        self._outline_color = parse_color('white')
        self._floor_color = parse_color('gray')
        self._colorbar_orientation = 'horizontal'

        self._colorbar_horizontal = _ColorbarConfig()
        self._colorbar_horizontal.width = 0.6
        self._colorbar_horizontal.height = 0.08
        self._colorbar_horizontal.position_x = 0.35
        self._colorbar_horizontal.position_y = 0.05

        self._colorbar_vertical = _ColorbarConfig()
        self._colorbar_vertical.width = 0.08
        self._colorbar_vertical.height = 0.45
        self._colorbar_vertical.position_x = 0.9
        self._colorbar_vertical.position_y = 0.02

        self._show_scalar_bar = True
        self._show_edges = False
        self._lighting = True
        self._interactive = False
        self._render_points_as_spheres = False
        self._transparent_background = False
        self._title = 'PyVista'
        self._axes = _AxesConfig()

        # Grab system flag for anti-aliasing
        try:
            self._multi_samples = int(os.environ.get('PYVISTA_MULTI_SAMPLES', 4))
        except ValueError:  # pragma: no cover
            self._multi_samples = 4

        # Grab system flag for auto-closing
        self._auto_close = os.environ.get('PYVISTA_AUTO_CLOSE', '').lower() != 'false'

        self._jupyter_backend = os.environ.get('PYVISTA_JUPYTER_BACKEND', 'ipyvtklink')

        self._multi_rendering_splitting_position = None
        self._volume_mapper = 'fixed_point' if os.name == 'nt' else 'smart'
        self._smooth_shading = False
        self._depth_peeling = _DepthPeelingConfig()
        self._silhouette = _SilhouetteConfig()
        self._slider_styles = _SliderConfig()

    @property
    def background(self):
        """Return or set the default background color of pyvista plots.

        Examples
        --------
        Set the default global background of all plots to white.

        >>> import pyvista
        >>> pyvista.global_theme.background = 'white'
        """
        return self._background

    @background.setter
    def background(self, new_background):
        self._background = parse_color(new_background)

    @property
    def jupyter_backend(self) -> str:
        """Return or set the jupyter notebook plotting backend.

        Jupyter backend to use when plotting.  Must be one of the
        following:

        * ``'ipyvtklink'`` : Render remotely and stream the
          resulting VTK images back to the client.  Supports all VTK
          methods, but suffers from lag due to remote rendering.
          Requires that a virtual framebuffer be setup when displaying
          on a headless server.  Must have ``ipyvtklink`` installed.

        * ``'panel'`` : Convert the VTK render window to a vtkjs
          object and then visualize that within jupyterlab. Supports
          most VTK objects.  Requires that a virtual framebuffer be
          setup when displaying on a headless server.  Must have
          ``panel`` installed.

        * ``'ipygany'`` : Convert all the meshes into ``ipygany``
          meshes and streams those to be rendered on the client side.
          Supports VTK meshes, but few others.  Aside from ``none``,
          this is the only method that does not require a virtual
          framebuffer.  Must have ``ipygany`` installed.

        * ``'static'`` : Display a single static image within the
          JupyterLab environment.  Still requires that a virtual
          framebuffer be setup when displaying on a headless server,
          but does not require any additional modules to be installed.

        * ``'none'`` : Do not display any plots within jupyterlab,
          instead display using dedicated VTK render windows.  This
          will generate nothing on headless servers even with a
          virtual framebuffer.

        Examples
        --------
        Enable the ipygany backend.

        >>> import pyvista as pv
        >>> pv.set_jupyter_backend('ipygany')

        Enable the panel backend.

        >>> pv.set_jupyter_backend('panel')

        Enable the ipyvtklink backend.

        >>> pv.set_jupyter_backend('ipyvtklink')

        Just show static images.

        >>> pv.set_jupyter_backend('static')

        Disable all plotting within JupyterLab and display using a
        standard desktop VTK render window.

        >>> pv.set_jupyter_backend(None)  # or 'none'

        """
        return self._jupyter_backend

    @jupyter_backend.setter
    def jupyter_backend(self, backend: 'str'):
        from pyvista.jupyter import _validate_jupyter_backend
        self._jupyter_backend = _validate_jupyter_backend(backend)

    @property
    def auto_close(self) -> bool:
        """Automatically close the figures when finished plotting.

        .. DANGER::
           Set to ``False`` with extreme caution.

        Examples
        --------
        >>> import pyvista
        >>> pyvista.global_theme.auto_close = False

        """
        return self._auto_close

    @auto_close.setter
    def auto_close(self, value: bool):
        self._auto_close = value

    @property
    def full_screen(self) -> bool:
        """Return if figures are shown in full screen.

        Examples
        --------
        Set windows to be full screen by default.

        >>> import pyvista
        >>> pyvista.global_theme.full_screen = True
        """
        return self._full_screen

    @full_screen.setter
    def full_screen(self, value: bool):
        self._full_screen = value

    @property
    def camera(self):
        """Return or set the default camera position.

        Examples
        --------
        Set both the position and view of the camera.

        >>> import pyvista
        >>> pyvista.global_theme.camera = {'position': [1, 1, 1],
        ...                                'viewup': [0, 0, 1]}

        Set the default position of the camera.

        >>> pyvista.global_theme.camera['position'] = [1, 1, 1]

        Set the default view of the camera.

        >>> pyvista.global_theme.camera['viewup'] = [0, 0, 1]

        """
        return self._camera

    @camera.setter
    def camera(self, camera):
        if not isinstance(camera, dict):
            raise TypeError('Expected ``camera`` to be a dict, not '
                            f'{type(camera).__name__}.')

        if 'position' not in camera:
            raise KeyError('Expected the "position" key in the camera dict.')
        if 'viewup' not in camera:
            raise KeyError('Expected the "viewup" key in the camera dict.')

        self._camera = camera

    @property
    def notebook(self) -> Union[bool, None]:
        """Return or set the state of notebook plotting.

        Setting this to ``True`` always enables notebook plotting,
        while setting it to ``False`` disables plotting even when
        plotting within a jupyter notebook and plots externally.

        Examples
        --------
        Disable all jupyter notebook plotting.

        >>> import pyvista
        >>> pyvista.global_theme.notebook = False

        """
        return self._notebook

    @notebook.setter
    def notebook(self, value: Union[bool, None]):
        self._notebook = value

    @property
    def window_size(self) -> List[int]:
        """Return or set the default render window size.

        Examples
        --------
        Set window size to ``[400, 400]``.

        >>> import pyvista
        >>> pyvista.global_theme.window_size = [400, 400]

        """
        return self._window_size

    @window_size.setter
    def window_size(self, window_size: List[int]):
        if len(window_size) != 2:
            raise ValueError('Expected a length 2 iterable for ``window_size``.')

        # ensure positive size
        if window_size[0] < 0 or window_size[1] < 0:
            raise ValueError('Window size must be a positive value.')

        self._window_size = window_size

    @property
    def font(self) -> _Font:
        """Return or set the default font size, family, and/or color.

        Examples
        --------
        Set the default font family to 'arial'.  Must be either
        'arial', 'courier', or 'times'.

        >>> import pyvista
        >>> pyvista.global_theme.font.family = 'arial'

        Set the default font size to 20.

        >>> pyvista.global_theme.font.size = 20

        Set the default title size to 40.

        >>> pyvista.global_theme.font.title_size = 40

        Set the default label size to 10.

        >>> pyvista.global_theme.font.label_size = 10

        Set the default text color to 'grey'.

        >>> pyvista.global_theme.font.color = 'grey'

        String formatter used to format numerical data to '%.6e'.

        >>> pyvista.global_theme.font.fmt = '%.6e'

        """
        return self._font

    @font.setter
    def font(self, config: _Font):
        if not isinstance(config, _Font):
            raise TypeError('Configuration type must be `_Font`.')
        self._font = config

    @property
    def cmap(self):
        """Return or set the default colormap of pyvista.

        See available Matplotlib colormaps.  Only applicable for when
        displaying ``scalars``. Requires Matplotlib to be installed.
        If ``colorcet`` or ``cmocean`` are installed, their colormaps
        can be specified by name.

        You can also specify a list of colors to override an existing
        colormap with a custom one.  For example, to create a three
        color colormap you might specify ``['green', 'red', 'blue']``

        Examples
        --------
        Set the default global colormap to 'jet'.

        >>> import pyvista
        >>> pyvista.global_theme.cmap = 'jet'

        """
        return self._cmap

    @cmap.setter
    def cmap(self, cmap):
        get_cmap_safe(cmap)  # for validation
        self._cmap = cmap

    @property
    def color(self) -> tuple:
        """Return or set the default color of meshes in pyvista.

        Used for meshes without ``scalars``.

        When setting, the value must be either a string, rgb list,
        or hex color string.  For example:

        * ``color='white'``
        * ``color='w'``
        * ``color=[1, 1, 1]``
        * ``color='#FFFFFF'``

        Examples
        --------
        Set the default mesh color to 'red'.

        >>> import pyvista
        >>> pyvista.global_theme.color = 'red'

        """
        return self._color

    @color.setter
    def color(self, color: Union[tuple, str]):
        self._color = parse_color(color)

    @property
    def nan_color(self) -> tuple:
        """Return or set the default NaN color.

        This color is used to plot all NaN values.

        Examples
        --------
        >>> import pyvista
        >>> pyvista.global_theme.nan_color = 'darkgray'

        """
        return self._nan_color

    @nan_color.setter
    def nan_color(self, nan_color: Union[tuple, str]):
        self._nan_color = parse_color(nan_color)

    @property
    def edge_color(self) -> tuple:
        """Return or set the default edge color.

        Examples
        --------
        Set the global edge color to 'blue'.

        >>> import pyvista
        >>> pyvista.global_theme.edge_color = 'blue'

        """
        return self._edge_color

    @edge_color.setter
    def edge_color(self, edge_color: Union[tuple, str]):
        self._edge_color = parse_color(edge_color)

    @property
    def outline_color(self) -> tuple:
        """Return or set the default outline color.

        Examples
        --------
        >>> import pyvista
        >>> pyvista.global_theme.outline_color = 'white'

        """
        return self._outline_color

    @outline_color.setter
    def outline_color(self, outline_color: Union[tuple, str]):
        self._outline_color = parse_color(outline_color)

    @property
    def floor_color(self) -> tuple:
        """Return or set the default floor color.

        Examples
        --------
        >>> import pyvista
        >>> pyvista.global_theme.floor_color = 'black'

        """
        return self._floor_color

    @floor_color.setter
    def floor_color(self, floor_color: Union[tuple, str]):
        self._floor_color = parse_color(floor_color)

    @property
    def colorbar_orientation(self) -> str:
        """Return or set the default colorbar orientation.

        Must be either ``'vertical'`` or ``'horizontal'``.

        Examples
        --------
        >>> import pyvista
        >>> pyvista.global_theme.colorbar_orientation = 'horizontal'

        """
        return self._colorbar_orientation

    @colorbar_orientation.setter
    def colorbar_orientation(self, colorbar_orientation: str):
        if colorbar_orientation not in ['vertical', 'horizontal']:
            raise ValueError('Colorbar orientation must be either "vertical" or '
                             '"horizontal"')
        self._colorbar_orientation = colorbar_orientation

    @property
    def colorbar_horizontal(self) -> _ColorbarConfig:
        """Return or set the default parameters of a horizontal colorbar.

        Examples
        --------
        Set the default horizontal colorbar width to 0.6.

        >>> import pyvista
        >>> pyvista.global_theme.colorbar_horizontal.width = 0.6

        Set the default horizontal colorbar height to 0.2.

        >>> pyvista.global_theme.colorbar_horizontal.height = 0.2

        """
        return self._colorbar_horizontal

    @colorbar_horizontal.setter
    def colorbar_horizontal(self, config: _ColorbarConfig):
        if not isinstance(config, _ColorbarConfig):
            raise TypeError('Configuration type must be `_ColorbarConfig`.')
        self._colorbar_horizontal = config

    @property
    def colorbar_vertical(self) -> _ColorbarConfig:
        """Return or set the default parameters of a vertical colorbar.

        Examples
        --------
        Set the default colorbar width to 0.45.

        >>> import pyvista
        >>> pyvista.global_theme.colorbar_vertical.width = 0.45

        Set the default colorbar height to 0.8.

        >>> import pyvista
        >>> pyvista.global_theme.colorbar_vertical.height = 0.8

        """
        return self._colorbar_vertical

    @colorbar_vertical.setter
    def colorbar_vertical(self, config: _ColorbarConfig):
        if not isinstance(config, _ColorbarConfig):
            raise TypeError('Configuration type must be `_ColorbarConfig`.')
        self._colorbar_vertical = config

    @property
    def show_scalar_bar(self) -> bool:
        """Return or set the default color bar visibility.

        Examples
        --------
        Show the scalar bar by default when scalars are available.

        >>> import pyvista
        >>> pyvista.global_theme.show_scalar_bar = True

        """
        return self._show_scalar_bar

    @show_scalar_bar.setter
    def show_scalar_bar(self, show_scalar_bar: bool):
        self._show_scalar_bar = bool(show_scalar_bar)

    @property
    def show_edges(self) -> bool:
        """Return or set the default edge visibility.

        Examples
        --------
        Show edges globally by default.

        >>> import pyvista
        >>> pyvista.global_theme.show_edges = True

        """
        return self._show_edges

    @show_edges.setter
    def show_edges(self, show_edges: bool):
        self._show_edges = bool(show_edges)

    @property
    def lighting(self) -> bool:
        """Return or set the default ``lighting``.

        Examples
        --------
        Disable lighting globally.

        >>> import pyvista
        >>> pyvista.global_theme.lighting = False
        """
        return self._lighting

    @lighting.setter
    def lighting(self, lighting: bool):
        self._lighting = lighting

    @property
    def interactive(self) -> bool:
        """Return or set the default ``interactive`` parameter.

        Examples
        --------
        Make all plots non-interactive globally.

        >>> import pyvista
        >>> pyvista.global_theme.interactive = False
        """
        return self._interactive

    @interactive.setter
    def interactive(self, interactive: bool):
        self._interactive = bool(interactive)

    @property
    def render_points_as_spheres(self) -> bool:
        """Return or set the default ``render_points_as_spheres`` parameter.

        Examples
        --------
        Render points as spheres by default globally.

        >>> import pyvista
        >>> pyvista.global_theme.render_points_as_spheres = True
        """
        return self._render_points_as_spheres

    @render_points_as_spheres.setter
    def render_points_as_spheres(self, render_points_as_spheres: bool):
        self._render_points_as_spheres = bool(render_points_as_spheres)

    @property
    def transparent_background(self) -> bool:
        """Return or set the default ``transparent_background`` parameter.

        Examples
        --------
        Set transparent_background globally to ``True``.

        >>> import pyvista
        >>> pyvista.global_theme.transparent_background = True

        """
        return self._transparent_background

    @transparent_background.setter
    def transparent_background(self, transparent_background: bool):
        self._transparent_background = transparent_background

    @property
    def title(self) -> str:
        """Return or set the default ``title`` parameter.

        This is the VTK render window title.

        Examples
        --------
        Set title globally to 'plot'.

        >>> import pyvista
        >>> pyvista.global_theme.title = 'plot'

        """
        return self._title

    @title.setter
    def title(self, title: str):
        self._title = title

    @property
    def multi_samples(self) -> int:
        """Return or set the default ``multi_samples`` parameter.

        Set the number of multisamples to enable hardware antialiasing.

        Examples
        --------
        Set the default number of multisamples to 2.

        >>> import pyvista
        >>> pyvista.global_theme.multi_samples = 2

        """
        return self._multi_samples

    @multi_samples.setter
    def multi_samples(self, multi_samples: int):
        self._multi_samples = int(multi_samples)

    @property
    def multi_rendering_splitting_position(self) -> float:
        """Return or set the default ``multi_rendering_splitting_position`` parameter.

        Examples
        --------
        Set multi_rendering_splitting_position globally to 0.5 (the
        middle of the window).

        >>> import pyvista
        >>> pyvista.global_theme.multi_rendering_splitting_position = 0.5

        """
        return self._multi_rendering_splitting_position

    @multi_rendering_splitting_position.setter
    def multi_rendering_splitting_position(self, multi_rendering_splitting_position: float):
        self._multi_rendering_splitting_position = multi_rendering_splitting_position

    @property
    def volume_mapper(self) -> str:
        """Return or set the default ``volume_mapper`` parameter.

        Must be one of the following strings, which are mapped to the
        following VTK volume mappers.

        * ``'fixed_point'`` : ``vtk.vtkFixedPointVolumeRayCastMapper``
        * ``'gpu'`` : ``vtk.vtkGPUVolumeRayCastMapper``
        * ``'open_gl'`` : ``vtk.vtkOpenGLGPUVolumeRayCastMapper``
        * ``'smart'`` : ``vtk.vtkSmartVolumeMapper``

        Examples
        --------
        Set default volume mapper globally to 'gpu'.

        >>> import pyvista
        >>> pyvista.global_theme.volume_mapper = 'gpu'

        """
        return self._volume_mapper

    @volume_mapper.setter
    def volume_mapper(self, mapper: str):
        mappers = ['fixed_point', 'gpu', 'open_gl', 'smart']
        if mapper not in mappers:
            raise ValueError(f"Mapper ({mapper}) unknown. Available volume mappers "
                             f"include:\n {', '.join(mappers)}")

        self._volume_mapper = mapper

    @property
    def smooth_shading(self) -> bool:
        """Return or set the default ``smooth_shading`` parameter.

        Examples
        --------
        Set the global smooth_shading parameter default to ``True``.

        >>> import pyvista
        >>> pyvista.global_theme.smooth_shading = True

        """
        return self._smooth_shading

    @smooth_shading.setter
    def smooth_shading(self, smooth_shading: bool):
        self._smooth_shading = bool(smooth_shading)

    @property
    def depth_peeling(self) -> _DepthPeelingConfig:
        """Return or set the default depth peeling parameters.

        Examples
        --------
        Set the global depth_peeling parameter default to be enabled
        with 8 peels.

        >>> import pyvista
        >>> pyvista.global_theme.depth_peeling.number_of_peels = 8
        >>> pyvista.global_theme.depth_peeling.occlusion_ratio = 0.0
        >>> pyvista.global_theme.depth_peeling.enabled = True

        """
        return self._depth_peeling

    @depth_peeling.setter
    def depth_peeling(self, config: _DepthPeelingConfig):
        if not isinstance(config, _DepthPeelingConfig):
            raise TypeError('Configuration type must be `_DepthPeelingConfig`.')
        self._depth_peeling = config

    @property
    def silhouette(self) -> _SilhouetteConfig:
        """Return or set the default ``silhouette`` configuration.

        Examples
        --------
        Set parameters of the silhouette.

        >>> import pyvista
        >>> pyvista.global_theme.silhouette.color = 'grey'
        >>> pyvista.global_theme.silhouette.line_width = 2.0
        >>> pyvista.global_theme.silhouette.feature_angle = 20

        """
        return self._silhouette

    @silhouette.setter
    def silhouette(self, config: _SilhouetteConfig):
        if not isinstance(config, _SilhouetteConfig):
            raise TypeError('Configuration type must be `_SilhouetteConfig`')
        self._silhouette = config

    @property
    def slider_styles(self) -> _SliderConfig:
        """Return the default slider style configurations."""
        return self._slider_styles

    @slider_styles.setter
    def slider_styles(self, config: _SliderConfig):
        if not isinstance(config, _SliderConfig):
            raise TypeError('Configuration type must be `_SliderConfig`.')
        self._slider_styles = config

    @property
    def axes(self) -> _AxesConfig:
        """Return or set the default ``axes`` configuration.

        Examples
        --------
        Set the x axis color to black.

        >>> import pyvista
        >>> pyvista.global_theme.axes.x_color = 'black'

        Show axes by default.

        >>> pyvista.global_theme.axes.show = True

        Use the ``vtk.vtkCubeAxesActor``.

        >>> pyvista.global_theme.axes.box = True

        """
        return self._axes

    @axes.setter
    def axes(self, config: _AxesConfig):
        if not isinstance(config, _AxesConfig):
            raise TypeError('Configuration type must be `_AxesConfig`.')
        self._axes = config

    def restore_defaults(self):
        """Restore the theme defaults.

        Examples
        --------
        >>> import pyvista
        >>> pyvista.global_theme.restore_defaults()

        """
        self.__init__()

    def __repr__(self):
        """User friendly representation of the current theme."""
        txt = [f'{self.name.capitalize()} Theme']
        txt.append('-'*len(txt[0]))
        parm = {
            'Background': 'background',
            'Jupyter backend': 'jupyter_backend',
            'Full screen': 'full_screen',
            'Window size': 'window_size',
            'Camera': 'camera',
            'Notebook': 'notebook',
            'Font': 'font',
            'Auto close': 'auto_close',
            'Colormap': 'cmap',
            'Color': 'color',
            'NaN color': 'nan_color',
            'Edge color': 'edge_color',
            'Outline color': 'outline_color',
            'Floor color': 'floor_color',
            'Colorbar orientation': 'colorbar_orientation',
            'Colorbar - horizontal': 'colorbar_horizontal',
            'Colorbar - vertical': 'colorbar_vertical',
            'Show scalar bar': 'show_scalar_bar',
            'Show edges': 'show_edges',
            'Lighting': 'lighting',
            'Interactive': 'interactive',
            'Render points as spheres': 'render_points_as_spheres',
            'Transparent Background': 'transparent_background',
            'Title': 'title',
            'Axes': 'axes',
            'Multi-samples': 'multi_samples',
            'Multi-renderer Split Pos': 'multi_rendering_splitting_position',
            'Volume mapper': 'volume_mapper',
            'Smooth shading': 'smooth_shading',
            'Depth peeling': 'depth_peeling',
            'Silhouette': 'silhouette',
            'Slider Styles': 'slider_styles',
        }
        for name, attr in parm.items():
            setting = getattr(self, attr)
            txt.append(f'{name:<25}: {setting}')

        return '\n'.join(txt)

    @property
    def name(self) -> str:
        """Return or set the name of the theme."""
        return self._name

    @name.setter
    def name(self, name: str):
        self._name = name

    def load_theme(self, theme):
        """Overwrite the current theme with a theme.

        Examples
        --------
        Create a custom theme from the default theme and load it into
        the global theme of pyvista.

        >>> import pyvista
        >>> from pyvista.themes import DefaultTheme
        >>> my_theme = DefaultTheme()
        >>> my_theme.font.size = 20
        >>> my_theme.font.title_size = 40
        >>> my_theme.cmap = 'jet'
        ...
        >>> pyvista.global_theme.load_theme(my_theme)
        >>> pyvista.global_theme.font.size
        20

        Create a custom theme from the dark theme and load it into
        pyvista.

        >>> from pyvista.themes import DarkTheme
        >>> my_theme = DarkTheme()
        >>> my_theme.show_edges = True
        >>> pyvista.global_theme.load_theme(my_theme)
        >>> pyvista.global_theme.show_edges
        True

        """
        if isinstance(theme, str):
            theme = load_theme(theme)

        if not isinstance(theme, DefaultTheme):
            raise TypeError('``theme`` must be a pyvista theme like '
                            '``pyvista.themes.DefaultTheme``.')

        for attr_name in theme.__slots__:
            setattr(self, attr_name, getattr(theme, attr_name))

    def save(self, filename):
        """Serialize this theme to a json file.

        Examples
        --------
        Export and then load back in a theme.

        >>> import pyvista
        >>> theme = pyvista.themes.DefaultTheme()
        >>> theme.background = 'white'
        >>> theme.save('my_theme.json')  # doctest:+SKIP
        >>> loaded_theme = pyvista.load_theme('my_theme.json')  # doctest:+SKIP

        """
        with open(filename, 'w') as f:
            json.dump(self.to_dict(), f)

    @property
    def use_ipyvtk(self):  # pragma: no cover
        """Set or return the usage of "ipyvtk" as a jupyter backend.

        Deprecated in favor of ``jupyter_backend``.
        """
        warnings.warn('use_ipyvtk is deprecated.  Please use '
                      '``pyvista.global_theme.jupyter_backend``', DeprecationWarning)
        return self.jupyter_backend == 'ipyvtklink'

    @use_ipyvtk.setter
    def use_ipyvtk(self, value):  # pragma: no cover
        warnings.warn('use_ipyvtk is deprecated.  Please use '
                      '``pyvista.global_theme.jupyter_backend``', DeprecationWarning)

        if value:
            self.jupyter_backend = 'ipyvtklink'
        else:
            self.jupyter_backend = 'static'


class DarkTheme(DefaultTheme):
    """Dark mode theme.

    Black background, "viridis" colormap, tan meshes, white (hidden) edges.

    Examples
    --------
    Make the dark theme the global default.

    >>> import pyvista
    >>> from pyvista import themes
    >>> pyvista.set_plot_theme(themes.DarkTheme())

    Alternatively, set via a string.

    >>> pyvista.set_plot_theme('dark')

    """

    def __init__(self):
        """Initialize the theme."""
        super().__init__()
        self.name = 'dark'
        self.background = 'black'
        self.cmap = 'viridis'
        self.font.color = 'white'
        self.show_edges = False
        self.color = 'tan'
        self.outline_color = 'white'
        self.edge_color = 'white'
        self.axes.x_color = 'tomato'
        self.axes.y_color = 'seagreen'
        self.axes.z_color = 'blue'


class ParaViewTheme(DefaultTheme):
    """A paraview-like theme.

    Examples
    --------
    Make the paraview-like theme the global default.

    >>> import pyvista
    >>> from pyvista import themes
    >>> pyvista.set_plot_theme(themes.ParaViewTheme())

    Alternatively, set via a string.

    >>> pyvista.set_plot_theme('paraview')

    """

    def __init__(self):
        """Initialize theme."""
        super().__init__()
        self.name = 'paraview'
        self.background = tuple(PARAVIEW_BACKGROUND)
        self.cmap = 'coolwarm'
        self.font.family = 'arial'
        self.font.label_size = 16
        self.font.color = 'white'
        self.show_edges = False
        self.color = 'white'
        self.outline_color = 'white'
        self.edge_color = 'black'
        self.axes.x_color = 'tomato'
        self.axes.y_color = 'gold'
        self.axes.z_color = 'green'


class DocumentTheme(DefaultTheme):
    """A document theme well suited for papers and presentations.

    This theme uses a white background, black fonts, the "viridis"
    colormap, and it disables edges.  Best used for presentations,
    papers, etc.

    Examples
    --------
    Make the document theme the global default.

    >>> import pyvista
    >>> from pyvista import themes
    >>> pyvista.set_plot_theme(themes.DocumentTheme())

    Alternatively, set via a string.

    >>> pyvista.set_plot_theme('document')

    """

    def __init__(self):
        """Initialize the theme."""
        super().__init__()
        self.name = 'document'
        self.background = 'white'
        self.cmap = 'viridis'
        self.font.size = 18
        self.font.title_size = 18
        self.font.label_size = 18
        self.font.color = 'black'
        self.show_edges = False
        self.color = 'tan'
        self.outline_color = 'black'
        self.edge_color = 'black'
        self.axes.x_color = 'tomato'
        self.axes.y_color = 'seagreen'
        self.axes.z_color = 'blue'


class _TestingTheme(DefaultTheme):
    """Low resolution testing theme for ``pytest``.

    Necessary for image regression.  Xvfb doesn't support
    multi-sampling, it's disabled for consistency between desktops and
    remote testing.
    """

    def __init__(self):
        super().__init__()
        self.name = 'testing'
        self.multi_samples = 1
        self.window_size = [400, 400]
        self.axes.show = False


class _ALLOWED_THEMES(Enum):
    """Global built-in themes available to PyVista."""

    paraview = ParaViewTheme
    document = DocumentTheme
    dark = DarkTheme
    default = DefaultTheme
    testing = _TestingTheme
