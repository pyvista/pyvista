"""PyVista Scalar bar module."""
import weakref

import numpy as np

import pyvista
from pyvista import MAX_N_COLOR_BARS

from . import _vtk
from .colors import Color
from .tools import parse_font_family


class ScalarBars:
    """Plotter Scalar Bars.

    Parameters
    ----------
    plotter : pyvista.Plotter
        Plotter that the scalar bars are associated with.

    """

    def __init__(self, plotter):
        """Initialize ScalarBars."""
        self._plotter = weakref.proxy(plotter)
        self._scalar_bar_ranges = {}
        self._scalar_bar_mappers = {}
        self._scalar_bar_actors = {}
        self._scalar_bar_widgets = {}

    def clear(self):
        """Remove all scalar bars and resets all scalar bar properties."""
        self._scalar_bar_ranges = {}
        self._scalar_bar_mappers = {}
        self._scalar_bar_actors = {}
        self._scalar_bar_widgets = {}

    def __repr__(self):
        """Nice representation of this class."""
        lines = []
        lines.append('Scalar Bar Title     Interactive')
        for title in self._scalar_bar_actors:
            interactive = title in self._scalar_bar_widgets
            title = f'"{title}"'
            lines.append(f'{title:20} {str(interactive):5}')
        return '\n'.join(lines)

    def _remove_mapper_from_plotter(
        self, actor, reset_camera=False, render=False
    ):  # numpydoc ignore=PR01,RT01
        """Remove an actor's mapper from the given plotter's _scalar_bar_mappers.

        This ensures that when actors are removed, their corresponding
        scalar bars are removed.

        """
        try:
            mapper = actor.GetMapper()
        except AttributeError:
            return

        # NOTE: keys to list to prevent iterator changing during loop
        for name in list(self._scalar_bar_mappers):
            try:
                self._scalar_bar_mappers[name].remove(mapper)
            except ValueError:
                pass

            if not self._scalar_bar_mappers[name]:
                slot = self._plotter._scalar_bar_slot_lookup.pop(name, None)
                if slot is not None:
                    self._scalar_bar_mappers.pop(name)
                    self._scalar_bar_ranges.pop(name)
                    self._plotter.remove_actor(
                        self._scalar_bar_actors.pop(name), reset_camera=reset_camera, render=render
                    )
                    self._plotter._scalar_bar_slots.add(slot)
            return

    def remove_scalar_bar(self, title=None, render=True):
        """Remove a scalar bar.

        Parameters
        ----------
        title : str, optional
            Title of the scalar bar to remove.  Required if there is
            more than one scalar bar.

        render : bool, default: True
            Render upon scalar bar removal.  Set this to ``False`` to
            stop the render window from rendering when a scalar bar
            is removed.

        Examples
        --------
        Remove a scalar bar from a plotter.

        >>> import pyvista as pv
        >>> mesh = pv.Sphere()
        >>> mesh['data'] = mesh.points[:, 2]
        >>> pl = pv.Plotter()
        >>> _ = pl.add_mesh(mesh, cmap='coolwarm')
        >>> pl.remove_scalar_bar()
        >>> pl.show()

        """
        if title is None:
            if len(self) > 1:
                titles = ', '.join(f'"{key}"' for key in self._scalar_bar_actors)
                raise ValueError(
                    'Multiple scalar bars found.  Pick title of the'
                    f'scalar bar from one of the following:\n{titles}'
                )
            else:
                title = list(self._scalar_bar_actors.keys())[0]

        actor = self._scalar_bar_actors.pop(title)
        self._plotter.remove_actor(actor, render=render)
        self._scalar_bar_ranges.pop(title)
        self._scalar_bar_mappers.pop(title)

        # add back in the scalar bar slot
        slot = self._plotter._scalar_bar_slot_lookup.pop(title, None)
        if slot is not None:
            self._plotter._scalar_bar_slots.add(slot)

        widget = self._scalar_bar_widgets.pop(title, None)
        if widget is not None:
            widget.SetEnabled(0)

    def __len__(self):
        """Return the number of scalar bar actors."""
        return len(self._scalar_bar_actors)

    def __getitem__(self, index):
        """Return a scalar bar actor."""
        return self._scalar_bar_actors[index]

    def keys(self):
        """Scalar bar keys."""
        return self._scalar_bar_actors.keys()

    def values(self):
        """Scalar bar values."""
        return self._scalar_bar_actors.values()

    def items(self):
        """Scalar bar items."""
        return self._scalar_bar_actors.items()

    def __contains__(self, key):
        """Check if a title is a valid actors."""
        return key in self._scalar_bar_actors

    def add_scalar_bar(
        self,
        title='',
        mapper=None,
        n_labels=5,
        italic=False,
        bold=False,
        title_font_size=None,
        label_font_size=None,
        color=None,
        font_family=None,
        shadow=False,
        width=None,
        height=None,
        position_x=None,
        position_y=None,
        vertical=None,
        interactive=None,
        fmt=None,
        use_opacity=True,
        outline=False,
        nan_annotation=False,
        below_label=None,
        above_label=None,
        background_color=None,
        n_colors=None,
        fill=False,
        render=False,
        theme=None,
        unconstrained_font_size=False,
    ):
        """Create scalar bar using the ranges as set by the last input mesh.

        Parameters
        ----------
        title : str, default: ""
            Title of the scalar bar.  Default is rendered as an empty title.

        mapper : vtkMapper, optional
            Mapper used for the scalar bar.  Defaults to the last
            mapper created by the plotter.

        n_labels : int, default: 5
            Number of labels to use for the scalar bar.

        italic : bool, default: False
            Italicises title and bar labels.

        bold : bool, default: False
            Bolds title and bar labels.

        title_font_size : float, optional
            Sets the size of the title font.  Defaults to ``None`` and is sized
            according to :attr:`pyvista.plotting.themes.Theme.font`.

        label_font_size : float, optional
            Sets the size of the title font.  Defaults to ``None`` and is sized
            according to :attr:`pyvista.plotting.themes.Theme.font`.

        color : ColorLike, optional
            Either a string, rgb list, or hex color string.  Default
            set by :attr:`pyvista.plotting.themes.Theme.font`.  Can be
            in one of the following formats:

            * ``color='white'``
            * ``color='w'``
            * ``color=[1.0, 1.0, 1.0]``
            * ``color='#FFFFFF'``

        font_family : {'courier', 'times', 'arial'}
            Font family.  Default is set by
            :attr:`pyvista.plotting.themes.Theme.font`.

        shadow : bool, default: False
            Adds a black shadow to the text.

        width : float, optional
            The percentage (0 to 1) width of the window for the colorbar.
            Default set by
            :attr:`pyvista.plotting.themes.Theme.colorbar_vertical` or
            :attr:`pyvista.plotting.themes.Theme.colorbar_horizontal`
            depending on the value of ``vertical``.

        height : float, optional
            The percentage (0 to 1) height of the window for the
            colorbar.  Default set by
            :attr:`pyvista.plotting.themes.Theme.colorbar_vertical` or
            :attr:`pyvista.plotting.themes.Theme.colorbar_horizontal`
            depending on the value of ``vertical``.

        position_x : float, optional
            The percentage (0 to 1) along the windows's horizontal
            direction to place the bottom left corner of the colorbar.
            Default set by
            :attr:`pyvista.plotting.themes.Theme.colorbar_vertical` or
            :attr:`pyvista.plotting.themes.Theme.colorbar_horizontal`
            depending on the value of ``vertical``.

        position_y : float, optional
            The percentage (0 to 1) along the windows's vertical
            direction to place the bottom left corner of the colorbar.
            Default set by
            :attr:`pyvista.plotting.themes.Theme.colorbar_vertical` or
            :attr:`pyvista.plotting.themes.Theme.colorbar_horizontal`
            depending on the value of ``vertical``.

        vertical : bool, optional
            Use vertical or horizontal scalar bar.  Default set by
            :attr:`pyvista.plotting.themes.Theme.colorbar_orientation`.

        interactive : bool, optional
            Use a widget to control the size and location of the scalar bar.
            Default set by :attr:`pyvista.plotting.themes.Theme.interactive`.

        fmt : str, optional
            ``printf`` format for labels.
            Default set by :attr:`pyvista.plotting.themes.Theme.font`.

        use_opacity : bool, default: True
            Optionally display the opacity mapping on the scalar bar.

        outline : bool, default: False
            Optionally outline the scalar bar to make opacity mappings more
            obvious.

        nan_annotation : bool, default: False
            Annotate the NaN color.

        below_label : str, optional
            String annotation for values below the scalars range.

        above_label : str, optional
            String annotation for values above the scalars range.

        background_color : ColorLike, optional
            The color used for the background in RGB format.

        n_colors : int, optional
            The maximum number of color displayed in the scalar bar.

        fill : bool, default: False
            Draw a filled box behind the scalar bar with the
            ``background_color``.

        render : bool, default: False
            Force a render when True.

        theme : pyvista.plotting.themes.Theme, optional
            Plot-specific theme.  By default, calling from the
            ``Plotter``, will use the plotter theme.  Setting to
            ``None`` will use the global theme.

        unconstrained_font_size : bool, default: False
            Whether the font size of title and labels is unconstrained.
            When it is constrained, the size of the scalar bar will constrain the font size.
            When it is not, the size of the font will always be respected.
            Using custom labels will force this to be ``True``.

            .. versionadded:: 0.44.0

        Returns
        -------
        vtk.vtkScalarBarActor
            Scalar bar actor.

        Notes
        -----
        Setting ``title_font_size``, or ``label_font_size`` disables
        automatic font sizing for both the title and label.

        Examples
        --------
        Add a custom interactive scalar bar that is horizontal, has an
        outline, and has a custom formatting.

        >>> import pyvista as pv
        >>> sphere = pv.Sphere()
        >>> sphere['Data'] = sphere.points[:, 2]
        >>> plotter = pv.Plotter()
        >>> _ = plotter.add_mesh(sphere, show_scalar_bar=False)
        >>> _ = plotter.add_scalar_bar(
        ...     'Data',
        ...     interactive=True,
        ...     vertical=False,
        ...     title_font_size=35,
        ...     label_font_size=30,
        ...     outline=True,
        ...     fmt='%10.5f',
        ... )
        >>> plotter.show()

        """
        if mapper is None:
            raise ValueError('Mapper cannot be ``None`` when creating a scalar bar')

        if theme is None:
            theme = pyvista.global_theme

        if interactive is None:
            interactive = theme.interactive
        if font_family is None:
            font_family = theme.font.family
        if label_font_size is None:
            label_font_size = theme.font.label_size
        if title_font_size is None:
            title_font_size = theme.font.title_size
        if fmt is None:
            fmt = theme.font.fmt
        if vertical is None:
            if theme.colorbar_orientation.lower() == 'vertical':
                vertical = True

        # Automatically choose size if not specified
        if width is None:
            if vertical:
                width = theme.colorbar_vertical.width
            else:
                width = theme.colorbar_horizontal.width
        if height is None:
            if vertical:
                height = theme.colorbar_vertical.height
            else:
                height = theme.colorbar_horizontal.height

        # Check that this data hasn't already been plotted
        if title in list(self._scalar_bar_ranges.keys()):
            clim = list(self._scalar_bar_ranges[title])
            newrng = mapper.scalar_range
            oldmappers = self._scalar_bar_mappers[title]
            # get max for range and reset everything
            if newrng[0] < clim[0]:
                clim[0] = newrng[0]
            if newrng[1] > clim[1]:
                clim[1] = newrng[1]
            for mh in oldmappers:
                mh.scalar_range = clim[0], clim[1]
            mapper.scalar_range = clim[0], clim[1]
            self._scalar_bar_mappers[title].append(mapper)
            self._scalar_bar_ranges[title] = clim
            self._scalar_bar_actors[title].SetLookupTable(mapper.lookup_table)
            # Color bar already present and ready to be used so returning
            return

        # Automatically choose location if not specified
        if position_x is None or position_y is None:
            if not self._plotter._scalar_bar_slots:
                raise RuntimeError(f'Maximum number of color bars ({MAX_N_COLOR_BARS}) reached.')

            slot = min(self._plotter._scalar_bar_slots)
            self._plotter._scalar_bar_slots.remove(slot)
            self._plotter._scalar_bar_slot_lookup[title] = slot

            if position_x is None:
                if vertical:
                    position_x = theme.colorbar_vertical.position_x
                    position_x -= slot * (width + 0.2 * width)
                else:
                    position_x = theme.colorbar_horizontal.position_x

            if position_y is None:
                if vertical:
                    position_y = theme.colorbar_vertical.position_y
                else:
                    position_y = theme.colorbar_horizontal.position_y
                    position_y += slot * height

        # parse color
        color = Color(color, default_color=theme.font.color)

        # Create scalar bar
        scalar_bar = _vtk.vtkScalarBarActor()
        # self._scalar_bars.append(scalar_bar)

        if background_color is not None:
            background_color = np.array(Color(background_color).int_rgba)
            scalar_bar.GetBackgroundProperty().SetColor(background_color[0:3])

            if fill:
                scalar_bar.DrawBackgroundOn()

            lut = pyvista.LookupTable()
            lut.DeepCopy(mapper.lookup_table)
            ctable = _vtk.vtk_to_numpy(lut.GetTable())
            alphas = ctable[:, -1][:, np.newaxis] / 255.0
            use_table = ctable.copy()
            use_table[:, -1] = 255.0
            ctable = (use_table * alphas) + background_color * (1 - alphas)
            lut.SetTable(_vtk.numpy_to_vtk(ctable, array_type=_vtk.VTK_UNSIGNED_CHAR))
        else:
            lut = mapper.lookup_table

        scalar_bar.SetLookupTable(lut)
        if n_colors is None:
            # ensure the number of colors in the scalarbar's lookup table is at
            # least the number in the mapper
            n_colors = mapper.lookup_table.n_values

        scalar_bar.SetMaximumNumberOfColors(n_colors)

        if n_labels < 1:
            scalar_bar.SetDrawTickLabels(False)
        else:
            scalar_bar.SetDrawTickLabels(True)
            scalar_bar.SetNumberOfLabels(n_labels)

        if nan_annotation:
            scalar_bar.DrawNanAnnotationOn()

        if above_label is not None:
            scalar_bar.DrawAboveRangeSwatchOn()
            scalar_bar.SetAboveRangeAnnotation(above_label)
        elif lut.above_range_color:
            scalar_bar.DrawAboveRangeSwatchOn()
            scalar_bar.SetAboveRangeAnnotation('above')
        if below_label is not None:
            scalar_bar.DrawBelowRangeSwatchOn()
            scalar_bar.SetBelowRangeAnnotation(below_label)
        elif lut.below_range_color:
            scalar_bar.DrawBelowRangeSwatchOn()
            scalar_bar.SetBelowRangeAnnotation('below')

        # edit the size of the colorbar
        scalar_bar.SetHeight(height)
        scalar_bar.SetWidth(width)
        scalar_bar.SetPosition(position_x, position_y)

        if fmt is not None:
            scalar_bar.SetLabelFormat(fmt)

        if vertical:
            scalar_bar.SetOrientationToVertical()
        else:
            scalar_bar.SetOrientationToHorizontal()

        if label_font_size is not None or title_font_size is not None:
            scalar_bar.SetUnconstrainedFontSize(True)
            scalar_bar.SetAnnotationTextScaling(False)
        else:
            scalar_bar.SetAnnotationTextScaling(True)

        label_text = scalar_bar.GetLabelTextProperty()
        anno_text = scalar_bar.GetAnnotationTextProperty()
        label_text.SetColor(color.float_rgb)
        anno_text.SetColor(color.float_rgb)
        label_text.SetShadow(shadow)
        anno_text.SetShadow(shadow)

        # Set font
        label_text.SetFontFamily(parse_font_family(font_family))
        anno_text.SetFontFamily(parse_font_family(font_family))
        label_text.SetItalic(italic)
        anno_text.SetItalic(italic)
        label_text.SetBold(bold)
        anno_text.SetBold(bold)
        if label_font_size:
            label_text.SetFontSize(label_font_size)
            anno_text.SetFontSize(label_font_size)

        # Set properties
        self._scalar_bar_ranges[title] = mapper.scalar_range
        self._scalar_bar_mappers[title] = [mapper]

        scalar_bar.SetTitle(title)
        title_text = scalar_bar.GetTitleTextProperty()

        title_text.SetJustificationToCentered()

        title_text.SetItalic(italic)
        title_text.SetBold(bold)
        title_text.SetShadow(shadow)
        if title_font_size:
            title_text.SetFontSize(title_font_size)

        # Set font
        title_text.SetFontFamily(parse_font_family(font_family))

        # set color
        title_text.SetColor(color.float_rgb)

        self._scalar_bar_actors[title] = scalar_bar
        if interactive:
            scalar_widget = _vtk.vtkScalarBarWidget()
            scalar_widget.SetScalarBarActor(scalar_bar)
            scalar_widget.SetInteractor(self._plotter.iren.interactor)
            scalar_widget.SetEnabled(1)
            rep = scalar_widget.GetRepresentation()

            scalar_widget.On()
            if vertical is True or vertical is None:
                rep.SetOrientation(1)  # 0 = Horizontal, 1 = Vertical
            else:
                # y position determined empirically
                y = -position_y / 2 - height - scalar_bar.GetPosition()[1]
                rep.GetPositionCoordinate().SetValue(width, y)
                rep.GetPosition2Coordinate().SetValue(height, width)
                rep.SetOrientation(0)  # 0 = Horizontal, 1 = Vertical
            self._scalar_bar_widgets[title] = scalar_widget

        if use_opacity:
            scalar_bar.SetUseOpacity(True)

        if outline:
            scalar_bar.SetDrawFrame(True)
            frame_prop = scalar_bar.GetFrameProperty()
            frame_prop.SetColor(color.float_rgb)
        else:
            scalar_bar.SetDrawFrame(False)

        if unconstrained_font_size:
            scalar_bar.SetUnconstrainedFontSize(True)

        # finally, add to the actor and return the scalar bar
        self._plotter.add_actor(scalar_bar, reset_camera=False, pickable=False, render=render)

        return scalar_bar
