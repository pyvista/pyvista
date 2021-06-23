"""PyVista Scalar bar module."""

import pyvista
import numpy as np
from pyvista import _vtk
from .tools import parse_font_family, parse_color


class ScalarBars():
    """Plotter Scalar Bars."""

    def __init__(self, plotter):
        """Initialize ScalarBars."""
        self._plotter = plotter
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

    def _remove_mapper_from_plotter(self, actor, reset_camera=False, render=False):
        """Remove an actor's mapper from the given plotter's _scalar_bar_mappers."""
        try:
            mapper = actor.GetMapper()
        except AttributeError:
            return
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
                    self._plotter.remove_actor(self._scalar_bar_actors.pop(name),
                                               reset_camera=reset_camera,
                                               render=render)
                    self._plotter._scalar_bar_slots.add(slot)
            return

    def remove_scalar_bar(self, title=None, render=True):
        """Remove a scalar bar.

        Parameters
        ----------
        title : str, optional
            Title of the scalar bar to remove.  Required if there is
            more than one scalar bar.

        render : bool, optional
            Render upon scalar bar removal.  Set this to ``False`` to
            stop the render window from rendering when an scalar bar
            is removed.

        """
        if title is None:
            if len(self) > 1:
                titles = ', '.join('"{key}"' for key in self._scalar_bar_actors)
                raise ValueError('Multiple scalar bars.  Specify the title of the'
                                 f'scalar bar from one of the following:\n{titles}')
            else:
                title = list(self._scalar_bar_actors.keys())[0]

        self._scalar_bar_ranges.pop(title)
        self._scalar_bar_mappers.pop(title)
        widget = self._scalar_bar_widgets.pop(title, None)
        if widget is not None:
            widget.SetEnabled(0)

        actor = self._scalar_bar_actors.pop(title)
        self._plotter.remove_actor(actor, render=render)

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

    def add_scalar_bar(self, title='', mapper=None, n_labels=5, italic=False,
                       bold=False, title_font_size=None,
                       label_font_size=None, color=None,
                       font_family=None, shadow=False, width=None,
                       height=None, position_x=None, position_y=None,
                       vertical=None, interactive=None, fmt=None,
                       use_opacity=True, outline=False,
                       nan_annotation=False, below_label=None,
                       above_label=None, background_color=None,
                       n_colors=None, fill=False, render=False):
        """Create scalar bar using the ranges as set by the last input mesh.

        Parameters
        ----------
        title : string, optional
            Title of the scalar bar.  Default ``''`` which is
            rendered as an empty title.

        mapper : vtkMapper, optional
            Mapper used for the scalar bar.  Defaults to the last
            mapper created by the plotter.

        n_labels : int, optional
            Number of labels to use for the scalar bar.

        italic : bool, optional
            Italicises title and bar labels.  Default False.

        bold  : bool, optional
            Bolds title and bar labels.  Default True

        title_font_size : float, optional
            Sets the size of the title font.  Defaults to ``None`` and is sized
            according to ``pyvista.global_theme``.

        label_font_size : float, optional
            Sets the size of the title font.  Defaults to ``None`` and is sized
            according to ``pyvista.global_theme``.

        color : string or 3 item list, optional
            Either a string, rgb list, or hex color string.  Defaults to white.
            For example:

            * ``color='white'``
            * ``color='w'``
            * ``color=[1, 1, 1]``
            * ``color='#FFFFFF'``

        font_family : {'courier', 'times', 'arial'}
            Font family.  Default is set by ``pyvista.global_theme``.

        shadow : bool, optional
            Adds a black shadow to the text.  Defaults to ``False``.

        width : float, optional
            The percentage (0 to 1) width of the window for the colorbar.
            Default set by ``pyvista.global_theme``.

        height : float, optional
            The percentage (0 to 1) height of the window for the colorbar.
            Default set by ``pyvista.global_theme``.

        position_x : float, optional
            The percentage (0 to 1) along the windows's horizontal
            direction to place the bottom left corner of the colorbar.
            Default is automatic placement.

        position_y : float, optional
            The percentage (0 to 1) along the windows's vertical
            direction to place the bottom left corner of the colorbar.
            Default is automatic placement.

        vertical : bool, optional
            Use vertical or horizontal scalar bar.
            Default set by ``pyvista.global_theme``.

        interactive : bool, optional
            Use a widget to control the size and location of the scalar bar.
            Default set by ``pyvista.global_theme``.

        fmt : str, optional
            ``printf`` format for labels.
            Default set by ``pyvista.global_theme``.

        use_opacity : bool, optional
            Optionally display the opacity mapping on the scalar bar.

        outline : bool, optional
            Optionally outline the scalar bar to make opacity mappings more
            obvious.

        nan_annotation : bool, optional
            Annotate the NaN color.

        below_label : str, optional
            String annotation for values below the scalars range.

        above_label : str, optional
            String annotation for values above the scalars range.

        background_color : array, optional
            The color used for the background in RGB format.

        n_colors : int, optional
            The maximum number of color displayed in the scalar bar.

        fill : bool, optional
            Draw a filled box behind the scalar bar with the
            ``background_color``.

        render : bool, optional
            Force a render when True.  Default ``True``.

        Examples
        --------
        Add a custom interactive scalar bar that is horizontal, has an
        outline, and has a custom formatting.

        >>> import pyvista as pv
        >>> sphere = pv.Sphere()
        >>> sphere['Data'] = sphere.points[:, 2]
        >>> plotter = pv.Plotter()
        >>> _ = plotter.add_mesh(sphere, show_scalar_bar=False)
        >>> _ = plotter.add_scalar_bar('Data', interactive=True, vertical=False,
        ...                            outline=True, fmt='%10.5f')

        Notes
        -----
        Setting title_font_size, or label_font_size disables automatic font
        sizing for both the title and label.

        """
        if mapper is None:
            raise ValueError('Mapper cannot be ``None`` when creating a scalar bar')

        if interactive is None:
            interactive = pyvista.global_theme.interactive
        if font_family is None:
            font_family = pyvista.global_theme.font.family
        if label_font_size is None:
            label_font_size = pyvista.global_theme.font.label_size
        if title_font_size is None:
            title_font_size = pyvista.global_theme.font.title_size
        if color is None:
            color = pyvista.global_theme.font.color
        if fmt is None:
            fmt = pyvista.global_theme.font.fmt
        if vertical is None:
            if pyvista.global_theme.colorbar_orientation.lower() == 'vertical':
                vertical = True

        # Automatically choose size if not specified
        if width is None:
            if vertical:
                width = pyvista.global_theme.colorbar_vertical.width
            else:
                width = pyvista.global_theme.colorbar_horizontal.width
        if height is None:
            if vertical:
                height = pyvista.global_theme.colorbar_vertical.height
            else:
                height = pyvista.global_theme.colorbar_horizontal.height

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
            try:
                slot = min(self._plotter._scalar_bar_slots)
                self._plotter._scalar_bar_slots.remove(slot)
                self._plotter._scalar_bar_slot_lookup[title] = slot
            except:
                raise RuntimeError('Maximum number of color bars reached.')
            if position_x is None:
                if vertical:
                    position_x = pyvista.global_theme.colorbar_vertical.position_x
                    position_x -= slot * (width + 0.2 * width)
                else:
                    position_x = pyvista.global_theme.colorbar_horizontal.position_x

            if position_y is None:
                if vertical:
                    position_y = pyvista.global_theme.colorbar_vertical.position_y
                else:
                    position_y = pyvista.global_theme.colorbar_horizontal.position_y
                    position_y += slot * height

        # parse color
        color = parse_color(color)

        # Create scalar bar
        scalar_bar = _vtk.vtkScalarBarActor()
        # self._scalar_bars.append(scalar_bar)

        if background_color is not None:
            background_color = parse_color(background_color, opacity=1.0)
            background_color = np.array(background_color) * 255
            scalar_bar.GetBackgroundProperty().SetColor(background_color[0:3])

            if fill:
                scalar_bar.DrawBackgroundOn()

            lut = _vtk.vtkLookupTable()
            lut.DeepCopy(mapper.lookup_table)
            ctable = _vtk.vtk_to_numpy(lut.GetTable())
            alphas = ctable[:, -1][:, np.newaxis] / 255.
            use_table = ctable.copy()
            use_table[:, -1] = 255.
            ctable = (use_table * alphas) + background_color * (1 - alphas)
            lut.SetTable(_vtk.numpy_to_vtk(ctable, array_type=_vtk.VTK_UNSIGNED_CHAR))
        else:
            lut = mapper.lookup_table

        scalar_bar.SetLookupTable(lut)
        if n_colors is not None:
            scalar_bar.SetMaximumNumberOfColors(n_colors)

        if n_labels < 1:
            scalar_bar.DrawTickLabelsOff()
        else:
            scalar_bar.DrawTickLabelsOn()
            scalar_bar.SetNumberOfLabels(n_labels)

        if nan_annotation:
            scalar_bar.DrawNanAnnotationOn()

        if above_label:
            scalar_bar.DrawAboveRangeSwatchOn()
            scalar_bar.SetAboveRangeAnnotation(above_label)
        if below_label:
            scalar_bar.DrawBelowRangeSwatchOn()
            scalar_bar.SetBelowRangeAnnotation(below_label)

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
            scalar_bar.UnconstrainedFontSizeOn()
            scalar_bar.AnnotationTextScalingOn()

        label_text = scalar_bar.GetLabelTextProperty()
        anno_text = scalar_bar.GetAnnotationTextProperty()
        label_text.SetColor(color)
        anno_text.SetColor(color)
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
        title_text.SetColor(color)

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
                # y position determined emperically
                y = -position_y/2 - height - scalar_bar.GetPosition()[1]
                rep.GetPositionCoordinate().SetValue(width, y)
                rep.GetPosition2Coordinate().SetValue(height, width)
                rep.SetOrientation(0)  # 0 = Horizontal, 1 = Vertical
            self._scalar_bar_widgets[title] = scalar_widget

        if use_opacity:
            scalar_bar.SetUseOpacity(True)

        if outline:
            scalar_bar.SetDrawFrame(True)
            frame_prop = scalar_bar.GetFrameProperty()
            frame_prop.SetColor(color)
        else:
            scalar_bar.SetDrawFrame(False)

        # finally, add to the actor and return the scalar bar
        self._plotter.add_actor(scalar_bar, reset_camera=False,
                                pickable=False, render=render)
        return scalar_bar
