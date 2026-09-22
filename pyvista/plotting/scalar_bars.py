"""PyVista Scalar bar module."""

from __future__ import annotations

import contextlib
import math
from typing import Any
import weakref

import pyvista_validation as _validation

import pyvista as pv
from pyvista import MAX_N_COLOR_BARS
from pyvista import _vtk
from pyvista.core.errors import VTKVersionError
from pyvista.core.utilities.arrays import convert_array
from pyvista.core.utilities.misc import _NoNewAttrMixin

from .colors import Color
from .tools import parse_font_family


def _title_width(text_property, title, dpi):
    """Return the width in pixels of a title rendered with this text property."""
    bounds = [0, 0, 0, 0]
    _vtk.vtkFreeTypeTools.GetInstance().GetBoundingBox(text_property, title, dpi, bounds)
    return bounds[1] - bounds[0] + 1


def _format_label(fmt, value):
    """Return a tick value formatted the way the scalar bar formats it."""
    try:
        return fmt % value
    except (TypeError, ValueError):
        return fmt.format(value)


def _custom_anchor(value, *, first, low, span, log_scale):
    """Return how far along the ramp a custom tick sits, or ``-1`` where it is not drawn."""
    if span <= 0:
        return 0.5 if value == first else -1.0
    if log_scale:
        return (math.log10(value) - low) / span if value > 0 else -1.0
    return (value - low) / span


def _label_ticks(scalar_bar):
    """Return how far along the ramp each tick label sits and the text it is drawn with."""
    fmt = scalar_bar.GetLabelFormat()
    lookup_table = scalar_bar.GetLookupTable()
    first, last = lookup_table.GetRange()
    log_scale = bool(lookup_table.UsingLogScale()) and first > 0 and last > 0
    low, high = (math.log10(first), math.log10(last)) if log_scale else (first, last)
    span = high - low
    if scalar_bar.GetUseCustomLabels():
        custom = scalar_bar.GetCustomLabels()
        count = custom.GetNumberOfTuples() if custom is not None else 0
        values = [custom.GetValue(index) for index in range(count)]
        anchors = [
            _custom_anchor(value, first=first, low=low, span=span, log_scale=log_scale)
            for value in values
        ]
    else:
        ticks = int(scalar_bar.GetNumberOfLabels())
        anchors = [step / (ticks - 1) if ticks > 1 else 0.5 for step in range(ticks)]
        places = [low + anchor * span for anchor in anchors]
        values = [10**place for place in places] if log_scale else places
    return [
        (anchor, _format_label(fmt, value)) for anchor, value in zip(anchors, values, strict=True)
    ]


def _label_texts(scalar_bar):
    """Return the text of every tick label a scalar bar lays out, drawn or not."""
    return [text for _, text in _label_ticks(scalar_bar)]


def _label_size(scalar_bar, text_property, dpi):
    """Return the size in pixels of the widest tick label a scalar bar draws."""
    fmt = scalar_bar.GetLabelFormat()
    widest, height = 0.0, 0.0
    low, high = scalar_bar.GetLookupTable().GetRange()
    # An interior tick can be wider than either end, so measure every one of them
    ticks = max(int(scalar_bar.GetNumberOfLabels()), 2)
    for step in range(ticks):
        text = _format_label(fmt, low + (high - low) * step / (ticks - 1))
        width = _title_width(text_property, text, dpi)
        if width >= widest:
            widest, height = width, _title_height(text_property, text, dpi)
    return widest, height


def _title_height(text_property, title, dpi):
    """Return the height in pixels of a title rendered with this text property."""
    bounds = [0, 0, 0, 0]
    _vtk.vtkFreeTypeTools.GetInstance().GetBoundingBox(text_property, title, dpi, bounds)
    return bounds[3] - bounds[2] + 1


def _bar_title_height(scalar_bar, dpi):
    """Return the height of a scalar bar's title, ignoring any offset applied to it."""
    probe = _vtk.vtkTextProperty()
    probe.ShallowCopy(scalar_bar.GetTitleTextProperty())
    probe.SetLineOffset(0)
    return _title_height(probe, scalar_bar.GetTitle(), dpi)


def _turned_title(scalar_bar):
    """Return whether a scalar bar draws its title alongside the bar."""
    return pv.vtk_version_info >= (9, 4, 0) and bool(scalar_bar.GetForceVerticalTitle())


def _title_separation(scalar_bar):
    """Return the space a scalar bar leaves between its title and its labels.

    Only meaningful for a title drawn across the end of the bar, since a turned
    one carries the offset that moves it alongside instead.
    """
    return -scalar_bar.GetTitleTextProperty().GetLineOffset()


def _rotated_title_offset(bar_width, title_height, pad):
    """Return the line offset that clears a rotated title off its bar by ``pad`` pixels."""
    # The offset moves the pen and grows the text bounds, so the title moves two pixels
    # per unit; the constant is measured across font sizes and bar widths
    return round((bar_width + title_height) / 2 - 2 + pad / 2)


def _layout_settings(scalar_bar):
    """Return the box a scalar bar draws, its proportions, and the size of its text."""
    return (
        scalar_bar.GetWidth(),
        scalar_bar.GetHeight(),
        scalar_bar.GetPosition(),
        scalar_bar.GetBarRatio(),
        scalar_bar.GetVerticalTitleSeparation(),
        scalar_bar.GetTitleRatio(),
        scalar_bar.GetTextPad(),
        scalar_bar.GetTitleTextProperty().GetFontSize(),
        scalar_bar.GetLabelTextProperty().GetFontSize(),
    )


def _text_size(viewport, text_property, text, *, font_size):
    """Return the size in pixels of text as a scalar bar's layout measures it."""
    probe = _vtk.vtkTextActor()
    probe_text = probe.GetTextProperty()
    probe_text.ShallowCopy(text_property)
    probe_text.SetFontSize(font_size)
    probe.SetInput(text)
    size = [0.0, 0.0]
    probe.GetSize(viewport, size)
    return size[0], size[1]


def _fitting_font(size_of, target_width, target_height, *, start):
    """Return the font size VTK's constrained layout settles on for a target box.

    The layout grows the font while the text fits the box and shrinks it while it does
    not, so it lands on the largest size that fits, and two sizes that measure the same
    pixel height are told apart by nothing.
    """

    def fits(font_size):
        width, height = size_of(font_size)
        return width <= target_width and height <= target_height

    font_size = max(start, 3)
    if not fits(font_size):
        return _shrunk_font(fits, start=font_size, floor=True)
    while font_size < 100 and fits(font_size + 1):
        font_size += 1
    return font_size


def _shrunk_font(fits, *, start, floor=False):
    """Return the largest font size up to ``start`` that fits.

    Where nothing fits, ``floor`` settles for the smallest size there is rather than
    handing back the size that was asked for.
    """
    font_size = max(int(start), 3)
    while font_size > 3 and not fits(font_size):
        font_size -= 1
    return font_size if floor or fits(font_size) else start


def _fitted_label_font(scalar_bar, *, vertical, title, viewport, start):
    """Return the largest label font size up to ``start`` that keeps the labels apart.

    VTK centers each tick label on its own tick and, with the font size unconstrained,
    draws the label at the size it asks for and leaves the ramp its full length, so the
    labels run into each other once the bar is shorter than the text needs.  A label is
    free to run past the edge of the viewport, and the size asked for is kept where the
    labels clear each other at no size.
    """
    # A bar can turn its tick labels off, an indexed lookup draws annotations in their
    # place, and a tick whose value falls outside the range is laid out but not drawn
    draws_ticks = scalar_bar.GetDrawTickLabels() and not (
        scalar_bar.GetLookupTable().GetIndexedLookup()
    )
    ticks = [tick for tick in _label_ticks(scalar_bar) if 0 <= tick[0] <= 1] if draws_ticks else []
    if len(ticks) < 2:
        return start
    ticks.sort()

    label_text = scalar_bar.GetLabelTextProperty()
    text_pad = scalar_bar.GetTextPad()
    box_width, box_height = _box_pixels(scalar_bar, viewport)
    thickness = math.ceil(scalar_bar.GetBarRatio() * (box_width if vertical else box_height))
    if vertical:
        # The bar gives up the end its title is drawn across, and the labels are stacked
        # along what is left of the ramp
        title_text = scalar_bar.GetTitleTextProperty()
        title_height = (
            _text_size(viewport, title_text, title, font_size=title_text.GetFontSize())[1]
            if title
            else 0
        )
        room = _tick_run(
            scalar_bar,
            height=box_height,
            thickness=thickness,
            title_height=title_height,
            text_pad=text_pad,
        )
    else:
        room = _ramp_room(scalar_bar, box_width, thickness)

    def labels_fit(font_size):
        # Each label has to clear the label before it
        edge = None
        for anchor, text in ticks:
            # VTK draws the text into whole pixels and reports a size a pixel under the
            # room it takes, so a label is measured as the pixels it covers
            width, height = (
                math.ceil(size) + 1
                for size in _text_size(viewport, label_text, text, font_size=font_size)
            )
            reach = (height if vertical else width) / 2
            center = room * anchor
            if edge is not None and center - reach < edge:
                return False
            edge = center + reach + text_pad
        return True

    return _shrunk_font(labels_fit, start=start)


def _box_pixels(scalar_bar, viewport):
    """Return the width and height in pixels VTK measures a scalar bar's box at."""
    corner = scalar_bar.GetPositionCoordinate().GetComputedViewportValue(viewport)
    far_corner = scalar_bar.GetPosition2Coordinate().GetComputedViewportValue(viewport)
    return far_corner[0] - corner[0], far_corner[1] - corner[1]


def _set_box_height(scalar_bar, viewport, pixels):
    """Give a scalar bar the box height VTK measures as this many pixels."""
    viewport_height = viewport.GetSize()[1]
    height = pixels / viewport_height
    scalar_bar.SetHeight(height)
    # The corners are rounded to pixels one at a time, so a pixel can go missing
    measured = _box_pixels(scalar_bar, viewport)[1]
    scalar_bar.SetHeight(height + (pixels - measured) / viewport_height)


def _lifted_ramp(ramp, text_pad):
    """Return the bar thickness VTK thins to ``ramp`` and how far it lifts the ramp."""
    thickness = ramp
    while int(thickness - min(thickness / 8, text_pad)) != ramp:
        thickness += 1
    return thickness, int(min(thickness / 8, text_pad))


def _swatch_sizes(scalar_bar, length, thickness):
    """Return the pad VTK leaves around a bar's swatches and the size of each one."""
    notes = scalar_bar.GetLookupTable().GetNumberOfAnnotatedValues()
    per_note = int(length) // notes if notes else 0
    swatch_pad = 4.0 if not notes or per_note > 16 else per_note / 4
    size = max(min(thickness, int(length) // 4), 4 * (length > 16))
    drawn = (
        scalar_bar.GetDrawNanAnnotation(),
        scalar_bar.GetDrawBelowRangeSwatch(),
        scalar_bar.GetDrawAboveRangeSwatch(),
    )
    return (swatch_pad, *(size if is_drawn else 0 for is_drawn in drawn))


def _ramp_room(scalar_bar, width, ramp):
    """Return the length in pixels VTK leaves a horizontal ramp beside its swatches."""
    swatch_pad, nan, below, above = _swatch_sizes(scalar_bar, width, ramp)
    room = int(width - (nan + swatch_pad))
    if below:
        room = int(room - (below + swatch_pad))
    if above:
        room -= above
        if nan:
            room = int(room - swatch_pad)
    return room


def _tick_run(scalar_bar, *, height, thickness, title_height, text_pad):
    """Return the span in pixels a vertical bar lays its tick labels out along.

    The ramp is cleared of every swatch and of the title drawn across its end.
    """
    swatch_pad, nan, below, above = _swatch_sizes(scalar_bar, height, thickness)
    span = height - title_height - 3 * text_pad - scalar_bar.GetVerticalTitleSeparation()
    for size in (nan, below, above):
        if size:
            # A swatch gives the text pad back once it is deep enough to spare it
            span -= (size - text_pad if size > 2 * text_pad else size) + swatch_pad
    return span


def _constrained_box(scalar_bar, *, title, pad, viewport, keep_height=False):
    """Return the box that holds a horizontal bar's text at the font sizes it asked for.

    VTK sizes the text of a horizontal bar to the box it is given, growing each font to
    the largest that fits its share of the box, and pulls the ramp in by half a label so
    the end labels stay inside.  The box is sized so that share holds the title and the
    labels at the sizes they asked for, or one size larger where two sizes measure the
    same height, with the ramp as thick as the bar's own size makes it and the title
    padded off the labels by ``pad``.  Text wider than the box is shrunk to fit it.  A
    box that keeps its height spends what it has to spare on that padding and on the
    ramp instead, and shrinks its text when it has too little.  Returns the height in
    pixels, with the bar ratio, the title ratio and the text pad that lay the box out
    that way.
    """
    line_width = int(scalar_bar.GetFrameProperty().GetLineWidth())
    label_text = scalar_bar.GetLabelTextProperty()
    title_text = scalar_bar.GetTitleTextProperty()
    box_width, box_height = _box_pixels(scalar_bar, viewport)
    # The ramp is as thick as the height asked for makes it, whichever pixel the box
    # lands on, so bars asking for the same height get the same ramp
    thickness = math.ceil(
        scalar_bar.GetHeight() * viewport.GetSize()[1] * scalar_bar.GetBarRatio()
    )
    ramp = int(thickness - min(thickness / 8, scalar_bar.GetTextPad()))
    labels = _label_texts(scalar_bar)

    def title_sizes(font_size):
        return _text_size(viewport, title_text, title, font_size=font_size)

    def label_sizes(font_size):
        sizes = [_text_size(viewport, label_text, text, font_size=font_size) for text in labels]
        return max(width for width, _ in sizes), max(height for _, height in sizes)

    def title_size(text_pad):
        # A title wider than the box is shrunk until it fits
        if not title:
            return 0.0, 0.0
        font_size = title_text.GetFontSize()
        while font_size > 3 and title_sizes(font_size)[0] > box_width - 2 * text_pad:
            font_size -= 1
        return title_sizes(font_size)

    def label_slot(text_pad, ramp):
        return int(
            (_ramp_room(scalar_bar, box_width, ramp) - text_pad * (len(labels) - 1)) / len(labels)
        )

    def label_size(text_pad, ramp):
        # Labels wider than their share of the ramp are shrunk until they fit
        if not labels:
            return 0.0, 0.0
        slot = label_slot(text_pad, ramp)
        font_size = label_text.GetFontSize()
        while font_size > 3 and label_sizes(font_size)[0] > slot:
            font_size -= 1
        return label_sizes(font_size)

    def widest_pad(ramp):
        # The pad also spaces the labels along the ramp and the title off the sides of
        # the box, so spare room widens it only as far as it narrows neither
        cap = box_width
        if title:
            cap = min(cap, (box_width - title_size(1)[0]) // 2)
        if len(labels) > 1:
            room = _ramp_room(scalar_bar, box_width, ramp)
            cap = min(cap, (room - len(labels) * label_size(1, ramp)[0]) // (len(labels) - 1))
        return int(cap)

    # The text is padded off the ramp and the frame by the text pad, and the title off
    # the labels by twice that less the room the ramp is lifted off the frame
    text_pad = 1
    while 2 * text_pad - line_width - _lifted_ramp(ramp, text_pad)[1] < pad:
        text_pad += 1
    thickness, lift = _lifted_ramp(ramp, text_pad)
    title_height = title_size(text_pad)[1]
    title_box = math.ceil(title_height)
    label_height = label_size(text_pad, ramp)[1]
    height = ramp + 4 * text_pad + title_box + int(label_height)

    if keep_height and box_height < height:
        # Too little room for the text at the sizes asked for, so the title and the
        # labels share what there is the way their heights compare, the title taking
        # what the labels cannot use, and neither growing past the size it asked for
        text_pad = 1
        thickness, lift = _lifted_ramp(ramp, text_pad)
        room = max(box_height - ramp - 4 * text_pad, 2)
        share = int(room * title_box / max(title_box + int(label_height), 1))
        title_height = min(max(share, room - int(label_height)), int(title_size(text_pad)[1]))
        if title and labels:
            # The labels are not to outgrow a title that asked to be the larger
            title_font = min(
                _fitting_font(
                    title_sizes,
                    box_width - 2 * text_pad,
                    title_height,
                    start=title_text.GetFontSize(),
                ),
                title_text.GetFontSize(),
            )
            label_font = min(
                _fitting_font(
                    label_sizes,
                    label_slot(text_pad, ramp),
                    room - math.ceil(title_sizes(title_font)[1]),
                    start=label_text.GetFontSize(),
                ),
                label_text.GetFontSize(),
            )
            if label_font > title_font and title_text.GetFontSize() >= label_text.GetFontSize():
                title_height = int(label_sizes(label_font)[1])
        height = box_height
    elif keep_height:
        # Spare room pads the text and the rest thickens the ramp.  A thicker ramp
        # widens the swatches beside it, so the labels are sized against the thickest
        spare = box_height - height
        label_height = label_size(text_pad, ramp + spare)[1]
        spare = box_height - (ramp + 4 * text_pad + title_box + int(label_height))
        text_pad = max(min(text_pad + spare // 4, widest_pad(ramp + spare)), text_pad)
        ramp = box_height - 4 * text_pad - title_box - int(label_height)
        thickness, lift = _lifted_ramp(ramp, text_pad)
        height = box_height

    height = max(height, 1)

    bar_ratio = (thickness - 0.5) / height
    title_ratio = 0.5
    if title:
        title_ratio = min((int(title_height) + 0.5) / max(height - ramp - lift - text_pad, 1), 1.0)
    return height, bar_ratio, title_ratio, text_pad


def _fitted_box(scalar_bar, *, vertical, title, label_text, pad, dpi, viewport):
    """Return the box that encloses a scalar bar's text, and the settings that fill it.

    The colour ramp keeps the size it was given; the box grows around it.  Returns the
    box width and height as a fraction of the viewport, the bar ratio that holds the
    ramp to its original size, the line offset that seats the title inside the box, and
    the separation that leaves the title its padding.
    """
    viewport_width, viewport_height = viewport.GetSize()
    text_pad = scalar_bar.GetTextPad()
    label_width, label_height = _label_size(scalar_bar, label_text, dpi)
    title_width = _title_width(scalar_bar.GetTitleTextProperty(), title, dpi)
    title_height = _title_height(scalar_bar.GetTitleTextProperty(), title, dpi)
    box_width = scalar_bar.GetWidth() * viewport_width
    box_height = scalar_bar.GetHeight() * viewport_height

    if vertical:
        ramp = scalar_bar.GetBarRatio() * box_width
        # The title is centered on the box and the labels are drawn past the ramp.  The
        # width the bar was given only sets how thick the ramp is, so the box is free to
        # be no wider than the text needs however large the window grows
        box_width = max(title_width + 2 * text_pad, ramp + label_width + 2 * text_pad)
        bar_ratio = ramp / box_width
        # A vertical title is lifted clear of the box by three quarters of a label, and
        # the offset that seats it again grows the title box at the ramp's expense.  The
        # offset carries the title and the ramp down together, so the padding between
        # them is the separation VTK leaves rather than anything the offset can buy
        offset = round(0.75 * label_height)
        separation = pad
        box_height += offset + pad + text_pad
    else:
        ramp = scalar_bar.GetBarRatio() * box_height
        # A horizontal title is stacked above the ramp and the labels, measured from the
        # bottom of the box, so the box only has to be tall enough to cover the stack
        box_height = ramp + label_height + title_height + pad + 2 * text_pad
        bar_ratio = ramp / box_height
        offset = -pad
        separation = 0

    return (
        box_width / viewport_width,
        box_height / viewport_height,
        bar_ratio,
        offset,
        separation,
    )


class ScalarBars(_NoNewAttrMixin):
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
        self._resync_titles: set[str] = set()
        self._scalar_bar_actors = {}
        self._scalar_bar_widgets = {}
        self._scalar_bar_fits: dict[str, dict[str, Any]] = {}

    def clear(self):
        """Remove all scalar bars and resets all scalar bar properties."""
        self._scalar_bar_ranges = {}
        self._scalar_bar_mappers = {}
        self._resync_titles = set()
        self._scalar_bar_actors = {}
        self._scalar_bar_widgets = {}
        for title in list(self._scalar_bar_fits):
            self._stop_fitting(title)

    def __plotter_close__(self) -> None:
        """Release scalar-bar state when the owning plotter closes."""
        self.clear()

    def __repr__(self):
        """Nice representation of this class."""
        lines = []
        lines.append('Scalar Bar Title     Interactive')
        for title in self._scalar_bar_actors:
            interactive = title in self._scalar_bar_widgets
            title_quotes = f'"{title}"'
            lines.append(f'{title_quotes:20} {interactive!s:5}')
        return '\n'.join(lines)

    def _stop_fitting(self, title):
        """Drop the observer that keeps a scalar bar's box fitted to its text."""
        fit = self._scalar_bar_fits.pop(title, None)
        if fit is None:
            return
        window = self._plotter.render_window
        if window is not None:
            window.RemoveObserver(fit['observer'])

    def _apply_fit(self, fit, scalar_bar):
        """Size a scalar bar's box to its text, or give it back the size it asked for."""
        (
            width,
            height,
            position,
            bar_ratio,
            separation,
            title_ratio,
            text_pad,
            title_font,
            label_font,
        ) = fit['request']
        # Measure against the size that was asked for rather than the last fit
        scalar_bar.SetWidth(width)
        scalar_bar.SetHeight(height)
        scalar_bar.SetBarRatio(bar_ratio)
        scalar_bar.SetVerticalTitleSeparation(separation)
        scalar_bar.SetTitleRatio(title_ratio)
        scalar_bar.SetTextPad(text_pad)
        scalar_bar.SetPosition(*position)
        title_text = scalar_bar.GetTitleTextProperty()
        title_text.SetFontSize(title_font)
        draws_box = scalar_bar.GetDrawFrame() or scalar_bar.GetDrawBackground()

        if not draws_box or fit['unconstrained']:
            # The text keeps the size it asks for, so the labels are held apart at a
            # size the bar has room for
            label_font = _fitted_label_font(
                scalar_bar,
                vertical=fit['vertical'],
                title=fit['title'],
                viewport=fit['renderer'],
                start=label_font,
            )
        scalar_bar.GetLabelTextProperty().SetFontSize(label_font)

        if not draws_box:
            # Nothing is drawn around the text, so there is nothing to fit it to
            scalar_bar.SetUnconstrainedFontSize(True)
            title_text.SetLineOffset(-fit['pad'])
            self._place_widget(fit['key'], scalar_bar)
            fit['applied'] = _layout_settings(scalar_bar)
            return

        if not (fit['vertical'] or fit['unconstrained']):
            # VTK lays a horizontal box out itself, sizing the text to the box, so the
            # box is sized to leave the text the size it asked for instead
            scalar_bar.SetUnconstrainedFontSize(False)
            title_text.SetLineOffset(0)
            viewport = fit['renderer']
            fitted_height, fitted_ratio, fitted_title_ratio, fitted_pad = _constrained_box(
                scalar_bar,
                title=fit['title'],
                pad=fit['pad'],
                viewport=viewport,
                keep_height=fit['sized'],
            )
            scalar_bar.SetBarRatio(fitted_ratio)
            scalar_bar.SetTitleRatio(fitted_title_ratio)
            scalar_bar.SetTextPad(fitted_pad)
            if not fit['sized']:
                _set_box_height(scalar_bar, viewport, fitted_height)
            self._place_widget(fit['key'], scalar_bar)
            fit['applied'] = _layout_settings(scalar_bar)
            return

        fitted_width, fitted_height, fitted_ratio, offset, fitted_separation = _fitted_box(
            scalar_bar,
            vertical=fit['vertical'],
            title=fit['title'],
            label_text=scalar_bar.GetLabelTextProperty(),
            pad=fit['pad'],
            dpi=self._plotter.render_window.GetDPI(),
            viewport=fit['renderer'],
        )
        scalar_bar.SetWidth(fitted_width)
        scalar_bar.SetHeight(fitted_height)
        scalar_bar.SetBarRatio(fitted_ratio)
        scalar_bar.SetVerticalTitleSeparation(fitted_separation)
        if fit['vertical']:
            # A vertical bar is anchored at its right edge, where its labels are, so the
            # box grows away from the window edge rather than through it
            scalar_bar.SetPosition(position[0] - (fitted_width - width), position[1])
        title_text.SetLineOffset(offset)
        # The representation is what an interactive bar is drawn from, so it carries the
        # fitted box too, and dragging the widget then asks for a box of its own
        self._place_widget(fit['key'], scalar_bar)
        fit['applied'] = _layout_settings(scalar_bar)

    def _keep_fitted(
        self, title, scalar_bar, *, vertical, display_title, pad, sized, unconstrained
    ):
        """Refit a scalar bar's box whenever the window it is drawn in changes."""
        window = self._plotter.render_window
        fit = {
            'request': _layout_settings(scalar_bar),
            'vertical': vertical,
            'key': title,
            'title': display_title,
            'pad': pad,
            'sized': sized,
            'unconstrained': unconstrained,
            'renderer': self._plotter.renderer,
            'state': None,
            'applied': None,
            'observer': None,
        }
        self._scalar_bar_fits[title] = fit

        def refit(*_args):
            # The bar is looked up by the key the fit carries, so a renamed bar is
            # still the one this observer measures
            bar = self._scalar_bar_actors.get(fit['key'])
            if bar is None:
                # The observer outlived the bar it was measuring
                return
            settings = _layout_settings(bar)
            if fit['applied'] is not None and settings != fit['applied']:
                # The bar was sized or placed after it was fitted, so that is what it
                # asks for now and the fit is measured against it from here on
                fit['request'] = tuple(
                    now if now != before else asked
                    for now, before, asked in zip(
                        settings, fit['applied'], fit['request'], strict=True
                    )
                )
                fit['state'] = None
            # The text is measured in pixels while the box is a fraction of the viewport,
            # so the fit only holds while the viewport, the box it draws and the labels
            # it measures hold
            state = (
                tuple(fit['renderer'].GetSize()),
                self._plotter.render_window.GetDPI(),
                bool(bar.GetDrawFrame() or bar.GetDrawBackground()),
                tuple(_label_texts(bar)),
            )
            if state == fit['state']:
                return
            # A fit that fails part way is neither kept nor mistaken for a request
            fit['applied'] = None
            self._apply_fit(fit, bar)
            fit['state'] = state

        fit['observer'] = window.AddObserver(_vtk.vtkCommand.StartEvent, refit)
        refit()

    def _place_widget(self, title, scalar_bar):
        """Give a scalar bar's widget the place and size the layout gave the bar."""
        widget = self._scalar_bar_widgets.get(title)
        if widget is not None:
            # An interactive bar is drawn from its representation, which is built before
            # the bar is laid out and would otherwise put it back where it started
            rep = widget.GetRepresentation()
            rep.GetPositionCoordinate().SetValue(*scalar_bar.GetPosition())
            rep.GetPosition2Coordinate().SetValue(scalar_bar.GetWidth(), scalar_bar.GetHeight())

    def _stacked_neighbor(self, slot):
        """Return the scalar bar actor occupying the slot below this one."""
        lookup = self._plotter._scalar_bar_slot_lookup
        title = next(name for name, taken in lookup.items() if taken == slot - 1)
        return self._scalar_bar_actors[title]

    def _stacked_beside(self, scalar_bar, neighbor, *, gap, label_text, pad, dpi, constrained):
        """Return the position that clears a vertical scalar bar of the one beside it."""
        viewport_width = self._plotter.renderer.GetSize()[0]
        bar_width = scalar_bar.GetWidth() * viewport_width
        center = (neighbor.GetPosition()[0] + neighbor.GetWidth() / 2) * viewport_width
        if gap is None:
            neighbor_bar = neighbor.GetWidth() * viewport_width
            # The ramp is only part of the bar's box and sits at the edge facing away
            # from the neighbor, so the labels drawn past it reach into the gap
            if scalar_bar.GetOrientation():
                labels = (
                    scalar_bar.GetBarRatio() * bar_width
                    + _label_size(scalar_bar, label_text, dpi)[0]
                    - bar_width / 2
                )
            else:
                # A horizontal bar fills the width it is given, and the label on the end
                # facing the neighbor is centered on the ramp so half of it reaches past,
                # unless the box is laid out to pull the ramp in and hold it
                labels = bar_width / 2
                if not constrained:
                    labels += _label_size(scalar_bar, label_text, dpi)[0] / 2
            this_title = _title_width(
                scalar_bar.GetTitleTextProperty(), scalar_bar.GetTitle(), dpi
            )
            neighbor_title = _title_width(
                neighbor.GetTitleTextProperty(), neighbor.GetTitle(), dpi
            )
            # A title is centered on its bar, so each bar claims half of it
            reach = max(labels, 0 if _turned_title(scalar_bar) else this_title / 2)
            if scalar_bar.GetDrawFrame() or scalar_bar.GetDrawBackground():
                reach = max(reach, bar_width / 2)
            if _turned_title(neighbor):
                # The neighbor turned its title into the gap this text uses
                neighbor_reach = neighbor_bar / 2 + pad + _bar_title_height(neighbor, dpi)
            else:
                neighbor_reach = max(neighbor_bar / 2, neighbor_title / 2)
            gap = reach + neighbor_reach + 0.2 * bar_width
        return (center - gap - bar_width / 2) / viewport_width

    def _stacked_above(self, neighbor, *, gap, dpi):
        """Return the position that clears a horizontal scalar bar of the one below it."""
        viewport_height = self._plotter.renderer.GetSize()[1]
        if gap is None:
            bar_height = neighbor.GetHeight() * viewport_height
            # A horizontal bar draws its title and labels above its ramp, so it is the
            # neighbor below whose annotations reach up into the gap
            stack = (
                neighbor.GetBarRatio() * bar_height
                + _label_size(neighbor, neighbor.GetLabelTextProperty(), dpi)[1]
                + _title_separation(neighbor)
                + _bar_title_height(neighbor, dpi)
            )
            if neighbor.GetDrawFrame() or neighbor.GetDrawBackground():
                stack = max(stack, bar_height)
            gap = stack + 0.2 * bar_height
        return neighbor.GetPosition()[1] + gap / viewport_height

    def _remove_mapper_from_plotter(
        self,
        actor,
        *,
        reset_camera: bool = False,
        render: bool = False,
    ):  # numpydoc ignore=PR01,RT01
        """Remove an actor's mapper from the given plotter's ``_scalar_bar_mappers``.

        This ensures that when actors are removed, their corresponding
        scalar bars are removed.

        """
        try:
            mapper = actor.GetMapper()
        except AttributeError:
            return

        # NOTE: keys to list to prevent iterator changing during loop
        for name in list(self._scalar_bar_mappers):
            with contextlib.suppress(ValueError):
                self._scalar_bar_mappers[name].remove(mapper)

            if not self._scalar_bar_mappers[name]:
                slot = self._plotter._scalar_bar_slot_lookup.pop(name, None)
                if slot is not None:
                    self._scalar_bar_mappers.pop(name)
                    self._scalar_bar_ranges.pop(name)
                    self._plotter.remove_actor(
                        self._scalar_bar_actors.pop(name),
                        reset_camera=reset_camera,
                        render=render,
                    )
                    self._plotter._scalar_bar_slots.add(slot)
                    self._stop_fitting(name)

    def remove_scalar_bar(self, title=None, *, render: bool = True):
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
                msg = (
                    'Multiple scalar bars found.  Pick title of the'
                    f'scalar bar from one of the following:\n{titles}'
                )
                raise ValueError(msg)
            else:
                title = next(iter(self._scalar_bar_actors.keys()))

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

        self._stop_fitting(title)

    def __len__(self):
        """Return the number of scalar bar actors."""
        return len(self._scalar_bar_actors)

    def __getitem__(self, index):
        """Return a scalar bar actor."""
        return self._scalar_bar_actors[index]

    def keys(self):  # numpydoc ignore=RT01
        """Scalar bar keys."""
        return self._scalar_bar_actors.keys()

    def values(self):  # numpydoc ignore=RT01
        """Scalar bar values."""
        return self._scalar_bar_actors.values()

    def items(self):  # numpydoc ignore=RT01
        """Scalar bar items."""
        return self._scalar_bar_actors.items()

    def __contains__(self, key) -> bool:
        """Check if a title is a valid actors."""
        return key in self._scalar_bar_actors

    def update_title(
        self,
        old_title: str,
        new_title: str,
        *,
        render: bool = False,
    ) -> None:
        """Update the title of an existing scalar bar.

        .. versionadded:: 0.48.0

        Parameters
        ----------
        old_title : str
            Current title of the scalar bar to update.

        new_title : str
            New title for the scalar bar.

        render : bool, default: False
            Force a render after updating the title.

        Raises
        ------
        KeyError
            If no scalar bar with ``old_title`` exists.

        ValueError
            If a scalar bar with ``new_title`` already exists.

        Examples
        --------
        Update the title of a scalar bar.

        >>> import pyvista as pv
        >>> mesh = pv.Sphere()
        >>> mesh['Data'] = mesh.points[:, 2]
        >>> pl = pv.Plotter()
        >>> _ = pl.add_mesh(mesh, scalars='Data')
        >>> pl.scalar_bars.update_title('Data', 'Elevation')
        >>> pl.show()

        """
        if old_title not in self._scalar_bar_actors:
            msg = f'Scalar bar with title "{old_title}" not found.'
            raise KeyError(msg)
        if old_title != new_title and new_title in self._scalar_bar_actors:
            msg = f'Scalar bar with title "{new_title}" already exists.'
            raise ValueError(msg)

        if old_title != new_title:
            self._scalar_bar_actors[new_title] = self._scalar_bar_actors.pop(old_title)
            self._scalar_bar_ranges[new_title] = self._scalar_bar_ranges.pop(old_title)
            self._scalar_bar_mappers[new_title] = self._scalar_bar_mappers.pop(old_title)
            if old_title in self._scalar_bar_widgets:
                self._scalar_bar_widgets[new_title] = self._scalar_bar_widgets.pop(old_title)
            if old_title in self._scalar_bar_fits:
                fit = self._scalar_bar_fits.pop(old_title)
                fit['key'] = new_title
                fit['title'] = new_title
                # The title is what the box is fitted around, so it has to be measured again
                fit['state'] = None
                self._scalar_bar_fits[new_title] = fit
            slot = self._plotter._scalar_bar_slot_lookup.pop(old_title, None)
            if slot is not None:
                self._plotter._scalar_bar_slot_lookup[new_title] = slot

        self._scalar_bar_actors[new_title].SetTitle(new_title)

        if render:
            self._plotter.render()

    def add_scalar_bar(
        self,
        title='',
        *,
        mapper=None,
        lookup_table=None,
        cmap=None,
        clim=None,
        n_labels=5,
        tick_locations=None,
        italic: bool = False,
        bold: bool = False,
        title_font_size=None,
        title_pad=None,
        label_font_size=None,
        color=None,
        font_family=None,
        shadow: bool = False,
        width=None,
        height=None,
        position_x=None,
        position_y=None,
        vertical=None,
        stacking_gap: float | None = None,
        rotate_title: bool | None = None,
        interactive=None,
        fmt=None,
        use_opacity: bool = True,
        outline: bool = False,
        nan_annotation: bool = False,
        below_label=None,
        above_label=None,
        background_color=None,
        n_colors=None,
        fill: bool = False,
        render: bool = False,
        theme=None,
        unconstrained_font_size: bool = False,
        unique_bar: bool = False,
    ):
        """Create a scalar bar.

        Uses the ranges as set by the last input mesh or, alternatively, the
        ones set by ``clim``, ``mapper``, or ``lookup_table``.

        Parameters
        ----------
        title : str, default: ""
            Title of the scalar bar.  Default is rendered as an empty title.

        mapper : :vtk:`vtkMapper`, optional
            Mapper used for the scalar bar. Defaults to the last mapper created
            by the plotter if neither ``mapper``, ``lookup_table``, or
            ``cmap`` is provided. Raises ValueError if more than one of
            ``mapper``, ``lookup_table``, or ``cmap`` is provided.

        lookup_table : :vtk:`vtkLookupTable`, optional
            Lookup table used for the scalar bar. Raises ValueError if more
            than one of ``mapper``, ``lookup_table``, or ``cmap`` is provided.

            .. versionadded:: 0.49

        cmap : str | list, optional
            Colormap used for the scalar bar. Raises ValueError if more than
            one of ``mapper``, ``lookup_table``, or ``cmap`` is provided.

            .. versionadded:: 0.49

        clim : sequence[float], optional
            Two item range for the scalar bar. Only used if ``cmap`` is
            specified.

            .. versionadded:: 0.49

        n_labels : int, default: 5
            Number of labels to use for the scalar bar.

        tick_locations : sequence[float], optional
            Scalar values to label, instead of ``n_labels`` evenly spaced values.
            The label text comes from ``fmt``. Values outside the scalar range
            are not drawn.

            .. versionadded:: 0.50

        italic : bool, default: False
            Italicises title and bar labels.

        bold : bool, default: False
            Bolds title and bar labels.

        title_font_size : float, optional
            Sets the size of the title font.  Defaults to ``None`` and is sized
            according to :attr:`pyvista.plotting.themes.Theme.font`.

        title_pad : float, optional
            Space between the title and the tick labels, as a multiple of the
            title font size.  Defaults to ``None`` and is sized according to
            :attr:`pyvista.plotting.themes.Theme.colorbar_horizontal` or
            :attr:`pyvista.plotting.themes.Theme.colorbar_vertical`.  Has no
            effect when the font size is constrained, or on a box given a size
            of its own, which pads its title with whatever height it has spare.

            .. versionadded:: 0.50

        label_font_size : float, optional
            Sets the size of the label font.  Defaults to ``None`` and is sized
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
            The percentage (0 to 1) width of the viewport for the colorbar.  Giving
            a width, or a height, keeps a box drawn by ``fill`` or ``outline``
            exactly that size rather than growing it around the text, though
            only a height holds a horizontal box whose text is not
            ``unconstrained_font_size``.
            Default set by
            :attr:`pyvista.plotting.themes.Theme.colorbar_vertical` or
            :attr:`pyvista.plotting.themes.Theme.colorbar_horizontal`
            depending on the value of ``vertical``.

        height : float, optional
            The percentage (0 to 1) height of the viewport for the
            colorbar.  Default set by
            :attr:`pyvista.plotting.themes.Theme.colorbar_vertical` or
            :attr:`pyvista.plotting.themes.Theme.colorbar_horizontal`
            depending on the value of ``vertical``.

        position_x : float, optional
            The percentage (0 to 1) along the viewport's horizontal
            direction to place the bottom left corner of the colorbar.
            Default set by
            :attr:`pyvista.plotting.themes.Theme.colorbar_vertical` or
            :attr:`pyvista.plotting.themes.Theme.colorbar_horizontal`
            depending on the value of ``vertical``.

        position_y : float, optional
            The percentage (0 to 1) along the viewport's vertical
            direction to place the bottom left corner of the colorbar.
            Default set by
            :attr:`pyvista.plotting.themes.Theme.colorbar_vertical` or
            :attr:`pyvista.plotting.themes.Theme.colorbar_horizontal`
            depending on the value of ``vertical``.

        vertical : bool, optional
            Use vertical or horizontal scalar bar.  Default set by
            :attr:`pyvista.plotting.themes.Theme.colorbar_orientation`.

        stacking_gap : float, optional
            Distance between stacked scalar bars, as a fraction of the viewport.
            Defaults to ``None``, which spaces them as tightly as their titles
            and tick labels allow, and is taken from
            :attr:`pyvista.plotting.themes.Theme.colorbar_horizontal` or
            :attr:`pyvista.plotting.themes.Theme.colorbar_vertical`.  A value
            small enough to overlap is used as given.  Has no effect when the
            font size is constrained.

            .. versionadded:: 0.50

        rotate_title : bool, optional
            Turn the title alongside the bar instead of drawing it across the
            end, so that stacked bars sit closer together.  Defaults to ``None``
            and is taken from
            :attr:`pyvista.plotting.themes._VerticalColorbarConfig.rotate_title`.
            Applies to vertical bars only.  Requires VTK 9.4.0 or newer, and has
            no effect when the font size is constrained.

            .. versionadded:: 0.50

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
            The color used for the background in RGB format. Only drawn when
            ``fill`` is ``True``.

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
            Using custom labels will force this to be ``True``.  A box drawn
            around a horizontal bar sizes the text itself unless this is ``True``.

            .. versionadded:: 0.44.0

        unique_bar : bool, default: False
            Whether to create a scalar bar which is unique to the subplot.
            If ``True``, the scalar bar will be created with a unique key
            which is not shared with other subplots, even if the input title is the same.

            .. note::

                Scalar bars are managed by a dictionary with the title
                as the key. By default, if a scalar bar with the same title
                already exists, the scalar bar will be shared.
                If ``unique_bar`` is ``True``, the scalar bar will be created
                with a unique key which is the title suffixed with
                ``_UNIQUE_ID_{active_renderer_index}``, where ``active_renderer_index``
                is the index of the active renderer in the plotter.
                This allows for multiple scalar bars with the same title
                to be created across different subplots.

            .. versionadded:: 0.48.0

        Returns
        -------
        :vtk:`vtkScalarBarActor`
            Scalar bar actor.

        Notes
        -----
        Setting ``title_font_size``, or ``label_font_size`` disables automatic
        font sizing for both the title and label.  The title is drawn at its
        size, and the labels at theirs where they have the room to stay clear
        of each other and smaller where they do not; a label is free to run
        past the edge of the viewport.  A box drawn around a horizontal bar
        sizes the text itself, so the box is laid out to keep the text at the
        size asked for, or one size larger where two sizes measure the same
        height; a box given too small a height, or too narrow for its text,
        shrinks the text to fit.

        The ``mapper``, ``lookup_table``, and ``cmap`` parameters can be used
        to set a custom color map for the scalar bar; otherwise, the bar will
        default to the last mapper created by the plotter - for example, when
        a mesh with scalars is added by :func:`pyvista.Plotter.add_mesh`. See
        examples. Only one parameter can be used to set the color mapping, so
        ValueError will be raised if more than one of ``mapper``,
        ``lookup_table``, or ``cmap`` is provided.

        See Also
        --------
        :ref:`scalar_bar_example`

        Examples
        --------
        Add a custom interactive scalar bar that is horizontal, has an
        outline, and has a custom formatting.

        >>> import pyvista as pv
        >>> sphere = pv.Sphere()
        >>> sphere['Data'] = sphere.points[:, 2]
        >>> pl = pv.Plotter()
        >>> _ = pl.add_mesh(sphere, show_scalar_bar=False)
        >>> _ = pl.add_scalar_bar(
        ...     'Data',
        ...     interactive=True,
        ...     vertical=False,
        ...     title_font_size=35,
        ...     label_font_size=30,
        ...     outline=True,
        ...     fmt='%10.5f',
        ... )
        >>> pl.show()

        Add a custom scalar bar without (or before) plotting data using the
        ``cmap`` and ``clim`` parameters:

        >>> import pyvista as pv
        >>> pl = pv.Plotter()
        >>> _ = pl.add_scalar_bar('Height', cmap='viridis', clim=(-2, 2))
        >>> pl.show()

        Stack three scalar bars with titles of different lengths.  They are
        spaced so that nothing overlaps.

        >>> import pyvista as pv
        >>> sphere = pv.Sphere()
        >>> sphere['Data'] = sphere.points[:, 2]
        >>> titles = ['A bit long', 'Short', 'Super duper long']
        >>> pl = pv.Plotter()
        >>> pl.theme.colorbar_vertical.position_x = 0.75
        >>> _ = pl.add_mesh(sphere, show_scalar_bar=False)
        >>> for title in titles:
        ...     _ = pl.add_scalar_bar(
        ...         title,
        ...         vertical=True,
        ...         title_font_size=30,
        ...         label_font_size=30,
        ...         mapper=pl.mapper,
        ...     )
        >>> pl.show()

        Turn the titles alongside the bars to stack them closer together.

        >>> pl = pv.Plotter()
        >>> pl.theme.colorbar_vertical.position_x = 0.75
        >>> _ = pl.add_mesh(sphere, show_scalar_bar=False)
        >>> for title in titles:
        ...     _ = pl.add_scalar_bar(
        ...         title,
        ...         vertical=True,
        ...         rotate_title=True,
        ...         title_font_size=30,
        ...         label_font_size=30,
        ...         mapper=pl.mapper,
        ...     )
        >>> pl.show()

        Space the bars evenly instead, whatever their titles measure.

        >>> pl = pv.Plotter()
        >>> pl.theme.colorbar_vertical.position_x = 0.75
        >>> _ = pl.add_mesh(sphere, show_scalar_bar=False)
        >>> for title in titles:
        ...     _ = pl.add_scalar_bar(
        ...         title,
        ...         vertical=True,
        ...         stacking_gap=0.2,
        ...         title_font_size=30,
        ...         label_font_size=30,
        ...         mapper=pl.mapper,
        ...     )
        >>> pl.show()

        A box drawn around a bar grows to hold the title and the tick labels, as long
        as the bar was not given a size of its own.

        >>> pl = pv.Plotter()
        >>> _ = pl.add_mesh(sphere, show_scalar_bar=False)
        >>> _ = pl.add_scalar_bar(
        ...     'Elevation (m)',
        ...     vertical=True,
        ...     outline=True,
        ...     title_font_size=30,
        ...     label_font_size=30,
        ...     mapper=pl.mapper,
        ... )
        >>> pl.show()

        A horizontal bar's ramp and tick labels are laid out inside its box, and the
        box grows to hold the padding the title is given.

        >>> pl = pv.Plotter()
        >>> _ = pl.add_mesh(sphere, show_scalar_bar=False)
        >>> _ = pl.add_scalar_bar(
        ...     'Elevation (m)',
        ...     vertical=False,
        ...     outline=True,
        ...     title_font_size=30,
        ...     label_font_size=30,
        ...     mapper=pl.mapper,
        ... )
        >>> pl.show()

        """
        if theme is None:
            theme = pv.global_theme

        provided = {
            'mapper': mapper is not None,
            'lookup_table': lookup_table is not None,
            'cmap': cmap is not None,
        }
        if sum(provided.values()) != 1:
            msg = (
                "Exactly one of 'mapper', 'lookup_table', or 'cmap' must be provided. "
                f'Got: {", ".join(k for k, v in provided.items() if v) or "none"}.'
            )
            raise ValueError(msg)

        if cmap is None and clim is not None:
            msg = '`cmap` must be specified when `clim` is provided.'
            raise ValueError(msg)

        if cmap is not None:
            lookup_table = pv.LookupTable(cmap=cmap, scalar_range=clim)

        if lookup_table is not None:
            mapper = pv.DataSetMapper(theme=theme)
            mapper.lookup_table = lookup_table

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
        if vertical is None and theme.colorbar_orientation.lower() == 'vertical':
            vertical = True

        config = theme.colorbar_vertical if vertical else theme.colorbar_horizontal
        if stacking_gap is None:
            stacking_gap = config.stacking_gap
        if stacking_gap is not None:
            _validation.check_greater_than(stacking_gap, 0, strict=False, name='stacking_gap')

        if rotate_title is None:
            rotate_title = vertical and theme.colorbar_vertical.rotate_title
        if rotate_title and not vertical:
            msg = 'A rotated title is not supported for horizontal scalar bars.'
            raise ValueError(msg)
        if rotate_title and pv.vtk_version_info < (9, 4, 0):
            msg = 'A rotated scalar bar title requires VTK 9.4.0 or newer.'
            raise VTKVersionError(msg)

        if title_pad is None:
            title_pad = (
                theme.colorbar_vertical.title_pad
                if vertical
                else theme.colorbar_horizontal.title_pad
            )

        # Automatically choose size if not specified
        # A box is only free to grow around the text when its size was left open
        sized = width is not None or height is not None
        given_height = height is not None
        if width is None:
            width = theme.colorbar_vertical.width if vertical else theme.colorbar_horizontal.width
        if height is None:
            if vertical:
                height = theme.colorbar_vertical.height
            else:
                height = theme.colorbar_horizontal.height

        display_title = title
        if unique_bar:
            title = f'{title}_UNIQUE_ID_{self._plotter.renderers.active_index}'

        # Check that this data hasn't already been plotted
        if title in list(self._scalar_bar_ranges.keys()):
            stored = list(self._scalar_bar_ranges[title])
            newrng = mapper.scalar_range
            oldmappers = self._scalar_bar_mappers[title]
            _clim = [min(newrng[0], stored[0]), max(newrng[1], stored[1])]
            # Optimization: the old mappers already hold the stored range from the add that
            # set it, so they are only reset when this mesh widens it, once on the second add
            # to pin the first mapper's auto range the way the setter pinned the others', or
            # after ``update_scalar_bar_range`` moved them away from the stored range
            if _clim != stored or len(oldmappers) == 1 or title in self._resync_titles:
                for mh in oldmappers:
                    mh.scalar_range = _clim[0], _clim[1]
                self._resync_titles.discard(title)
            mapper.scalar_range = _clim[0], _clim[1]
            self._scalar_bar_mappers[title].append(mapper)
            self._scalar_bar_ranges[title] = _clim
            self._scalar_bar_actors[title].SetLookupTable(mapper.lookup_table)
            # Color bar already present and ready to be used so returning
            return None

        # Automatically choose location if not specified
        stacked_slot = 0
        if position_x is None or position_y is None:
            if not self._plotter._scalar_bar_slots:
                msg = f'Maximum number of color bars ({MAX_N_COLOR_BARS}) reached.'
                raise RuntimeError(msg)

            slot = min(self._plotter._scalar_bar_slots)
            self._plotter._scalar_bar_slots.remove(slot)
            self._plotter._scalar_bar_slot_lookup[title] = slot

            if position_x is None:
                if vertical:
                    position_x = theme.colorbar_vertical.position_x
                    position_x -= slot * (width + 0.2 * width)
                    stacked_slot = slot
                else:
                    position_x = theme.colorbar_horizontal.position_x

            if position_y is None:
                if vertical:
                    position_y = theme.colorbar_vertical.position_y
                else:
                    position_y = theme.colorbar_horizontal.position_y
                    position_y += slot * height
                    stacked_slot = slot

        # parse color
        color = Color(color, default_color=theme.font.color)

        # Create scalar bar
        scalar_bar = _vtk.vtkScalarBarActor()
        # self._scalar_bars.append(scalar_bar)

        if background_color is not None:
            scalar_bar.GetBackgroundProperty().SetColor(Color(background_color).float_rgb)
            if fill:
                scalar_bar.DrawBackgroundOn()

        lut = mapper.lookup_table
        scalar_bar.SetLookupTable(lut)
        if n_colors is None:
            # ensure the number of colors in the scalarbar's lookup table is at
            # least the number in the mapper
            n_colors = mapper.lookup_table.n_values

        scalar_bar.SetMaximumNumberOfColors(n_colors)

        if n_labels < 1:
            scalar_bar.SetDrawTickLabels(False)
        elif tick_locations is not None:
            labels = _validation.validate_arrayN(
                tick_locations, dtype_out=float, name='tick_locations'
            )
            scalar_bar.SetDrawTickLabels(True)
            scalar_bar.SetCustomLabels(convert_array(labels))
            scalar_bar.UseCustomLabelsOn()
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
        # Preset the justification the layout applies to the medial label after measuring it
        if vertical:
            label_text.SetJustificationToLeft()
            anno_text.SetJustificationToRight()
            anno_text.SetVerticalJustificationToCentered()
        else:
            label_text.SetJustificationToCentered()
            anno_text.SetJustificationToCentered()
            anno_text.SetVerticalJustificationToTop()
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

        scalar_bar.SetTitle(display_title)
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
                rep.SetOrientation(1)  # type: ignore[attr-defined] # 0 = Horizontal, 1 = Vertical
            else:
                # y position determined empirically
                y = -position_y / 2 - height - scalar_bar.GetPosition()[1]
                rep.GetPositionCoordinate().SetValue(width, y)  # type: ignore[attr-defined]
                rep.GetPosition2Coordinate().SetValue(height, width)  # type: ignore[attr-defined]
                rep.SetOrientation(0)  # type: ignore[attr-defined] # 0 = Horizontal, 1 = Vertical
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

        draws_box = scalar_bar.GetDrawFrame() or scalar_bar.GetDrawBackground()
        unconstrained = bool(scalar_bar.GetUnconstrainedFontSize())
        # A horizontal box is laid out by VTK, which sizes the text to the box and keeps
        # it inside, so the box is sized to the text instead; only its height pins it
        constrained = draws_box and not vertical and not unconstrained_font_size
        if constrained:
            sized = given_height
        fits_box = draws_box and not sized
        # A box is sized without the padding unless it is fitted around the title
        keeps_pad = fits_box or not draws_box
        pad = round(title_pad * title_text.GetFontSize()) if title_pad and keeps_pad else 0
        viewport_width, viewport_height = self._plotter.renderer.GetSize()
        dpi = self._plotter.render_window.GetDPI()

        keep_fitted = False
        if unconstrained:
            if rotate_title:
                scalar_bar.SetForceVerticalTitle(True)
                title_height = _title_height(title_text, display_title, dpi)
                bar_width = width * viewport_width
                title_text.SetLineOffset(-_rotated_title_offset(bar_width, title_height, pad))
            elif not sized or constrained or not vertical or not draws_box:
                # The box is free to grow and the bar has not been placed yet, or the
                # text has to be fitted to a bar that is not free to grow around it
                keep_fitted = True
            elif pad:
                title_text.SetLineOffset(-pad)

        # The gap between stacked bars is a fraction of the viewport but the annotations
        # are not, so the annotations set that gap once the viewport is small
        if stacked_slot and unconstrained:
            # Slots fill from the bottom up, so the one below this is taken
            neighbor = self._stacked_neighbor(stacked_slot)
            x, y = scalar_bar.GetPosition()
            # A bar is cleared along the axis its neighbor stacks on: past a vertical
            # neighbor, which fills the height it is given, and over a horizontal one,
            # which fills the width.  Bars drawn the same way stack as they always did,
            # and one turned across its neighbor takes the short way out instead
            if neighbor.GetOrientation():
                gap = stacking_gap * viewport_width if stacking_gap is not None else None
                scalar_bar.SetPosition(
                    self._stacked_beside(
                        scalar_bar,
                        neighbor,
                        gap=gap,
                        label_text=label_text,
                        pad=pad,
                        dpi=dpi,
                        constrained=constrained,
                    ),
                    y,
                )
            else:
                gap = stacking_gap * viewport_height if stacking_gap is not None else None
                scalar_bar.SetPosition(x, self._stacked_above(neighbor, gap=gap, dpi=dpi))

        self._place_widget(title, scalar_bar)

        if keep_fitted:
            # Fit once the bar is where it belongs, and keep it fitted as the window
            # it is measured against changes
            self._keep_fitted(
                title,
                scalar_bar,
                vertical=vertical,
                display_title=display_title,
                pad=pad,
                sized=sized,
                unconstrained=unconstrained_font_size,
            )

        # finally, add to the actor and return the scalar bar
        self._plotter.add_actor(scalar_bar, reset_camera=False, pickable=False, render=render)

        return scalar_bar
