"""Test charting functionality"""

import itertools
import platform

import numpy as np
import pytest

import pyvista
from pyvista import examples
from pyvista.plotting import charts, system_supports_plotting
from pyvista.utilities.misc import can_create_mpl_figure

skip_mac = pytest.mark.skipif(
    platform.system() == 'Darwin', reason='MacOS CI fails when downloading examples'
)

skip_no_plotting = pytest.mark.skipif(
    not system_supports_plotting(), reason="Test requires system to support plotting"
)

skip_no_mpl_figure = pytest.mark.skipif(
    not can_create_mpl_figure(), reason="Cannot create a figure using matplotlib"
)

# skip all tests if VTK<9.1.0
if pyvista.vtk_version_info < (9, 1):
    pytestmark = pytest.mark.skip

# TODO: add tests for new and compat functionality?


def vtk_array_to_tuple(arr):
    return tuple(arr.GetValue(i) for i in range(arr.GetNumberOfValues()))


def to_vtk_scientific(val):
    parts = val.split('e')
    sign, exp = parts[1][0], parts[1][1:]
    exp = exp.lstrip("0")  # Remove leading zeros of exponent
    return (
        parts[0] + "e" + sign + exp if exp != "" else parts[0]
    )  # Remove exponent altogether if it is 0


@pytest.fixture
def pl():
    p = pyvista.Plotter(window_size=(600, 600))
    p.background_color = 'w'
    return p


@pytest.fixture
def chart_2d():
    return pyvista.Chart2D()


@pytest.fixture
def chart_box():
    return pyvista.ChartBox([[1, 2, 3]])


@pytest.fixture
def chart_pie():
    return pyvista.ChartPie([1, 2, 3])


@pytest.fixture
def chart_mpl():
    import matplotlib.pyplot as plt

    f, ax = plt.subplots()
    ax.plot([0, 1, 2], [3, 1, 2])
    return pyvista.ChartMPL(f)


@pytest.fixture
def line_plot_2d(chart_2d):
    return chart_2d.line([0, 1, 2], [3, 1, 2])


@pytest.fixture
def scatter_plot_2d(chart_2d):
    return chart_2d.scatter([0, 1, 2], [3, 1, 2])


@pytest.fixture
def area_plot(chart_2d):
    return chart_2d.area([0, 1, 2], [2, 1, 3], [0, 2, 0])


@pytest.fixture
def bar_plot(chart_2d):
    return chart_2d.bar([0, 1, 2], [[2, 1, 3], [1, 2, 0]])


@pytest.fixture
def stack_plot(chart_2d):
    return chart_2d.stack([0, 1, 2], [[2, 1, 3], [1, 2, 0]])


@pytest.fixture
def box_plot(chart_box):
    return chart_box.plot


@pytest.fixture
def pie_plot(chart_pie):
    return chart_pie.plot


def test_pen():
    c_red, c_blue = (1.0, 0.0, 0.0, 1.0), (0.0, 0.0, 1.0, 1.0)
    w_thin, w_thick = 2, 10
    s_dash, s_dot, s_inv = "--", ":", "|"
    assert s_inv not in charts.Pen.LINE_STYLES, "New line styles added? Change this test."

    # Test constructor arguments
    pen = charts.Pen(color=c_red, width=w_thin, style=s_dash)
    assert pen.color == c_red
    assert np.isclose(pen.width, w_thin)
    assert pen.style == s_dash

    # Test properties
    pen.color = c_blue
    color = [0.0, 0.0, 0.0]
    pen.GetColorF(color)
    color.append(pen.GetOpacity() / 255)
    assert pen.color == c_blue
    assert np.allclose(color, c_blue)

    pen.width = w_thick
    assert np.isclose(pen.width, w_thick)
    assert np.isclose(pen.GetWidth(), w_thick)

    pen.style = s_dot
    assert pen.style == s_dot
    assert pen.GetLineType() == charts.Pen.LINE_STYLES[s_dot]["id"]
    with pytest.raises(ValueError):
        pen.style = s_inv


def test_wrapping():
    width = 5
    # Test wrapping of VTK Pen object
    vtkPen = pyvista._vtk.vtkPen()
    wrappedPen = charts.Pen(_wrap=vtkPen)
    assert wrappedPen.__this__ == vtkPen.__this__
    assert wrappedPen.width == vtkPen.GetWidth()
    wrappedPen.width = width
    assert wrappedPen.width == vtkPen.GetWidth()
    assert vtkPen.GetWidth() == width


@skip_mac
def test_brush():
    c_red, c_blue = (1.0, 0.0, 0.0, 1.0), (0.0, 0.0, 1.0, 1.0)
    t_masonry = examples.download_masonry_texture()
    t_puppy = examples.download_puppy_texture()

    # Test constructor arguments
    brush = charts.Brush(color=c_red, texture=t_masonry)
    assert brush.color == c_red
    assert np.allclose(brush.texture.to_array(), t_masonry.to_array())

    # Test properties
    brush.color = c_blue
    color = [0.0, 0.0, 0.0, 0.0]
    brush.GetColorF(color)
    assert brush.color == c_blue
    assert np.allclose(color, c_blue)

    brush.texture = t_puppy
    t = pyvista.Texture(brush.GetTexture())
    assert np.allclose(brush.texture.to_array(), t_puppy.to_array())
    assert np.allclose(t.to_array(), t_puppy.to_array())

    brush.texture_interpolate = False
    assert not brush.texture_interpolate

    NEAREST = 0x01
    assert brush.GetTextureProperties() & NEAREST

    brush.texture_repeat = True
    assert brush.texture_repeat
    REPEAT = 0x08
    assert brush.GetTextureProperties() & REPEAT


@skip_no_plotting
def test_axis(chart_2d):
    l = "Y axis"
    r_fix, r_auto = [2, 5], None
    m = 50
    tc = 10
    tlabels = ["Foo", "Blub", "Spam"]
    tlocs, tlocs_large = [1, 5.5, 8], [5.2, 340, 9999.999]
    ts = 5
    tlo = 10

    # Test constructor arguments
    axis = charts.Axis(label=l, range=r_fix, grid=True)
    assert axis.label == l
    assert np.allclose(axis.range, r_fix) and axis.behavior == "fixed"
    assert axis.grid

    # Test properties, using the y axis of a 2D chart
    chart_2d.line([0, 1], [1, 10])
    chart_2d.show()
    axis = chart_2d.y_axis

    axis.label = l
    assert axis.label == l
    assert axis.GetTitle() == l

    axis.label_visible = False
    assert not axis.label_visible
    assert not axis.GetTitleVisible()

    axis.range = r_auto
    assert axis.behavior == "auto"
    axis.range = r_fix
    r = [0.0, 0.0]
    axis.GetRange(r)
    assert np.allclose(axis.range, r_fix)
    assert np.allclose(r, r_fix)
    assert axis.behavior == "fixed"

    assert axis.GetBehavior() == charts.Axis.BEHAVIORS["fixed"]
    axis.behavior = "auto"
    assert axis.behavior == "auto"
    assert axis.GetBehavior() == charts.Axis.BEHAVIORS["auto"]
    with pytest.raises(ValueError):
        axis.behavior = "invalid"

    axis.margin = m
    assert axis.margin == m
    assert axis.GetMargins()[0] == m

    axis.log_scale = True  # Log scale can be enabled for the currently drawn plot
    chart_2d.show()  # We have to call show to update all chart properties (calls Update and Paint methods of chart/plot objects).
    assert axis.log_scale
    assert axis.GetLogScaleActive()
    axis.log_scale = False
    chart_2d.show()
    assert not axis.log_scale
    assert not axis.GetLogScaleActive()
    # Note: following lines cause "vtkMath::Jacobi: Error extracting eigenfunctions" warning to be printed.
    # Should be fixed on VTK side, but tricky without breaking stuff (see !8828 for reference).
    chart_2d.line([0, 1], [-10, 10])  # Plot for which log scale cannot be enabled
    axis.log_scale = True
    chart_2d.show()
    assert not axis.log_scale
    assert not axis.GetLogScaleActive()

    axis.grid = False
    assert not axis.grid
    assert not axis.GetGridVisible()

    axis.visible = False
    assert not axis.visible
    assert not axis.GetAxisVisible()
    axis.toggle()
    assert axis.visible
    assert axis.GetAxisVisible()

    tc0 = axis.tick_count
    axis.tick_count = tc
    assert axis.tick_count == tc
    assert axis.GetNumberOfTicks() == tc
    axis.tick_count = None
    assert axis.tick_count == tc0
    assert axis.GetNumberOfTicks() == tc0
    axis.tick_count = -1
    assert axis.tick_count == tc0
    assert axis.GetNumberOfTicks() == tc0

    tlocs0 = axis.tick_locations
    tlabels0 = axis.tick_labels
    axis.tick_locations = tlocs
    axis.tick_labels = tlabels
    assert np.allclose(axis.tick_locations, tlocs)
    assert np.allclose(axis.GetTickPositions(), tlocs)
    assert tuple(axis.tick_labels) == tuple(tlabels)
    assert vtk_array_to_tuple(axis.GetTickLabels()) == tuple(tlabels)
    axis.tick_labels = "2f"
    chart_2d.show()
    assert tuple(axis.tick_labels) == tuple(f"{loc:.2f}" for loc in tlocs)
    assert vtk_array_to_tuple(axis.GetTickLabels()) == tuple(f"{loc:.2f}" for loc in tlocs)
    assert axis.GetNotation() == charts.Axis.FIXED_NOTATION
    assert axis.GetPrecision() == 2
    axis.tick_labels = "4e"
    axis.tick_locations = tlocs_large  # Add some more variety to labels
    chart_2d.show()
    assert tuple(axis.tick_labels) == tuple(to_vtk_scientific(f"{loc:.4e}") for loc in tlocs_large)
    assert vtk_array_to_tuple(axis.GetTickLabels()) == tuple(
        to_vtk_scientific(f"{loc:.4e}") for loc in tlocs_large
    )
    assert axis.GetNotation() == charts.Axis.SCIENTIFIC_NOTATION
    assert axis.GetPrecision() == 4
    axis.tick_locations = None
    axis.tick_labels = None
    chart_2d.show()
    assert np.allclose(axis.tick_locations, tlocs0)
    assert np.allclose(axis.GetTickPositions(), tlocs0)
    assert tuple(axis.tick_labels) == tuple(tlabels0)
    assert vtk_array_to_tuple(axis.GetTickLabels()) == tuple(tlabels0)

    axis.tick_size = ts
    assert axis.tick_size == ts
    assert axis.GetTickLength() == ts

    axis.tick_labels_offset = tlo
    assert axis.tick_labels_offset == tlo
    assert axis.GetLabelOffset() == tlo

    axis.tick_labels_visible = False
    assert not axis.tick_labels_visible
    assert not axis.GetLabelsVisible()
    assert not axis.GetRangeLabelsVisible()

    axis.ticks_visible = False
    assert not axis.ticks_visible
    assert not axis.GetTicksVisible()


def test_axis_label_font_size(chart_2d):
    _ = chart_2d.line([0, 1, 2], [2, 1, 3])
    axis = chart_2d.x_axis
    font_size = 20

    axis.label_size = font_size
    assert axis.label_size == font_size
    assert axis.GetTitleProperties().GetFontSize() == font_size

    axis.tick_label_size = font_size
    assert axis.tick_label_size == font_size
    assert axis.GetLabelProperties().GetFontSize() == font_size


@skip_no_plotting
@skip_no_mpl_figure
@pytest.mark.parametrize("chart_f", ("chart_2d", "chart_box", "chart_pie", "chart_mpl"))
def test_chart_common(pl, chart_f, request):
    # Test the common chart functionalities
    chart = request.getfixturevalue(chart_f)
    title = "Chart title"
    c_red, c_blue = (1.0, 0.0, 0.0, 1.0), (0.0, 0.0, 1.0, 1.0)
    bw = 10
    bs = "--"

    # Check scene and renderer properties
    assert chart._scene is None
    assert chart._renderer is None
    pl.add_chart(chart)
    assert chart._scene is pl.renderer._charts._scene
    assert chart._renderer is pl.renderer and chart._renderer is pl.renderer._charts._renderer

    with pytest.raises((AssertionError, ValueError)):
        chart.size = (-1, 1)
    with pytest.raises((AssertionError, ValueError)):
        chart.loc = (-1, 1)
    try:  # Try block for now as not all charts support a custom size and loc
        chart.size = (0.5, 0.5)
        chart.loc = (0.25, 0.25)
        assert chart.size == (0.5, 0.5)
        assert chart.loc == (0.25, 0.25)
    except ValueError:
        pass

    # Check geometry and resizing
    w, h = pl.window_size
    chart._render_event()
    assert chart._geometry == (
        chart.loc[0] * w,
        chart.loc[1] * h,
        chart.size[0] * w,
        chart.size[1] * h,
    )
    w, h = pl.window_size = [200, 200]
    chart._render_event()
    assert chart._geometry == (
        chart.loc[0] * w,
        chart.loc[1] * h,
        chart.size[0] * w,
        chart.size[1] * h,
    )

    # Check is_within
    assert chart._is_within(
        ((chart.loc[0] + chart.size[0] / 2) * w, (chart.loc[1] + chart.size[1] / 2) * h)
    )
    assert not chart._is_within(((chart.loc[0] + chart.size[0] / 2) * w, chart.loc[1] * h - 5))
    assert not chart._is_within((chart.loc[0] * w - 5, (chart.loc[1] + chart.size[1] / 2) * h))
    assert not chart._is_within((chart.loc[0] * w - 5, chart.loc[1] * h - 5))

    chart.border_color = c_red
    assert chart.border_color == c_red
    chart.border_width = bw
    assert chart.border_width == bw
    chart.border_style = bs
    assert chart.border_style == bs
    chart.background_color = c_blue
    assert chart.background_color == c_blue

    # Check remaining properties and methods
    chart.visible = False
    assert not chart.visible
    assert not chart.GetVisible()
    chart.toggle()
    assert chart.visible
    assert chart.GetVisible()
    chart.title = title
    assert chart.title == title
    chart.legend_visible = False
    assert not chart.legend_visible


@pytest.mark.parametrize(
    "plot_f",
    (
        "line_plot_2d",
        "scatter_plot_2d",
        "area_plot",
        "bar_plot",
        "stack_plot",
        "box_plot",
        "pie_plot",
    ),
)
def test_plot_common(plot_f, request):
    # Test the common plot functionalities
    plot = request.getfixturevalue(plot_f)
    c = (1.0, 0.0, 1.0, 1.0)
    w = 5
    s = "-."
    l = "Label"

    plot.color = c
    assert plot.color == c
    assert plot.brush.color == c

    if hasattr(plot, "GetPen"):
        assert plot.pen.__this__ == plot.GetPen().__this__
    if hasattr(plot, "GetBrush"):
        assert plot.brush.__this__ == plot.GetBrush().__this__

    plot.line_width = w
    assert plot.pen.width == w
    plot.line_style = s
    assert plot.pen.style == s

    plot.label = l
    assert plot.label == l
    assert plot.GetLabel() == l

    plot.visible = False
    assert not plot.visible
    assert not plot.GetVisible()
    plot.toggle()
    assert plot.visible
    assert plot.GetVisible()


@pytest.mark.parametrize("plot_f", ("bar_plot", "stack_plot", "box_plot", "pie_plot"))
def test_multicomp_plot_common(plot_f, request):
    # Test the common multicomp plot functionalities
    plot = request.getfixturevalue(plot_f)
    cs = "spectrum"
    cs_colors = [
        (0.0, 0.0, 0.0, 1.0),
        (0.8941176470588236, 0.10196078431372549, 0.10980392156862745, 1.0),
        (0.21568627450980393, 0.49411764705882355, 0.7215686274509804, 1.0),
        (0.30196078431372547, 0.6862745098039216, 0.2901960784313726, 1.0),
        (0.596078431372549, 0.3058823529411765, 0.6392156862745098, 1.0),
        (1.0, 0.4980392156862745, 0.0, 1.0),
        (0.6509803921568628, 0.33725490196078434, 0.1568627450980392, 1.0),
    ]
    colors = [(1.0, 0.0, 1.0, 1.0), (0.0, 1.0, 1.0, 1.0), (1.0, 1.0, 0.0, 1.0)]
    labels = ["Foo", "Spam", "Bla"]

    plot.color_scheme = cs
    assert plot.color_scheme == cs
    assert plot._color_series.GetColorScheme() == plot.COLOR_SCHEMES[cs]["id"]
    assert all(pc == cs for pc, cs in zip(plot.colors, cs_colors))
    series_colors = [
        pyvista.Color(plot._color_series.GetColor(i)).float_rgba for i in range(len(cs_colors))
    ]
    assert np.allclose(series_colors, cs_colors)
    lookup_colors = [plot._lookup_table.GetTableValue(i) for i in range(len(cs_colors))]
    assert np.allclose(lookup_colors, cs_colors)
    assert plot.brush.color == cs_colors[0]

    plot.colors = None
    assert plot.color_scheme == plot.DEFAULT_COLOR_SCHEME
    plot.colors = cs
    assert plot.color_scheme == cs
    plot.colors = colors
    assert all(pc == c for pc, c in zip(plot.colors, colors))
    series_colors = [
        pyvista.Color(plot._color_series.GetColor(i)).float_rgba for i in range(len(colors))
    ]
    assert np.allclose(series_colors, colors)
    lookup_colors = [plot._lookup_table.GetTableValue(i) for i in range(len(colors))]
    assert np.allclose(lookup_colors, colors)
    assert plot.brush.color == colors[0]

    plot.color = colors[1]
    assert plot.color == colors[1]
    assert len(plot.colors) == 1 and plot.colors[0] == colors[1]
    assert plot.brush.color == colors[1]

    plot.labels = labels
    assert tuple(plot.labels) == tuple(labels)
    assert plot.label == labels[0]
    plot.labels = None
    assert plot.labels == []
    assert plot.label == ""

    plot.label = labels[1]
    assert tuple(plot.labels) == (labels[1],)
    assert plot.label == labels[1]
    plot.label = None
    assert plot.labels == []
    assert plot.label == ""


def test_lineplot2d(line_plot_2d):
    x = [-2, -1, 0, 1, 2]
    y = [4, 1, 0, -1, -4]
    c = (1.0, 0.0, 1.0, 1.0)
    w = 5
    s = "-."
    l = "Line"

    # Test constructor
    plot = charts.LinePlot2D(x, y, c, w, s, l)
    assert np.allclose(plot.x, x)
    assert np.allclose(plot.y, y)
    assert plot.color == c
    assert plot.line_width == w
    assert plot.line_style == s
    assert plot.label == l

    # Test remaining properties
    line_plot_2d.update(x, y)
    assert np.allclose(line_plot_2d.x, x)
    assert np.allclose(line_plot_2d.y, y)


def test_scatterplot2d(scatter_plot_2d):
    x = [-2, -1, 0, 1, 2]
    y = [4, 1, 0, -1, -4]
    c = (1.0, 0.0, 1.0, 1.0)
    sz = 5
    st, st_inv = "o", "^"
    l = "Scatter"
    assert (
        st_inv not in charts.ScatterPlot2D.MARKER_STYLES
    ), "New marker styles added? Change this test."

    # Test constructor
    plot = charts.ScatterPlot2D(x, y, c, sz, st, l)
    assert np.allclose(plot.x, x)
    assert np.allclose(plot.y, y)
    assert plot.color == c
    assert plot.marker_size == sz
    assert plot.marker_style == st
    assert plot.label == l

    # Test remaining properties
    scatter_plot_2d.update(x, y)
    assert np.allclose(scatter_plot_2d.x, x)
    assert np.allclose(scatter_plot_2d.y, y)

    scatter_plot_2d.marker_size = sz
    assert scatter_plot_2d.marker_size == sz
    assert scatter_plot_2d.GetMarkerSize() == sz

    scatter_plot_2d.marker_style = None
    assert scatter_plot_2d.marker_style == ""
    scatter_plot_2d.marker_style = st
    assert scatter_plot_2d.marker_style == st
    assert scatter_plot_2d.GetMarkerStyle() == scatter_plot_2d.MARKER_STYLES[st]["id"]
    with pytest.raises(ValueError):
        scatter_plot_2d.marker_style = st_inv


def test_areaplot(area_plot):
    x = [-2, -1, 0, 1, 2]
    y1 = [4, 1, 0, -1, -4]
    y2 = [-4, -2, 0, 2, 4]
    c = (1.0, 0.0, 1.0, 1.0)
    l = "Line"

    # Test constructor
    plot = charts.AreaPlot(x, y1, y2, c, l)
    assert np.allclose(plot.x, x)
    assert np.allclose(plot.y1, y1)
    assert np.allclose(plot.y2, y2)
    assert plot.color == c
    assert plot.label == l

    # Test remaining properties
    area_plot.update(x, y1, y2)
    assert np.allclose(area_plot.x, x)
    assert np.allclose(area_plot.y1, y1)
    assert np.allclose(area_plot.y2, y2)


def test_barplot(bar_plot):
    x = [0, 1, 2]
    y = [[1, 2, 3], [2, 1, 0], [1, 1, 1]]
    c = [(1.0, 0.0, 1.0, 1.0), (1.0, 1.0, 0.0, 1.0), (0.0, 1.0, 1.0, 1.0)]
    ori, ori_inv = "H", "I"
    l = ["Foo", "Spam", "Bla"]
    assert ori_inv not in charts.BarPlot.ORIENTATIONS, "New orientations added? Change this test."

    # Test multi comp constructor
    plot = charts.BarPlot(x, y, c, ori, l)
    assert np.allclose(plot.x, x)
    assert np.allclose(plot.y, y)
    assert all(pc == ci for pc, ci in zip(plot.colors, c))
    assert plot.orientation == ori
    assert plot.labels == l

    # Test single comp constructor
    plot = charts.BarPlot(x, y[0], c[0], ori, l[0])
    assert np.allclose(plot.x, x)
    assert np.allclose(plot.y, y[0])
    assert plot.color == c[0]
    assert plot.orientation == ori
    assert plot.label == l[0]

    # Test multi and single comp constructors with inconsistent arguments
    with pytest.raises(ValueError):
        charts.BarPlot(x, y, c[0], ori, l)
    # charts.BarPlot(x, y, c, off, ori, l[0])  # This one is valid
    with pytest.raises(ValueError):
        charts.BarPlot(x, y[0], c, ori, l[0])
    with pytest.raises(ValueError):
        charts.BarPlot(x, y[0], c[0], ori, l)

    # Test remaining properties
    bar_plot.update(x, y)
    assert np.allclose(bar_plot.x, x)
    assert np.allclose(bar_plot.y, y)

    bar_plot.orientation = ori
    assert bar_plot.orientation == ori
    assert bar_plot.GetOrientation() == bar_plot.ORIENTATIONS[ori]
    with pytest.raises(ValueError):
        bar_plot.orientation = ori_inv


def test_stackplot(stack_plot):
    x = [0, 1, 2]
    ys = [[1, 2, 3], [2, 1, 0], [1, 1, 1]]
    c = [(1.0, 0.0, 1.0, 1.0), (1.0, 1.0, 0.0, 1.0), (0.0, 1.0, 1.0, 1.0)]
    l = ["Foo", "Spam", "Bla"]

    # Test multi comp constructor
    plot = charts.StackPlot(x, ys, c, l)
    assert np.allclose(plot.x, x)
    assert np.allclose(plot.ys, ys)
    assert all(pc == ci for pc, ci in zip(plot.colors, c))
    assert plot.labels == l

    # Test single comp constructor
    plot = charts.StackPlot(x, ys[0], c[0], l[0])
    assert np.allclose(plot.x, x)
    assert np.allclose(plot.ys, ys[0])
    assert plot.color == c[0]
    assert plot.label == l[0]

    # Test multi and single comp constructors with inconsistent arguments
    with pytest.raises(ValueError):
        charts.StackPlot(x, ys, c[0], l)
    # charts.StackPlot(x, ys, c, l[0])  # This one is valid
    with pytest.raises(ValueError):
        charts.StackPlot(x, ys[0], c, l[0])
    with pytest.raises(ValueError):
        charts.StackPlot(x, ys[0], c[0], l)

    # Test remaining properties
    stack_plot.update(x, ys)
    assert np.allclose(stack_plot.x, x)
    assert np.allclose(stack_plot.ys, ys)


@skip_no_plotting
def test_chart_2d(pl, chart_2d):
    size = (0.5, 0.5)
    loc = (0.25, 0.25)
    lx = "X label"
    ly = "Y label"
    rx = [0, 5]
    ry = [0, 1]
    x = np.arange(11) - 5
    y = x**2
    ys = [np.sin(x), np.cos(x), np.tanh(x)]
    col = (1.0, 0.0, 1.0, 1.0)
    cs = "citrus"
    sz = 5
    ms = "d"
    w = 10
    ls = "-."
    ori = "V"

    # Test constructor
    chart = pyvista.Chart2D(size, loc, lx, ly, False)
    assert chart.size == size
    assert chart.loc == loc
    assert chart.x_label == lx
    assert chart.y_label == ly
    assert not chart.grid

    # Test geometry and resizing
    pl.add_chart(chart)
    r_w, r_h = chart._renderer.GetSize()
    pl.show(auto_close=False)
    assert np.allclose(chart._geometry, (loc[0] * r_w, loc[1] * r_h, size[0] * r_w, size[1] * r_h))
    pl.window_size = (int(pl.window_size[0] / 2), int(pl.window_size[1] / 2))
    pl.show()  # This will also call chart._resize
    assert np.allclose(
        chart._geometry, (loc[0] * r_w / 2, loc[1] * r_h / 2, size[0] * r_w / 2, size[1] * r_h / 2)
    )

    # Test parse_format
    hex_colors = ["#fa09b6", "0xa53a8d", "#b02239f0", "0xcee6927f"]
    colors = itertools.chain(pyvista.hexcolors, pyvista.colors.color_synonyms, [*hex_colors, ""])
    for m in charts.ScatterPlot2D.MARKER_STYLES:
        for l in charts.Pen.LINE_STYLES:
            for c in colors:
                cp = "b" if c == "" else c
                assert (m, l, cp) == chart_2d._parse_format(m + l + c)
                assert (m, l, cp) == chart_2d._parse_format(m + c + l)
                assert (m, l, cp) == chart_2d._parse_format(l + m + c)
                assert (m, l, cp) == chart_2d._parse_format(l + c + m)
                assert (m, l, cp) == chart_2d._parse_format(c + m + l)
                assert (m, l, cp) == chart_2d._parse_format(c + l + m)

    # Test plotting methods
    s, l = chart_2d.plot(x, y, "")
    assert s is None and l is None
    assert len([*chart_2d.plots()]) == 0
    s, l = chart_2d.plot(y, "-")
    assert s is None and l is not None
    assert l in chart_2d.plots("line")
    chart_2d.remove_plot(l)
    assert len([*chart_2d.plots()]) == 0
    s, l = chart_2d.plot(y, "x")
    assert s is not None and l is None
    assert s in chart_2d.plots("scatter")
    chart_2d.clear("scatter")
    assert len([*chart_2d.plots()]) == 0
    s, l = chart_2d.plot(x, y, "x-")
    assert s is not None and l is not None
    assert s in chart_2d.plots("scatter") and l in chart_2d.plots("line")
    chart_2d.plot(x, y, "x-")  # Check clearing of multiple plots (of the same type)
    chart_2d.clear()
    assert len([*chart_2d.plots()]) == 0

    s = chart_2d.scatter(x, y, col, sz, ms, lx)
    assert np.allclose(s.x, x)
    assert np.allclose(s.y, y)
    assert s.color == col
    assert s.marker_size == sz
    assert s.marker_style == ms
    assert s.label == lx
    assert s in chart_2d.plots("scatter")
    assert chart_2d.GetPlotIndex(s) >= 0

    l = chart_2d.line(x, y, col, w, ls, lx)
    assert np.allclose(l.x, x)
    assert np.allclose(l.y, y)
    assert l.color == col
    assert l.line_width == w
    assert l.line_style == ls
    assert l.label == lx
    assert l in chart_2d.plots("line")
    assert chart_2d.GetPlotIndex(l) >= 0

    a = chart_2d.area(x, -y, y, col, lx)
    assert np.allclose(a.x, x)
    assert np.allclose(a.y1, -y)
    assert np.allclose(a.y2, y)
    assert a.color == col
    assert a.label == lx
    assert a in chart_2d.plots("area")
    assert chart_2d.GetPlotIndex(a) >= 0

    b = chart_2d.bar(x, -y, col, ori, lx)
    assert np.allclose(b.x, x)
    assert np.allclose(b.y, -y)
    assert b.color == col
    assert b.orientation == ori
    assert b.label == lx
    assert b in chart_2d.plots("bar")
    assert chart_2d.GetPlotIndex(b) >= 0

    s = chart_2d.stack(x, ys, cs, [lx, ly])
    assert np.allclose(s.x, x)
    assert np.allclose(s.ys, ys)
    assert s.color_scheme == cs
    assert tuple(s.labels) == (lx, ly)
    assert s in chart_2d.plots("stack")
    assert chart_2d.GetPlotIndex(s) >= 0

    inv_type = "blub"
    with pytest.raises(KeyError):
        next(chart_2d.plots(inv_type))
    with pytest.raises(KeyError):
        chart_2d.clear(inv_type)
    assert len([*chart_2d.plots()]) == 5
    chart_2d.clear()
    assert len([*chart_2d.plots()]) == 0
    with pytest.raises(ValueError):
        chart_2d.remove_plot(s)

    # Check remaining properties
    assert chart_2d.x_axis.__this__ == chart_2d.GetAxis(charts.Axis.BOTTOM).__this__
    assert chart_2d.y_axis.__this__ == chart_2d.GetAxis(charts.Axis.LEFT).__this__

    chart_2d.x_label = lx
    assert chart_2d.x_label == lx
    assert chart_2d.x_axis.label == lx
    chart_2d.y_label = ly
    assert chart_2d.y_label == ly
    assert chart_2d.y_axis.label == ly

    chart_2d.x_range = rx
    assert np.allclose(chart_2d.x_range, rx)
    assert np.allclose(chart_2d.x_axis.range, rx)
    chart_2d.y_range = ry
    assert np.allclose(chart_2d.y_range, ry)
    assert np.allclose(chart_2d.y_axis.range, ry)

    chart_2d.grid = True
    assert chart_2d.grid
    assert chart_2d.x_axis.grid and chart_2d.y_axis.grid

    chart_2d.hide_axes()
    for axis in (chart_2d.x_axis, chart_2d.y_axis):
        assert not (
            axis.visible
            or axis.label_visible
            or axis.ticks_visible
            or axis.tick_labels_visible
            or axis.grid
        )


@skip_no_plotting
def test_chart_box(pl, chart_box, box_plot):
    data = [[0, 1, 1, 1, 2, 2, 3, 4, 4, 5, 5, 5, 6]]
    stats = [np.quantile(d, [0.0, 0.25, 0.5, 0.75, 1.0]) for d in data]
    cs = "wild_flower"
    ls = ["Datalabel"]

    # Test constructor
    chart = pyvista.ChartBox(data, cs, ls)
    assert np.allclose(chart.plot.data, data)
    assert chart.plot.color_scheme == cs
    assert tuple(chart.plot.labels) == tuple(ls)

    # Test geometry and resizing
    pl.add_chart(chart)
    r_w, r_h = chart._renderer.GetSize()
    pl.show(auto_close=False)
    assert np.allclose(chart._geometry, (0, 0, r_w, r_h))
    pl.window_size = (int(pl.window_size[0] / 2), int(pl.window_size[1] / 2))
    pl.show(auto_close=False)  # This will also call chart._resize
    assert np.allclose(chart._geometry, (0, 0, r_w / 2, r_h / 2))

    # Test remaining properties
    assert chart_box.loc == (0, 0)
    assert chart_box.size == (1, 1)
    assert chart_box.plot.__this__ == chart_box.GetPlot(0).__this__

    box_plot.update(data)
    assert np.allclose(box_plot.data, data)
    assert np.allclose(box_plot.stats, stats)


@skip_no_plotting
def test_chart_pie(pl, chart_pie, pie_plot):
    data = [3, 4, 5]
    cs = "wild_flower"
    ls = ["Tic", "Tac", "Toe"]

    # Test constructor
    chart = pyvista.ChartPie(data, cs, ls)
    assert np.allclose(chart.plot.data, data)
    assert chart.plot.color_scheme == cs
    assert tuple(chart.plot.labels) == tuple(ls)

    # Test geometry and resizing
    pl.add_chart(chart)
    r_w, r_h = chart._renderer.GetSize()
    pl.show(auto_close=False)
    assert np.allclose(chart._geometry, (0, 0, r_w, r_h))
    pl.window_size = (int(pl.window_size[0] / 2), int(pl.window_size[1] / 2))
    pl.show(auto_close=False)  # This will also call chart._resize
    assert np.allclose(chart._geometry, (0, 0, r_w / 2, r_h / 2))

    # Test remaining properties
    assert chart_pie.loc == (0, 0)
    assert chart_pie.size == (1, 1)
    assert chart_pie.plot.__this__ == chart_pie.GetPlot(0).__this__

    pie_plot.update(data)
    assert np.allclose(pie_plot.data, data)


@skip_no_plotting
@skip_no_mpl_figure
def test_chart_mpl(pl, chart_mpl):
    import matplotlib.pyplot as plt

    size = (0.5, 0.5)
    loc = (0.25, 0.25)

    # Test constructor
    f, ax = plt.subplots()
    chart = pyvista.ChartMPL(f, size, loc)
    assert chart.size == size
    assert chart.loc == loc

    # Test geometry and resizing
    pl.add_chart(chart)
    r_w, r_h = chart._renderer.GetSize()
    pl.show(auto_close=False)
    assert np.allclose(chart._geometry, (loc[0] * r_w, loc[1] * r_h, size[0] * r_w, size[1] * r_h))
    assert np.allclose(chart.position, (loc[0] * r_w, loc[1] * r_h))
    assert np.allclose(chart._canvas.get_width_height(), (size[0] * r_w, size[1] * r_h))
    pl.window_size = (int(pl.window_size[0] / 2), int(pl.window_size[1] / 2))
    pl.show(auto_close=False)  # This will also call chart._resize
    assert np.allclose(
        chart._geometry, (loc[0] * r_w / 2, loc[1] * r_h / 2, size[0] * r_w / 2, size[1] * r_h / 2)
    )
    assert np.allclose(chart.position, (loc[0] * r_w / 2, loc[1] * r_h / 2))
    assert np.allclose(chart._canvas.get_width_height(), (size[0] * r_w / 2, size[1] * r_h / 2))

    # test set position throw
    with pytest.raises(ValueError, match="must be length 2"):
        chart.position = (1, 2, 3)


@skip_no_plotting
def test_charts(pl):
    win_size = pl.window_size
    top_left = pyvista.Chart2D(size=(0.5, 0.5), loc=(0, 0.5))
    bottom_right = pyvista.Chart2D(size=(0.5, 0.5), loc=(0.5, 0))

    # Test add_chart
    pl.add_chart(top_left)
    assert pl.renderers[0].__this__ == top_left._renderer.__this__
    assert pl.renderers[0]._charts._scene.__this__ == top_left._scene.__this__
    pl.add_chart(bottom_right)
    assert len(pl.renderers[0]._charts) == 2

    # Test toggle_interaction
    pl.show(auto_close=False)  # We need to plot once to let the charts compute their true geometry
    assert not top_left.GetInteractive()
    assert not bottom_right.GetInteractive()
    assert (
        pl.renderers[0]._charts.toggle_interaction((0.75 * win_size[0], 0.25 * win_size[1]))
        is bottom_right._scene
    )
    assert not top_left.GetInteractive()
    assert bottom_right.GetInteractive()
    assert pl.renderers[0]._charts.toggle_interaction((0, 0)) is None
    assert not top_left.GetInteractive()
    assert not bottom_right.GetInteractive()

    # Test remove_chart
    pl.remove_chart(1)
    assert len(pl.renderers[0]._charts) == 1
    assert pl.renderers[0]._charts[0] == top_left
    assert top_left in pl.renderers[0]._charts
    pl.remove_chart(top_left)
    assert len(pl.renderers[0]._charts) == 0

    # Test deep_clean
    pl.add_chart(top_left, bottom_right)
    pl.deep_clean()
    assert len(pl.renderers[0]._charts) == 0
    assert pl.renderers[0]._charts._scene is None


@skip_no_plotting
def test_iren_context_style(pl):
    chart = pyvista.Chart2D(size=(0.5, 0.5), loc=(0.5, 0.5))
    win_size = pl.window_size
    pl.add_chart(chart)
    pl.show(auto_close=False)  # We need to plot once to let the charts compute their true geometry
    style = pl.iren._style
    style_class = pl.iren._style_class

    # Simulate double left click on the chart:
    pl.iren._mouse_left_button_click(int(0.75 * win_size[0]), int(0.75 * win_size[1]), count=2)
    assert chart.GetInteractive()
    assert pl.iren._style == "Context"
    assert pl.iren._style_class == pl.iren._context_style
    assert pl.iren._context_style.GetScene().__this__ == chart._scene.__this__

    # Simulate right click outside the chart:
    pl.iren._mouse_left_button_click(0, 0, count=2)
    assert not chart.GetInteractive()
    assert pl.iren._style == style
    assert pl.iren._style_class == style_class
    assert pl.iren._context_style.GetScene() is None


@skip_mac
def test_get_background_texture(chart_2d):
    t_puppy = examples.download_puppy_texture()
    chart_2d.background_texture = t_puppy
    assert chart_2d.background_texture == t_puppy
