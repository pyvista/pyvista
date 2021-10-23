"""Test charting functionality"""

import platform

import pytest
import numpy as np
import matplotlib.pyplot as plt
import itertools

import pyvista
from pyvista.plotting import charts
from pyvista import examples

skip_mac = pytest.mark.skipif(platform.system() == 'Darwin',
                              reason='MacOS CI fails when downloading examples')


def vtk_array_to_tuple(arr):
    return tuple(arr.GetValue(i) for i in range(arr.GetNumberOfValues()))


def to_vtk_scientific(val):
    parts = val.split('e')
    sign, exp = parts[1][0], parts[1][1:]
    exp = exp.lstrip("0")  # Remove leading zeros of exponent
    return parts[0] + "e" + sign + exp if exp != "" else parts[0]  # Remove exponent altogether if it is 0


@pytest.fixture
def pl():
    p = pyvista.Plotter(window_size=(600, 600))
    p.background_color = 'w'
    return p


@pytest.fixture
def chart2D():
    chart = pyvista.Chart2D()
    return chart


@pytest.fixture
def chartBox():
    return pyvista.ChartBox([[1, 2, 3]])


@pytest.fixture
def chartPie():
    return pyvista.ChartPie([1, 2, 3])


@pytest.fixture
def chartMPL():
    f, ax = plt.subplots()
    ax.plot([0, 1, 2], [3, 1, 2])
    return pyvista.ChartMPL(f)


@pytest.fixture
def linePlot2D(chart2D):
    plot = chart2D.line([0, 1, 2], [3, 1, 2])
    return plot


@pytest.fixture
def scatterPlot2D(chart2D):
    plot = chart2D.scatter([0, 1, 2], [3, 1, 2])
    return plot


@pytest.fixture
def areaPlot(chart2D):
    plot = chart2D.area([0, 1, 2], [2, 1, 3], [0, 2, 0])
    return plot


@pytest.fixture
def barPlot(chart2D):
    plot = chart2D.bar([0, 1, 2], [[2, 1, 3], [1, 2, 0]])
    return plot


@pytest.fixture
def stackPlot(chart2D):
    plot = chart2D.stack([0, 1, 2], [[2, 1, 3], [1, 2, 0]])
    return plot


@pytest.fixture
def boxPlot(chartBox):
    return chartBox.plot


@pytest.fixture
def piePlot(chartPie):
    return chartPie.plot


def test_pen():
    c_red, c_blue = (1, 0, 0, 1), (0, 0, 1, 1)
    w_thin, w_thick = 2, 10
    s_dash, s_dot, s_inv = "--", ":", "|"
    assert s_inv not in charts.Pen.LINE_STYLES, "New line styles added? Change this test."

    # Test constructor arguments
    pen = charts.Pen(color=c_red, width=w_thin, style=s_dash)
    assert np.allclose(pen.color, c_red)
    assert np.isclose(pen.width, w_thin)
    assert pen.style == s_dash

    # Test properties
    pen.color = c_blue
    color = [0.0, 0.0, 0.0]
    pen.GetColorF(color)
    color.append(pen.GetOpacity() / 255)
    assert np.allclose(pen.color, c_blue)
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
    c_red, c_blue = (1, 0, 0, 1), (0, 0, 1, 1)
    t_masonry = examples.download_masonry_texture()
    t_puppy = examples.download_puppy_texture()

    # Test constructor arguments
    brush = charts.Brush(color=c_red, texture=t_masonry)
    assert np.allclose(brush.color, c_red)
    assert np.allclose(brush.texture.to_array(), t_masonry.to_array())

    # Test properties
    brush.color = c_blue
    color = [0.0, 0.0, 0.0, 0.0]
    brush.GetColorF(color)
    assert np.allclose(brush.color, c_blue)
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


@pytest.mark.skip  # TODO: test hangs from time to time?
def test_axis(chart2D):
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
    chart2D.line([0, 1], [1, 10])
    chart2D.show()
    axis = chart2D.y_axis

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
    chart2D.show()  # TODO: find alternative to update all chart properties without plotting (not sure if possible though)
    assert axis.log_scale
    assert axis.GetLogScaleActive()
    axis.log_scale = False
    chart2D.show()
    assert not axis.log_scale
    assert not axis.GetLogScaleActive()
    # TODO: following lines cause "vtkMath::Jacobi: Error extracting eigenfunctions" warning to be printed
    #  probably a vtk issue. Minimum code to reproduce:
    #   chart = pyvista.Chart2D()
    #   chart.line([0, 1], [-10, 10])
    #   axis = chart.y_axis
    #   axis.range = [2, 5]
    #   axis.behavior = "auto"
    #   axis.log_scale = True
    #   chart.show()
    chart2D.line([0, 1], [-10, 10])  # Plot for which log scale cannot be enabled
    axis.log_scale = True
    chart2D.show()
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
    chart2D.show()
    assert tuple(axis.tick_labels) == tuple(f"{loc:.2f}" for loc in tlocs)
    assert vtk_array_to_tuple(axis.GetTickLabels()) == tuple(f"{loc:.2f}" for loc in tlocs)
    assert axis.GetNotation() == charts.Axis.FIXED_NOTATION
    assert axis.GetPrecision() == 2
    axis.tick_labels = "4e"
    axis.tick_locations = tlocs_large  # Add some more variety to labels
    chart2D.show()
    assert tuple(axis.tick_labels) == tuple(to_vtk_scientific(f"{loc:.4e}") for loc in tlocs_large)
    assert vtk_array_to_tuple(axis.GetTickLabels()) == tuple(to_vtk_scientific(f"{loc:.4e}") for loc in tlocs_large)
    assert axis.GetNotation() == charts.Axis.SCIENTIFIC_NOTATION
    assert axis.GetPrecision() == 4
    axis.tick_locations = None
    axis.tick_labels = None
    chart2D.show()
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


@pytest.mark.parametrize("chart_f", ("chart2D", "chartBox", "chartPie", "chartMPL"))
def test_chart_common(pl, chart_f, request):
    # Test the common chart functionalities
    chart = request.getfixturevalue(chart_f)
    title = "Chart title"

    # Check scene and renderer properties
    assert chart._scene is None
    assert chart._renderer is None
    pl.add_chart(chart)
    assert chart._scene is pl.renderer._charts._scene
    assert chart._renderer is pl.renderer and chart._renderer is pl.renderer._charts._renderer

    # TODO: check size and loc properties in dedicated tests
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
    assert chart._geometry == (chart.loc[0]*w, chart.loc[1]*h, chart.size[0]*w, chart.size[1]*h)
    w, h = pl.window_size = [200, 200]
    chart._render_event()
    assert chart._geometry == (chart.loc[0]*w, chart.loc[1]*h, chart.size[0]*w, chart.size[1]*h)

    # Check is_within
    assert chart._is_within(((chart.loc[0]+chart.size[0]/2)*w, (chart.loc[1]+chart.size[1]/2)*h))
    assert not chart._is_within(((chart.loc[0]+chart.size[0]/2)*w, chart.loc[1]*h-5))
    assert not chart._is_within((chart.loc[0]*w-5, (chart.loc[1]+chart.size[1]/2)*h))
    assert not chart._is_within((chart.loc[0]*w-5, chart.loc[1]*h-5))

    # TODO: check chart background properties and title/legend_visible properties using image cache

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


@pytest.mark.parametrize("plot_f", ("linePlot2D", "scatterPlot2D", "areaPlot", "barPlot", "stackPlot", "boxPlot", "piePlot"))
def test_plot_common(plot_f, request):
    # Test the common plot functionalities
    plot = request.getfixturevalue(plot_f)
    c = (1, 0, 1, 1)
    w = 5
    s = "-."
    l = "Label"

    plot.color = c
    assert np.allclose(plot.color, c)
    assert np.allclose(plot.brush.color, c)

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


@pytest.mark.parametrize("plot_f", ("barPlot", "stackPlot", "boxPlot", "piePlot"))
def test_multicomp_plot_common(plot_f, request):
    # Test the common multicomp plot functionalities
    plot = request.getfixturevalue(plot_f)
    cs = "spectrum"
    cs_colors = [(0.0, 0.0, 0.0, 1.0),
                 (0.8941176470588236, 0.10196078431372549, 0.10980392156862745, 1.0),
                 (0.21568627450980393, 0.49411764705882355, 0.7215686274509804, 1.0),
                 (0.30196078431372547, 0.6862745098039216, 0.2901960784313726, 1.0),
                 (0.596078431372549, 0.3058823529411765, 0.6392156862745098, 1.0),
                 (1.0, 0.4980392156862745, 0.0, 1.0),
                 (0.6509803921568628, 0.33725490196078434, 0.1568627450980392, 1.0)]
    colors = [(1, 0, 1, 1), (0, 1, 1, 1), (1, 1, 0, 1)]
    labels = ["Foo", "Spam", "Bla"]

    plot.color_scheme = cs
    assert plot.color_scheme == cs
    assert plot._color_series.GetColorScheme() == plot.COLOR_SCHEMES[cs]["id"]
    assert np.allclose(plot.colors, cs_colors)
    series_colors = [plot._from_c3ub(plot._color_series.GetColor(i)) for i in range(len(cs_colors))]
    assert np.allclose(series_colors, cs_colors)
    lookup_colors = [plot._lookup_table.GetTableValue(i) for i in range(len(cs_colors))]
    assert np.allclose(lookup_colors, cs_colors)
    assert np.allclose(plot.brush.color, cs_colors[0])

    plot.colors = None
    assert plot.color_scheme == plot.DEFAULT_COLOR_SCHEME
    plot.colors = cs
    assert plot.color_scheme == cs
    plot.colors = colors
    assert np.allclose(plot.colors, colors)
    series_colors = [plot._from_c3ub(plot._color_series.GetColor(i)) for i in range(len(colors))]
    assert np.allclose(series_colors, colors)
    lookup_colors = [plot._lookup_table.GetTableValue(i) for i in range(len(colors))]
    assert np.allclose(lookup_colors, colors)
    assert np.allclose(plot.brush.color, colors[0])

    plot.color = colors[1]
    assert np.allclose(plot.color, colors[1])
    assert np.allclose(plot.colors, [colors[1]])
    assert np.allclose(plot.brush.color, colors[1])

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


def test_lineplot2d(linePlot2D):
    x = [-2, -1, 0, 1, 2]
    y = [4, 1, 0, -1, -4]
    c = (1, 0, 1, 1)
    w = 5
    s = "-."
    l = "Line"

    # Test constructor
    plot = charts.LinePlot2D(x, y, c, w, s, l)
    assert np.allclose(plot.x, x)
    assert np.allclose(plot.y, y)
    assert np.allclose(plot.color, c)
    assert plot.line_width == w
    assert plot.line_style == s
    assert plot.label == l

    # Test remaining properties
    linePlot2D.update(x, y)
    assert np.allclose(linePlot2D.x, x)
    assert np.allclose(linePlot2D.y, y)


def test_scatterplot2d(scatterPlot2D):
    x = [-2, -1, 0, 1, 2]
    y = [4, 1, 0, -1, -4]
    c = (1, 0, 1, 1)
    sz = 5
    st, st_inv = "o", "^"
    l = "Scatter"
    assert st_inv not in charts.ScatterPlot2D.MARKER_STYLES, "New marker styles added? Change this test."

    # Test constructor
    plot = charts.ScatterPlot2D(x, y, c, sz, st, l)
    assert np.allclose(plot.x, x)
    assert np.allclose(plot.y, y)
    assert np.allclose(plot.color, c)
    assert plot.marker_size == sz
    assert plot.marker_style == st
    assert plot.label == l

    # Test remaining properties
    scatterPlot2D.update(x, y)
    assert np.allclose(scatterPlot2D.x, x)
    assert np.allclose(scatterPlot2D.y, y)

    scatterPlot2D.marker_size = sz
    assert scatterPlot2D.marker_size == sz
    assert scatterPlot2D.GetMarkerSize() == sz

    scatterPlot2D.marker_style = None
    assert scatterPlot2D.marker_style == ""
    scatterPlot2D.marker_style = st
    assert scatterPlot2D.marker_style == st
    assert scatterPlot2D.GetMarkerStyle() == scatterPlot2D.MARKER_STYLES[st]["id"]
    with pytest.raises(ValueError):
        scatterPlot2D.marker_style = st_inv


def test_areaplot(areaPlot):
    x = [-2, -1, 0, 1, 2]
    y1 = [4, 1, 0, -1, -4]
    y2 = [-4, -2, 0, 2, 4]
    c = (1, 0, 1, 1)
    l = "Line"

    # Test constructor
    plot = charts.AreaPlot(x, y1, y2, c, l)
    assert np.allclose(plot.x, x)
    assert np.allclose(plot.y1, y1)
    assert np.allclose(plot.y2, y2)
    assert np.allclose(plot.color, c)
    assert plot.label == l

    # Test remaining properties
    areaPlot.update(x, y1, y2)
    assert np.allclose(areaPlot.x, x)
    assert np.allclose(areaPlot.y1, y1)
    assert np.allclose(areaPlot.y2, y2)


def test_barplot(barPlot):
    x = [0, 1, 2]
    y = [[1, 2, 3], [2, 1, 0], [1, 1, 1]]
    c = [(1, 0, 1, 1), (1, 1, 0, 1), (0, 1, 1, 1)]
    off = 20
    ori, ori_inv = "H", "I"
    l = ["Foo", "Spam", "Bla"]
    assert ori_inv not in charts.BarPlot.ORIENTATIONS, "New orientations added? Change this test."

    # Test multi comp constructor
    plot = charts.BarPlot(x, y, c, off, ori, l)
    assert np.allclose(plot.x, x)
    assert np.allclose(plot.y, y)
    assert np.allclose(plot.colors, c)
    assert plot.offset == off
    assert plot.orientation == ori
    assert plot.labels == l

    # Test single comp constructor
    plot = charts.BarPlot(x, y[0], c[0], off, ori, l[0])
    assert np.allclose(plot.x, x)
    assert np.allclose(plot.y, y[0])
    assert np.allclose(plot.color, c[0])
    assert plot.offset == off
    assert plot.orientation == ori
    assert plot.label == l[0]

    # Test multi and single comp constructors with inconsistent arguments
    with pytest.raises(ValueError):
        charts.BarPlot(x, y, c[0], off, ori, l)
    # charts.BarPlot(x, y, c, off, ori, l[0])  # This one is valid
    with pytest.raises(ValueError):
        charts.BarPlot(x, y[0], c, off, ori, l[0])
    with pytest.raises(ValueError):
        charts.BarPlot(x, y[0], c[0], off, ori, l)

    # Test remaining properties
    barPlot.update(x, y)
    assert np.allclose(barPlot.x, x)
    assert np.allclose(barPlot.y, y)

    barPlot.offset = off
    assert barPlot.offset == off
    assert barPlot.GetOffset() == off

    barPlot.orientation = ori
    assert barPlot.orientation == ori
    assert barPlot.GetOrientation() == barPlot.ORIENTATIONS[ori]
    with pytest.raises(ValueError):
        barPlot.orientation = ori_inv


def test_stackplot(stackPlot):
    x = [0, 1, 2]
    ys = [[1, 2, 3], [2, 1, 0], [1, 1, 1]]
    c = [(1, 0, 1, 1), (1, 1, 0, 1), (0, 1, 1, 1)]
    l = ["Foo", "Spam", "Bla"]

    # Test multi comp constructor
    plot = charts.StackPlot(x, ys, c, l)
    assert np.allclose(plot.x, x)
    assert np.allclose(plot.ys, ys)
    assert np.allclose(plot.colors, c)
    assert plot.labels == l

    # Test single comp constructor
    plot = charts.StackPlot(x, ys[0], c[0], l[0])
    assert np.allclose(plot.x, x)
    assert np.allclose(plot.ys, ys[0])
    assert np.allclose(plot.color, c[0])
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
    stackPlot.update(x, ys)
    assert np.allclose(stackPlot.x, x)
    assert np.allclose(stackPlot.ys, ys)


def test_chart2D(pl, chart2D):
    size = (0.5, 0.5)
    loc = (0.25, 0.25)
    lx = "X label"
    ly = "Y label"
    rx = [0, 5]
    ry = [0, 1]
    x = np.arange(11)-5
    y = x**2
    ys = [np.sin(x), np.cos(x), np.tanh(x)]
    col = (1, 0, 1, 1)
    cs = "citrus"
    sz = 5
    ms = "d"
    w = 10
    ls = "-."
    off = 12
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
    assert np.allclose(chart._geometry, (loc[0]*r_w, loc[1]*r_h, size[0]*r_w, size[1]*r_h))
    pl.window_size = (int(pl.window_size[0]/2), int(pl.window_size[1]/2))
    pl.show()  # This will also call chart._resize
    assert np.allclose(chart._geometry, (loc[0]*r_w/2, loc[1]*r_h/2, size[0]*r_w/2, size[1]*r_h/2))

    # Test parse_format
    colors = itertools.chain(pyvista.hexcolors, pyvista.color_char_to_word, ["#fa09b6", ""])
    for m in charts.ScatterPlot2D.MARKER_STYLES:
        for l in charts.Pen.LINE_STYLES:
            for c in colors:
                cp = "b" if c == "" else c
                assert (m, l, cp) == chart2D._parse_format(m + l + c)
                assert (m, l, cp) == chart2D._parse_format(m + c + l)
                assert (m, l, cp) == chart2D._parse_format(l + m + c)
                assert (m, l, cp) == chart2D._parse_format(l + c + m)
                assert (m, l, cp) == chart2D._parse_format(c + m + l)
                assert (m, l, cp) == chart2D._parse_format(c + l + m)

    # Test plotting methods
    s, l = chart2D.plot(x, y, "")
    assert s is None and l is None
    assert len([*chart2D.plots()]) == 0
    s, l = chart2D.plot(y, "-")
    assert s is None and l is not None
    assert l in chart2D.plots("line")
    chart2D.remove_plot(l)
    assert len([*chart2D.plots()]) == 0
    s, l = chart2D.plot(y, "x")
    assert s is not None and l is None
    assert s in chart2D.plots("scatter")
    chart2D.clear("scatter")
    assert len([*chart2D.plots()]) == 0
    s, l = chart2D.plot(x, y, "x-")
    assert s is not None and l is not None
    assert s in chart2D.plots("scatter") and l in chart2D.plots("line")
    chart2D.plot(x, y, "x-")  # Check clearing of multiple plots (of the same type)
    chart2D.clear()
    assert len([*chart2D.plots()]) == 0

    s = chart2D.scatter(x, y, col, sz, ms, lx)
    assert np.allclose(s.x, x)
    assert np.allclose(s.y, y)
    assert np.allclose(s.color, col)
    assert s.marker_size == sz
    assert s.marker_style == ms
    assert s.label == lx
    assert s in chart2D.plots("scatter")
    assert chart2D.GetPlotIndex(s) >= 0

    l = chart2D.line(x, y, col, w, ls, lx)
    assert np.allclose(l.x, x)
    assert np.allclose(l.y, y)
    assert np.allclose(l.color, col)
    assert l.line_width == w
    assert l.line_style == ls
    assert l.label == lx
    assert l in chart2D.plots("line")
    assert chart2D.GetPlotIndex(l) >= 0

    a = chart2D.area(x, -y, y, col, lx)
    assert np.allclose(a.x, x)
    assert np.allclose(a.y1, -y)
    assert np.allclose(a.y2, y)
    assert np.allclose(a.color, col)
    assert a.label == lx
    assert a in chart2D.plots("area")
    assert chart2D.GetPlotIndex(a) >= 0

    b = chart2D.bar(x, -y, col, off, ori, lx)
    assert np.allclose(b.x, x)
    assert np.allclose(b.y, -y)
    assert np.allclose(b.color, col)
    assert b.offset == off
    assert b.orientation == ori
    assert b.label == lx
    assert b in chart2D.plots("bar")
    assert chart2D.GetPlotIndex(b) >= 0

    s = chart2D.stack(x, ys, cs, [lx, ly])
    assert np.allclose(s.x, x)
    assert np.allclose(s.ys, ys)
    assert s.color_scheme == cs
    assert tuple(s.labels) == (lx, ly)
    assert s in chart2D.plots("stack")
    assert chart2D.GetPlotIndex(s) >= 0

    inv_type = "blub"
    with pytest.raises(KeyError):
        next(chart2D.plots(inv_type))
    with pytest.raises(KeyError):
        chart2D.clear(inv_type)
    assert len([*chart2D.plots()]) == 5
    chart2D.clear()
    assert len([*chart2D.plots()]) == 0
    with pytest.raises(ValueError):
        chart2D.remove_plot(s)

    # Check remaining properties
    assert chart2D.x_axis.__this__ == chart2D.GetAxis(charts.Axis.BOTTOM).__this__
    assert chart2D.y_axis.__this__ == chart2D.GetAxis(charts.Axis.LEFT).__this__

    chart2D.x_label = lx
    assert chart2D.x_label == lx
    assert chart2D.x_axis.label == lx
    chart2D.y_label = ly
    assert chart2D.y_label == ly
    assert chart2D.y_axis.label == ly

    chart2D.x_range = rx
    assert np.allclose(chart2D.x_range, rx)
    assert np.allclose(chart2D.x_axis.range, rx)
    chart2D.y_range = ry
    assert np.allclose(chart2D.y_range, ry)
    assert np.allclose(chart2D.y_axis.range, ry)

    chart2D.grid = True
    assert chart2D.grid
    assert chart2D.x_axis.grid and chart2D.y_axis.grid

    chart2D.hide_axes()
    for axis in (chart2D.x_axis, chart2D.y_axis):
        assert not (axis.visible or axis.label_visible or axis.ticks_visible or axis.tick_labels_visible or axis.grid)


def test_chartBox(pl, chartBox, boxPlot):
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
    pl.window_size = (int(pl.window_size[0]/2), int(pl.window_size[1]/2))
    pl.show()  # This will also call chart._resize
    assert np.allclose(chart._geometry, (0, 0, r_w/2, r_h/2))

    # Test remaining properties
    assert chartBox.loc == (0, 0)
    assert chartBox.size == (1, 1)
    assert chartBox.plot.__this__ == chartBox.GetPlot(0).__this__

    boxPlot.update(data)
    assert np.allclose(boxPlot.data, data)
    assert np.allclose(boxPlot.stats, stats)


def test_chartPie(pl, chartPie, piePlot):
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
    pl.window_size = (int(pl.window_size[0]/2), int(pl.window_size[1]/2))
    pl.show()  # This will also call chart._resize
    assert np.allclose(chart._geometry, (0, 0, r_w/2, r_h/2))

    # Test remaining properties
    assert chartPie.loc == (0, 0)
    assert chartPie.size == (1, 1)
    assert chartPie.plot.__this__ == chartPie.GetPlot(0).__this__

    piePlot.update(data)
    assert np.allclose(piePlot.data, data)
