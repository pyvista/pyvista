"""Test charting functionality"""

import platform

import pytest
import numpy as np
import matplotlib.pyplot as plt

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
    p = pyvista.Plotter()
    p.background_color = 'w'
    return p


@pytest.fixture
def chart2D():
    chart = pyvista.Chart2D()
    chart.plot([0, 1, 2], [3, 1, 2])
    return chart


@pytest.fixture
def chartBox():
    return pyvista.ChartBox([1, 2, 3])


@pytest.fixture
def chartPie():
    return pyvista.ChartPie([1, 2, 3])


@pytest.fixture
def chartMPL():
    f, ax = plt.subplots()
    ax.plot([0, 1, 2], [3, 1, 2])
    return pyvista.ChartMPL(f)


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


def test_axis():
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
    chart = pyvista.Chart2D()
    chart.line([0, 1], [1, 10])
    chart.show()
    axis = chart.y_axis

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
    chart.show()  # TODO: find alternative to update all chart properties without plotting (not sure if possible though)
    assert axis.log_scale
    assert axis.GetLogScaleActive()
    axis.log_scale = False
    chart.show()
    assert not axis.log_scale
    assert not axis.GetLogScaleActive()
    # TODO: following lines cause "vtkMath::Jacobi: Error extracting eigenfunctions" warning to be printed
    # chart.line([0, 1], [-10, 10])  # Plot for which log scale cannot be enabled
    # axis.log_scale = True
    # chart.show()
    # assert not axis.log_scale
    # assert not axis.GetLogScaleActive()

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
    chart.show()
    assert tuple(axis.tick_labels) == tuple(f"{loc:.2f}" for loc in tlocs)
    assert vtk_array_to_tuple(axis.GetTickLabels()) == tuple(f"{loc:.2f}" for loc in tlocs)
    assert axis.GetNotation() == pyvista._vtk.vtkAxis.FIXED_NOTATION
    assert axis.GetPrecision() == 2
    axis.tick_labels = "4e"
    axis.tick_locations = tlocs_large  # Add some more variety to labels
    chart.show()
    assert tuple(axis.tick_labels) == tuple(to_vtk_scientific(f"{loc:.4e}") for loc in tlocs_large)
    assert vtk_array_to_tuple(axis.GetTickLabels()) == tuple(to_vtk_scientific(f"{loc:.4e}") for loc in tlocs_large)
    assert axis.GetNotation() == pyvista._vtk.vtkAxis.SCIENTIFIC_NOTATION
    assert axis.GetPrecision() == 4
    axis.tick_locations = None
    axis.tick_labels = None
    chart.show()
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


def test_plot_common():
    # Test the common plot functionalities
    pass


def test_multicomp_plot_common():
    # Test the common multicomp plot functionalities
    pass
