import os
import pyvista as pv


def make_line_style_img(line_style, path):
    p = pv.Plotter(off_screen=True, window_size=[100, 50])
    p.background_color = 'w'
    chart = pv.Chart2D()
    chart.line([0, 1], [0, 0], color="b", width=3.0, style=line_style)
    chart.hide_axes()
    p.add_chart(chart)
    _, img = p.show(screenshot=True, return_cpos=True)
    # Crop the image and save it
    p._save_image(img[18:25, 22:85, :], path, False)


def make_line_style_table():
    header = """
.. list-table:: Line styles
   :widths: 20 40 40
   :header-rows: 1

   * - Style
     - Description
     - Example
"""
    row_template = """
   * - ``"{}"``
     - {}
     - .. image:: /{}
"""
    filename = "./api/plotting/pen_line_styles.rst"
    if os.path.exists(filename):
        os.remove(filename)
    with open(filename, "w") as f:
        f.write(header)
        for i, (ls, li) in enumerate(pv.charts.Pen.LINE_STYLES.items()):
            if li["descr"] is not None:
                img_path = f"images/charts/ls_{i}.png"
                make_line_style_img(ls, img_path)
                f.write(row_template.format(ls, li["descr"], img_path))
    pv.close_all()


def make_marker_style_img(marker_style, path):
    p = pv.Plotter(off_screen=True, window_size=[100, 100])
    p.background_color = 'w'
    chart = pv.Chart2D()
    chart.scatter([0], [0], color="b", size=9, style=marker_style)
    chart.hide_axes()
    p.add_chart(chart)
    _, img = p.show(screenshot=True, return_cpos=True)
    # Crop the image and save it
    # p._save_image(img[18:25, 50:57, :], path, False)  # window_size=[100,50] and marker_size=3
    p._save_image(img[40:53, 47:60, :], path, False)


def make_marker_style_table():
    header = """
.. list-table:: Marker styles
   :widths: 20 40 40
   :header-rows: 1

   * - Style
     - Description
     - Example
"""
    row_template = """
   * - ``"{}"``
     - {}
     - .. image:: /{}
"""
    filename = "./api/plotting/scatter_marker_styles.rst"
    if os.path.exists(filename):
        os.remove(filename)
    with open(filename, "w") as f:
        f.write(header)
        for i, (ms, mi) in enumerate(pv.charts.ScatterPlot2D.MARKER_STYLES.items()):
            if mi["descr"] is not None:
                img_path = f"images/charts/ms_{i}.png"
                make_marker_style_img(ms, img_path)
                f.write(row_template.format(ms, mi["descr"], img_path))
    pv.close_all()


def make_color_scheme_img(color_scheme, path):
    p = pv.Plotter(off_screen=True, window_size=[240, 120])
    p.background_color = 'w'
    chart = pv.Chart2D()
    tmp_plot = chart.bar([0], [[1]]*2, color=color_scheme, orientation="H")
    N = len(tmp_plot.colors)  # Use a temporary plot to determine the total number of colors in this scheme
    plot = chart.bar([0], [[1]]*N, color=color_scheme, orientation="H")
    chart.remove_plot(tmp_plot)
    plot.pen.color = 'w'
    chart.x_range = [0, N]
    chart.hide_axes()
    p.add_chart(chart)
    _, img = p.show(screenshot=True, return_cpos=True)
    # Crop the image and save it
    p._save_image(img[34:78, 22:225, :], path, False)
    return N


def make_color_scheme_table():
    header = """
.. list-table:: Color schemes
   :widths: 15 50 5 30
   :header-rows: 1

   * - Color scheme
     - Description
     - # colors
     - Example
"""
    row_template = """
   * - ``"{}"``
     - {}
     - {}
     - .. image:: /{}
"""
    filename = "./api/plotting/plot_color_schemes.rst"
    if os.path.exists(filename):
        os.remove(filename)
    with open(filename, "w", encoding="utf-8") as f:
        f.write(header)
        for i, (cs, ci) in enumerate(pv.charts._MultiCompPlot.COLOR_SCHEMES.items()):
            if ci["descr"] is not None:
                img_path = f"images/charts/cs_{i}.png"
                N = make_color_scheme_img(cs, img_path)
                f.write(row_template.format(cs, ci["descr"], N, img_path))
    pv.close_all()


def make_all():
    make_line_style_table()
    make_marker_style_table()
    make_color_scheme_table()
