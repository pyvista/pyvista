import os
import pyvista as pv

TABLE_DIR = "api/plotting/charts"
IMAGE_DIR = "images/charts"


def make_table(file_name, header, get_token_row, tokens):
    path = f"{TABLE_DIR}/{file_name}.rst"
    if os.path.exists(path):
        os.remove(path)
    with open(path, "w", encoding="utf-8") as f:
        f.write(header)
        for i, (token, data) in enumerate(tokens.items()):
            if data["descr"] is not None:
                f.write(get_token_row(i, token, data))
    pv.close_all()


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


def get_line_style_row(i, token, data):
    row_template = """
   * - ``"{}"``
     - {}
     - .. image:: /{}
"""
    img_path = f"{IMAGE_DIR}/ls_{i}.png"
    make_line_style_img(token, img_path)
    return row_template.format(token, data["descr"], img_path)


def make_line_style_table():
    header = """
.. list-table:: Line styles
   :widths: 20 40 40
   :header-rows: 1

   * - Style
     - Description
     - Example
"""
    make_table("pen_line_styles", header, get_line_style_row, pv.charts.Pen.LINE_STYLES)


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


def get_marker_style_row(i, token, data):
    row_template = """
   * - ``"{}"``
     - {}
     - .. image:: /{}
"""
    img_path = f"{IMAGE_DIR}/ms_{i}.png"
    make_marker_style_img(token, img_path)
    return row_template.format(token, data["descr"], img_path)


def make_marker_style_table():
    header = """
.. list-table:: Marker styles
   :widths: 20 40 40
   :header-rows: 1

   * - Style
     - Description
     - Example
"""
    make_table("scatter_marker_styles", header, get_marker_style_row, pv.charts.ScatterPlot2D.MARKER_STYLES)


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


def get_color_scheme_row(i, token, data):
    row_template = """
   * - ``"{}"``
     - {}
     - {}
     - .. image:: /{}
"""
    img_path = f"{IMAGE_DIR}/cs_{i}.png"
    N = make_color_scheme_img(token, img_path)
    return row_template.format(token, data["descr"], N, img_path)


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
    make_table("plot_color_schemes", header, get_color_scheme_row, pv.charts._MultiCompPlot.COLOR_SCHEMES)


def make_all():
    os.makedirs(IMAGE_DIR, exist_ok=True)
    make_line_style_table()
    make_marker_style_table()
    make_color_scheme_table()
