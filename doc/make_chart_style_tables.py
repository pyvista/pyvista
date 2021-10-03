import os
import pyvista as pv


def make_line_style_img(line_style, path):
    p = pv.Plotter(off_screen=True, window_size=[100, 50])
    p.background_color = 'w'
    chart = pv.Chart2D()
    chart.line([0, 1], [0, 0], color="k", width=3.0, style=line_style)
    chart.hide_axes()
    p.add_chart(chart)
    _, img = p.show(screenshot=True, return_cpos=True)
    # Crop the image and save it
    p._save_image(img[18:25, 22:85, :], path, False)


def make_line_style_table():
    ls_skips = []
    ls_descr = {
        "": "Hidden",
        "-": "Solid",
        "--": "Dashed",
        ":": "Dotted",
        "-.": "Dash-dot",
        "-..": "Dash-dot-dot"
    }
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
     - .. figure:: ../../../{}
"""
    filename = "./api/plotting/pen_line_styles.rst"
    if os.path.exists(filename):
        os.remove(filename)
    with open(filename, "w") as f:
        f.write(header)
        for i, ls in enumerate(pv.charts.Pen.LINE_STYLES.keys()):
            if ls not in ls_skips:
                img_path = f"images/charts/ls_{i}.png"
                make_line_style_img(ls, img_path)
                f.write(row_template.format(ls, ls_descr[ls], img_path))
    pv.close_all()


def make_all():
    make_line_style_table()
    # make_marker_style_table()
    # make_color_scheme_table()
