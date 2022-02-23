import os
import textwrap

import pyvista as pv

CHARTS_TABLE_DIR = "api/plotting/charts"
CHARTS_IMAGE_DIR = "images/charts"
COLORS_TABLE_DIR = "api/utilities"


class DocTable:
    """Helper class to create tables for the documentation."""

    path = None  # Path to the rst file to which the table will be written

    @classmethod
    def generate(cls):
        assert cls.path is not None, "Subclasses should specify a path."
        if os.path.exists(cls.path):
            os.remove(cls.path)
        tokens = cls.fetch_tokens()
        with open(cls.path, "w", encoding="utf-8") as f:
            f.write(cls.get_header(tokens))
            for i, (token, data) in enumerate(tokens.items()):
                row = cls.get_row(i, token, data)
                if row is not None:
                    f.write(row)
        pv.close_all()

    @classmethod
    def fetch_tokens(cls):
        raise NotImplementedError("Subclasses should specify a fetch_tokens method.")

    @classmethod
    def get_header(cls, tokens):
        raise NotImplementedError("Subclasses should specify a table header.")

    @classmethod
    def get_row(cls, i, token, data):
        raise NotImplementedError("Subclasses should specify a get_row method.")


class LineStyleTable(DocTable):
    """Class to generate line style table."""

    path = f"{CHARTS_TABLE_DIR}/pen_line_styles.rst"
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

    @classmethod
    def fetch_tokens(cls):
        return pv.charts.Pen.LINE_STYLES

    @classmethod
    def get_header(cls, tokens):
        return cls.header

    @classmethod
    def get_row(cls, i, token, data):
        if data["descr"] is None:
            return None
        else:
            img_path = f"{CHARTS_IMAGE_DIR}/ls_{i}.png"
            cls.generate_img(token, img_path)
            return cls.row_template.format(token, data["descr"], img_path)

    @staticmethod
    def generate_img(line_style, img_path):
        p = pv.Plotter(off_screen=True, window_size=[100, 50])
        p.background_color = 'w'
        chart = pv.Chart2D()
        chart.line([0, 1], [0, 0], color="b", width=3.0, style=line_style)
        chart.hide_axes()
        p.add_chart(chart)
        _, img = p.show(screenshot=True, return_cpos=True)
        # Crop the image and save it
        p._save_image(img[18:25, 22:85, :], img_path, False)


class MarkerStyleTable(DocTable):
    """Class to generate marker style table."""

    path = f"{CHARTS_TABLE_DIR}/scatter_marker_styles.rst"
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

    @classmethod
    def fetch_tokens(cls):
        return pv.charts.ScatterPlot2D.MARKER_STYLES

    @classmethod
    def get_header(cls, tokens):
        return cls.header

    @classmethod
    def get_row(cls, i, token, data):
        if data["descr"] is None:
            return None
        else:
            img_path = f"{CHARTS_IMAGE_DIR}/ms_{i}.png"
            cls.generate_img(token, img_path)
            return cls.row_template.format(token, data["descr"], img_path)

    @staticmethod
    def generate_img(marker_style, img_path):
        p = pv.Plotter(off_screen=True, window_size=[100, 100])
        p.background_color = 'w'
        chart = pv.Chart2D()
        chart.scatter([0], [0], color="b", size=9, style=marker_style)
        chart.hide_axes()
        p.add_chart(chart)
        _, img = p.show(screenshot=True, return_cpos=True)
        # Crop the image and save it
        # p._save_image(img[18:25, 50:57, :], path, False)  # window_size=[100,50] and marker_size=3
        p._save_image(img[40:53, 47:60, :], img_path, False)


class ColorSchemeTable(DocTable):
    """Class to generate color scheme table."""

    path = f"{CHARTS_TABLE_DIR}/plot_color_schemes.rst"
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

    @classmethod
    def fetch_tokens(cls):
        return pv.charts._MultiCompPlot.COLOR_SCHEMES

    @classmethod
    def get_header(cls, tokens):
        return cls.header

    @classmethod
    def get_row(cls, i, token, data):
        if data["descr"] is None:
            return None
        else:
            img_path = f"{CHARTS_IMAGE_DIR}/cs_{i}.png"
            n_colors = cls.generate_img(token, img_path)
            return cls.row_template.format(token, data["descr"], n_colors, img_path)

    @staticmethod
    def generate_img(color_scheme, img_path):
        p = pv.Plotter(off_screen=True, window_size=[240, 120])
        p.background_color = 'w'
        chart = pv.Chart2D()
        # Use a temporary plot to determine the total number of colors in this scheme
        tmp_plot = chart.bar([0], [[1]] * 2, color=color_scheme, orientation="H")
        n_colors = len(tmp_plot.colors)
        plot = chart.bar([0], [[1]] * n_colors, color=color_scheme, orientation="H")
        chart.remove_plot(tmp_plot)
        plot.pen.color = 'w'
        chart.x_range = [0, n_colors]
        chart.hide_axes()
        p.add_chart(chart)
        _, img = p.show(screenshot=True, return_cpos=True)
        # Crop the image and save it
        p._save_image(img[34:78, 22:225, :], img_path, False)
        return n_colors


class ColorTable(DocTable):
    """Class to generate colors table."""

    path = f"{COLORS_TABLE_DIR}/colors.rst"
    header = textwrap.dedent(
        """
        .. list-table::
           :widths: 50 20 30
           :header-rows: 1

           * - Name
             - Hex value
             - Example
        """
    )
    row_template = textwrap.indent(
        textwrap.dedent(
            """
            * - {}
              - ``{}``
              - .. raw:: html

                   <span style='width:100%; height:100%; display:block; background-color: {};'>&nbsp;</span>
            """
        ),
        "   ",  # Extra indent needed to make it part of the table
    )

    @classmethod
    def fetch_tokens(cls):
        tokens = {token: {"hex": hex} for token, hex in pv.hexcolors.items()}
        for s, token in pv.colors.color_synonyms.items():
            if "synonyms" not in tokens[token]:
                tokens[token]["synonyms"] = []
            tokens[token]["synonyms"].append(s)
        return tokens

    @classmethod
    def get_header(cls, tokens):
        return cls.header

    @classmethod
    def get_row(cls, i, token, data):
        token_template = '``"{}"``'
        names = [token] + data.get("synonyms", [])
        name = " or ".join(token_template.format(n) for n in names)
        return cls.row_template.format(name, data["hex"], data["hex"])


def make_all():
    os.makedirs(CHARTS_IMAGE_DIR, exist_ok=True)
    LineStyleTable.generate()
    MarkerStyleTable.generate()
    ColorSchemeTable.generate()
    ColorTable.generate()
