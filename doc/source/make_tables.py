"""This is a helper module to generate tables that can be included in the documentation."""

import inspect
import io
import os
import textwrap
from types import FunctionType, ModuleType
from typing import Dict, Type, TypeVar

import pyvista as pv
from pyvista.core.errors import VTKVersionError
from pyvista.examples import _example_loader, downloads

# Paths to directories in which resulting rst files and images are stored.
CHARTS_TABLE_DIR = "api/plotting/charts"
CHARTS_IMAGE_DIR = "images/charts"
COLORS_TABLE_DIR = "api/utilities"
DOWNLOADS_TABLE_DIR = "examples"
PREVIEW_EXAMPLES_IMAGES_DIR = "images/preview-examples"


def _aligned_dedent(txt):
    """Custom variant of `textwrap.dedent`.

    Helper method to dedent the provided text up to the special alignment character ``'|'``.
    """
    return textwrap.dedent(txt).replace('|', '')


class DocTable:
    """Helper class to create tables for the documentation.

    The ``generate`` method creates the table rst file (and possibly any other
    files, such as images, used by the table). This method internally calls
    the ``fetch_data``, ``get_header`` and ``get_row`` methods, which should be
    provided by any subclass.
    Each table is generated from a list of 'row_data' provided by the ``fetch_data``
    method. The ``get_header`` and ``get_row`` methods generate the required rst
    for the table's header and table's rows respectively.
    """

    path = None  # Path to the rst file to which the table will be written

    @classmethod
    def generate(cls):
        """Generate this table."""
        assert cls.path is not None, "Subclasses should specify a path."
        data = cls.fetch_data()

        with io.StringIO() as fnew:
            fnew.write(cls.get_header(data))
            for i, row_data in enumerate(data):
                row = cls.get_row(i, row_data)
                if row is not None:
                    fnew.write(row)

            # if file exists, verify that we have no new content
            fnew.seek(0)
            new_txt = fnew.read()

        # determine if existing file needs to be rewritten
        if os.path.exists(cls.path):
            with open(cls.path) as fold:
                orig_txt = fold.read()
            if orig_txt == new_txt:
                new_txt = ''

        # write if there is any text to write. This avoids resetting the documentation cache
        if new_txt:
            with open(cls.path, 'w', encoding="utf-8") as fout:
                fout.write(new_txt)

        pv.close_all()

    @classmethod
    def fetch_data(cls):
        """Get a list of row_data used to generate the table."""
        raise NotImplementedError("Subclasses should specify a fetch_data method.")

    @classmethod
    def get_header(cls, data):
        """Get the table's header rst."""
        raise NotImplementedError("Subclasses should specify a table header.")

    @classmethod
    def get_row(cls, i, row_data):
        """Get the rst for the given row. Can return ``None`` if no row should
        be generated for the provided ``row_data``."""
        raise NotImplementedError("Subclasses should specify a get_row method.")


class LineStyleTable(DocTable):
    """Class to generate line style table."""

    path = f"{CHARTS_TABLE_DIR}/pen_line_styles.rst"
    header = _aligned_dedent(
        """
        |.. list-table:: Line styles
        |   :widths: 20 40 40
        |   :header-rows: 1
        |
        |   * - Style
        |     - Description
        |     - Example
        """
    )
    row_template = _aligned_dedent(
        """
        |   * - ``"{}"``
        |     - {}
        |     - .. image:: /{}
        """
    )

    @classmethod
    def fetch_data(cls):
        # Fetch table data from ``LINE_STYLES`` dictionary.
        return [{"style": ls, **data} for (ls, data) in pv.charts.Pen.LINE_STYLES.items()]

    @classmethod
    def get_header(cls, data):
        return cls.header

    @classmethod
    def get_row(cls, i, row_data):
        if row_data["descr"] is None:
            return None  # Skip line style if description is set to ``None``.
        else:
            # Create an image from the given line style and generate the row rst.
            img_path = f"{CHARTS_IMAGE_DIR}/ls_{i}.png"
            cls.generate_img(row_data["style"], img_path)
            return cls.row_template.format(row_data["style"], row_data["descr"], img_path)

    @staticmethod
    def generate_img(line_style, img_path):
        """Generate and save an image of the given line_style."""
        p = pv.Plotter(off_screen=True, window_size=[100, 50])
        p.background_color = 'w'
        chart = pv.Chart2D()
        chart.line([0, 1], [0, 0], color="b", width=3.0, style=line_style)
        chart.hide_axes()
        p.add_chart(chart)

        # Generate and crop the image
        _, img = p.show(screenshot=True, return_cpos=True)
        img = img[18:25, 22:85, :]

        # exit early if the image already exists and is the same
        if os.path.isfile(img_path) and pv.compare_images(img, img_path) < 1:
            return

        # save it
        p._save_image(img, img_path, False)


class MarkerStyleTable(DocTable):
    """Class to generate marker style table."""

    path = f"{CHARTS_TABLE_DIR}/scatter_marker_styles.rst"
    header = _aligned_dedent(
        """
        |.. list-table:: Marker styles
        |   :widths: 20 40 40
        |   :header-rows: 1
        |
        |   * - Style
        |     - Description
        |     - Example
        """
    )
    row_template = _aligned_dedent(
        """
        |   * - ``"{}"``
        |     - {}
        |     - .. image:: /{}
        """
    )

    @classmethod
    def fetch_data(cls):
        # Fetch table data from ``MARKER_STYLES`` dictionary.
        return [
            {"style": ms, **data} for (ms, data) in pv.charts.ScatterPlot2D.MARKER_STYLES.items()
        ]

    @classmethod
    def get_header(cls, data):
        return cls.header

    @classmethod
    def get_row(cls, i, row_data):
        if row_data["descr"] is None:
            return None  # Skip marker style if description is set to ``None``.
        else:
            # Create an image from the given marker style and generate the row rst.
            img_path = f"{CHARTS_IMAGE_DIR}/ms_{i}.png"
            cls.generate_img(row_data["style"], img_path)
            return cls.row_template.format(row_data["style"], row_data["descr"], img_path)

    @staticmethod
    def generate_img(marker_style, img_path):
        """Generate and save an image of the given marker_style."""
        p = pv.Plotter(off_screen=True, window_size=[100, 100])
        p.background_color = 'w'
        chart = pv.Chart2D()
        chart.scatter([0], [0], color="b", size=9, style=marker_style)
        chart.hide_axes()
        p.add_chart(chart)

        # generate and crop the image
        _, img = p.show(screenshot=True, return_cpos=True)
        img = img[40:53, 47:60, :]

        # exit early if the image already exists and is the same
        if os.path.isfile(img_path) and pv.compare_images(img, img_path) < 1:
            return

        # save it
        p._save_image(img, img_path, False)


class ColorSchemeTable(DocTable):
    """Class to generate color scheme table."""

    path = f"{CHARTS_TABLE_DIR}/plot_color_schemes.rst"
    header = _aligned_dedent(
        """
        |.. list-table:: Color schemes
        |   :widths: 15 50 5 30
        |   :header-rows: 1
        |
        |   * - Color scheme
        |     - Description
        |     - # colors
        |     - Example
        """
    )
    row_template = _aligned_dedent(
        """
        |   * - ``"{}"``
        |     - {}
        |     - {}
        |     - .. image:: /{}
        """
    )

    @classmethod
    def fetch_data(cls):
        # Fetch table data from ``COLOR_SCHEMES`` dictionary.
        return [{"scheme": cs, **data} for (cs, data) in pv.colors.COLOR_SCHEMES.items()]

    @classmethod
    def get_header(cls, data):
        return cls.header

    @classmethod
    def get_row(cls, i, row_data):
        if row_data["descr"] is None:
            return None  # Skip color scheme if description is set to ``None``.
        else:
            # Create an image from the given color scheme and generate the row rst.
            img_path = f"{CHARTS_IMAGE_DIR}/cs_{i}.png"
            n_colors = cls.generate_img(row_data["scheme"], img_path)
            return cls.row_template.format(
                row_data["scheme"], row_data["descr"], n_colors, img_path
            )

    @staticmethod
    def generate_img(color_scheme, img_path):
        """Generate and save an image of the given color_scheme."""
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

        # Generate and crop the image
        _, img = p.show(screenshot=True, return_cpos=True)
        img = img[34:78, 22:225, :]

        # exit early if the image already exists and is the same
        if os.path.isfile(img_path) and pv.compare_images(img, img_path) < 1:
            return n_colors

        # save it
        p._save_image(img, img_path, False)

        return n_colors


class ColorTable(DocTable):
    """Class to generate colors table."""

    path = f"{COLORS_TABLE_DIR}/colors.rst"
    header = _aligned_dedent(
        """
        |.. list-table::
        |   :widths: 50 20 30
        |   :header-rows: 1
        |
        |   * - Name
        |     - Hex value
        |     - Example
        """
    )
    row_template = _aligned_dedent(
        """
        |   * - {}
        |     - ``{}``
        |     - .. raw:: html
        |
        |          <span style='width:100%; height:100%; display:block; background-color: {};'>&nbsp;</span>
        """
    )

    @classmethod
    def fetch_data(cls):
        # Fetch table data from ``hexcolors`` dictionary.
        colors = {
            name: {"name": name, "hex": hex, "synonyms": []} for name, hex in pv.hexcolors.items()
        }
        # Add synonyms defined in ``color_synonyms`` dictionary.
        for s, name in pv.colors.color_synonyms.items():
            colors[name]["synonyms"].append(s)
        return colors.values()

    @classmethod
    def get_header(cls, data):
        return cls.header

    @classmethod
    def get_row(cls, i, row_data):
        name_template = '``"{}"``'
        names = [row_data["name"]] + row_data["synonyms"]
        name = " or ".join(name_template.format(n) for n in names)
        return cls.row_template.format(name, row_data["hex"], row_data["hex"])


class DownloadsInfoTable(DocTable):
    """Class to generate info about pyvista downloadable examples."""

    path = f"{DOWNLOADS_TABLE_DIR}/downloads_info.rst"
    header = _aligned_dedent(
        """
        |.. list-table::
        |   :widths: 100
        |   :header-rows: 1
        |
        |   * - Examples
        """
    )
    # each example has its own sub-table
    row_template = _aligned_dedent(
        """
        |.. list-table::
        |   :widths: 50 50
        |   :header-rows: 1
        |
        |   * :func:`~{}`
        |
        |   * - Info
        |     - Preview
        |
        |   * - {}
        |       Size on Disk: {}
        |       Num Files: {}
        |       Extension: {}
        |       Reader: {}
        |       Representation:
        |       {}
        |     - {}
        """
    )

    @classmethod
    def fetch_data(cls):
        os.makedirs(PREVIEW_EXAMPLES_IMAGES_DIR, exist_ok=True)
        # Collect all `_example_<name>` file loaders
        module_members = dict(inspect.getmembers(pv.examples.downloads))
        example_file_loaders = {
            name: item
            for name, item in module_members.items()
            if name.startswith('_example_')
            and isinstance(
                item,
                (
                    _example_loader._SingleFileDownloadable,
                    _example_loader._MultiFileDownloadableLoadable,
                ),
            )
        }
        # new = dict(example_file_loaders)
        # for i, key in enumerate(example_file_loaders):
        #     if i < 100:
        #         new.pop(key)
        # example_file_loaders = new
        return sorted(example_file_loaders.items())

    @classmethod
    def get_header(cls, data):
        return cls.header

    @classmethod
    def get_row(cls, i, row_data):
        loader_name, loader = row_data
        download_name = loader_name.replace('_example_', 'download_')

        # Get the corresponding 'download' function for the example
        download_func = getattr(downloads, download_name)
        func_fullname = _get_fullname(download_func)
        doc_line = download_func.__doc__.splitlines()[0]

        try:
            loader.download()
        except VTKVersionError:
            # Skip example
            return None

        # Get file info
        file_size = loader.total_size
        num_files = loader.num_files
        file_ext = ' '.join(loader.extension)
        reader_type = loader.reader

        # Get instance info
        dataset = loader.load()
        # TODO: parse repr string and replace any types with linkable references to the class doc(s)
        data_repr = repr(dataset)

        # Create a preview image of the dataset
        img_path = f"{PREVIEW_EXAMPLES_IMAGES_DIR}/{download_name}.png"
        try:
            cls.generate_img(dataset, img_path)
            img_rst = '..image:: /' + img_path
        except NotImplementedError:
            # Cannot generate image for this example (e.g. loading arbitrary numpy array)
            img_rst = ''
        except (RuntimeError, AttributeError, ValueError, TypeError, FileNotFoundError):
            raise RuntimeError(
                f"Unable to generate a preview image for example \'{download_name}\'"
            )

        return cls.row_template.format(
            func_fullname,
            doc_line,
            file_size,
            num_files,
            file_ext,
            reader_type,
            data_repr,
            img_rst,
        )

    @staticmethod
    def generate_img(dataset, img_path):
        """Generate and save an image of the given download object."""
        p = pv.Plotter(off_screen=True, window_size=[300, 300])
        p.background_color = 'w'
        if isinstance(dataset, pv.Texture):
            p.add_mesh(pv.Plane(), texture=dataset)
        elif isinstance(dataset, (tuple, pv.MultiBlock)):
            p.add_composite(dataset)
        else:
            p.add_mesh(dataset)
        img = p.show(screenshot=True)

        # # exit early if the image already exists and is the same
        # if os.path.isfile(img_path) and pv.compare_images(img, img_path) < 1:
        #     return
        # save it
        p._save_image(img, img_path, False)
        return True


_TypeType = TypeVar('_TypeType', bound=Type)


def _get_module_members(module: ModuleType, typ: _TypeType) -> Dict[str, _TypeType]:
    """Get all members of a specified type which are defined locally inside a module."""

    def is_local(obj):
        return type(obj) is typ and obj.__module__ == module.__name__

    return dict(inspect.getmembers(module, predicate=is_local))


def _get_module_functions(module: ModuleType):
    """Get all functions defined locally inside a module."""
    return _get_module_members(module, typ=FunctionType)


def _get_fullname(typ) -> str:
    return f"{typ.__module__}.{typ.__qualname__}"


def make_all_tables():
    os.makedirs(CHARTS_IMAGE_DIR, exist_ok=True)
    LineStyleTable.generate()
    MarkerStyleTable.generate()
    ColorSchemeTable.generate()
    ColorTable.generate()
    DownloadsInfoTable.generate()
