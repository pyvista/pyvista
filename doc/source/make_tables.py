"""This is a helper module to generate tables that can be included in the documentation."""

# ruff: noqa: PTH102,PTH103,PTH107,PTH112,PTH113,PTH117,PTH118,PTH119,PTH122,PTH123,PTH202
import inspect
import io
import os
from pathlib import Path
import re
import textwrap
from types import FunctionType
from typing import Any, Callable, Dict, List, Type

import pyvista as pv
from pyvista.core.errors import VTKVersionError
from pyvista.examples import _dataset_loader, downloads

# Paths to directories in which resulting rst files and images are stored.
CHARTS_TABLE_DIR = "api/plotting/charts"
CHARTS_IMAGE_DIR = "images/charts"
COLORS_TABLE_DIR = "api/utilities"
DATASET_GALLERY_TABLE_DIR = "api/examples/dataset-gallery"
DATASET_GALLERY_IMAGE_DIR = "images/dataset-gallery"


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
        if Path(cls.path).exists():
            with Path(cls.path).open() as fold:
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


class DownloadsMetadataTable(DocTable):
    """Class to generate metadata about pyvista downloadable datasets.

    For each 'row' of this table:

    Create a card with a 1-column grid with two grid items (rows).
    The first row is a nested grid with two items displayed as a single
    column for small screen sizes, or two columns for larger screens.

    Each row has a structure similar to:

        Dataset Name
        +-Card---------------------+
        | Function Name            |
        | Docstring                |
        |  +-Grid---------------+  |
        |  |  +-Grid-+-------+  |  |
        |  |  | Info | Image |  |  |
        |  |  +------+-------+  |  |
        |  +--------------------+  |
        |  | Repr               |  |
        |  +--------------------+  |
        +--------------------------+

    See https://sphinx-design.readthedocs.io/en/latest/index.html for
    details on the directives used and their formatting.
    """

    NOT_AVAILABLE_IMG_PATH = os.path.join(DATASET_GALLERY_IMAGE_DIR, 'not_available.png')
    path = f"{DATASET_GALLERY_TABLE_DIR}/downloads_gallery_table.rst"

    # No main header; each row/dataset is a separate card
    header = ""
    row_template = _aligned_dedent(
        """
        |.. index:: {}
        |
        |.. _{}:
        |
        |{}
        |
        |.. card:: {}
        |
        |   {}
        |
        |   .. grid:: 1
        |
        |      .. grid-item::
        |
        |         .. grid:: 1 2 2 2
        |            :reverse:
        |
        |            .. grid-item::
        |
        |               .. grid:: 1
        |
        |                  .. grid-item-card::
        |
        |                     .. image:: /{}
        |
        |            .. grid-item::
        |
        |               - Size on Disk: {}
        |               - Num Files: {}
        |               - Extension: {}
        |               - Reader: {}
        |               - Dataset Type: {}
        |
        |      .. grid-item::
        |
        |         .. code-block::
        |            :caption: Representation
        |
        |            {}
        |
        |   .. dropdown:: :octicon:`globe` Data Source Links
        |
        |      {}
        |
        """
    )
    # Set the indentation level for the representation item
    # Level should be one more than the representation's '.. code-block::' directive in the
    # row template above
    REPR_INDENT_LEVEL = 4

    LINKS_INDENT_LEVEL = 2

    @classmethod
    def fetch_data(cls):
        # Collect all `_dataset_<name>` file loaders
        module_members: Dict[str, FunctionType] = dict(inspect.getmembers(pv.examples.downloads))
        file_loaders_dict = {
            name: item
            for name, item in module_members.items()
            if name.startswith('_dataset_')
            and isinstance(
                item,
                (
                    _dataset_loader._SingleFileDownloadable,
                    _dataset_loader._MultiFileDownloadableLoadable,
                ),
            )
        }
        return sorted(file_loaders_dict.items())

    @classmethod
    def get_header(cls, data):
        return cls.header

    @classmethod
    def get_row(cls, i, row_data):
        loader_name, loader = row_data

        # Get dataset name-related info
        index_name, ref_name, dataset_heading, func_ref, func_doc, func_name = (
            cls._format_dataset_name(loader_name)
        )

        # Get file and instance metadata
        try:
            loader.download()
        except VTKVersionError:
            # Set default values
            NOT_AVAILABLE = '``Not available``'
            NOT_AVAILABLE_NO_BACKTICKS = NOT_AVAILABLE.replace('`', '')
            file_size = NOT_AVAILABLE
            num_files = NOT_AVAILABLE
            file_ext = NOT_AVAILABLE
            reader_type = NOT_AVAILABLE
            dataset_type = NOT_AVAILABLE
            dataset_repr = NOT_AVAILABLE_NO_BACKTICKS
            datasource_links = NOT_AVAILABLE_NO_BACKTICKS
            img_path = cls.NOT_AVAILABLE_IMG_PATH
        else:
            # Get data from loader
            loader.load()
            file_size = cls._format_file_size(loader)
            num_files = cls._format_num_files(loader)
            file_ext = cls._format_file_ext(loader)
            reader_type = cls._format_reader_type(loader)
            dataset_type = cls._format_dataset_type(loader)
            dataset_repr = cls._format_dataset_repr(loader, cls.REPR_INDENT_LEVEL)
            datasource_links = cls._format_datasource_links(loader, cls.LINKS_INDENT_LEVEL)
            img_path = cls._search_image_path(func_name)

        cls._process_img(img_path)

        return cls.row_template.format(
            index_name,
            ref_name,
            dataset_heading,
            func_ref,
            func_doc,
            img_path,
            file_size,
            num_files,
            file_ext,
            reader_type,
            dataset_type,
            dataset_repr,
            datasource_links,
        )

    @staticmethod
    def _format_dataset_name(loader_name: str):
        # Extract data set name from loader name
        dataset_name = loader_name.replace('_dataset_', '')

        # Format dataset name for indexing and section heading
        index_name = dataset_name + '_dataset'
        ref_name = 'ref' + index_name
        dataset_heading = ' '.join([word.capitalize() for word in index_name.split('_')])
        dataset_heading += '\n' + _repeat_string('*', len(dataset_heading))

        # Get the corresponding 'download' function of the loader
        func_name = 'download_' + dataset_name
        func = getattr(downloads, func_name)

        # Get the card's header info
        func_ref = f':func:`~{_get_fullname(func)}`'
        func_doc = _get_doc(func)
        return index_name, ref_name, dataset_heading, func_ref, func_doc, func_name

    @staticmethod
    def _format_file_size(loader: _dataset_loader._FileProps):
        return '``' + loader.total_size + '``'

    @staticmethod
    def _format_num_files(loader: _dataset_loader._FileProps):
        return '``' + str(loader.num_files) + '``'

    @staticmethod
    def _format_file_ext(loader: _dataset_loader._FileProps):
        # Format extension as single str with rst backticks
        # Multiple extensions are comma-separated
        file_ext = loader.unique_extension
        file_ext = file_ext if isinstance(file_ext, str) else ' '.join(ext for ext in file_ext)
        file_ext = '``\'' + file_ext.replace(' ', '\'``, ``\'') + '\'``'
        return file_ext

    @staticmethod
    def _format_reader_type(loader: _dataset_loader._FileProps):
        """Format reader type(s) with doc references to reader class(es)."""
        reader_type = (
            repr(loader.unique_reader_type)
            .replace('<class \'', ':class:`~')
            .replace('\'>', '`')
            .replace('(', '')
            .replace(')', '')
        )
        return reader_type

    @staticmethod
    def _format_dataset_type(loader: _dataset_loader._FileProps):
        """Format dataset type(s) with doc references to dataset class(es)."""
        dataset_type = (
            repr(loader.unique_dataset_type)
            .replace('<class \'', ':class:`~')
            .replace('\'>', '`')
            .replace('(', '')
            .replace(')', '')
        )
        return dataset_type

    @staticmethod
    def _format_dataset_repr(loader: _dataset_loader._FileProps, indent_level: int) -> str:
        """Format the dataset's representation as a single multi-line string.

        The returned string is indented up to the specified indent level.
        """
        # Replace any hex code memory addresses with ellipses
        dataset_repr = repr(loader.dataset)
        dataset_repr = re.sub(
            pattern=r'0x[0-9a-f]*',
            repl='...',
            string=dataset_repr,
        )
        return _indent_multi_line_string(dataset_repr, indent_size=3, indent_level=indent_level)

    @staticmethod
    def _format_datasource_links(loader: _dataset_loader._Downloadable, indent_level: int) -> str:
        def _rst_link(url):
            return f'`{url} <{url}>`_'

        links = [url] if isinstance(url := loader.source_url, str) else url
        links = [_rst_link(url) for url in links]
        links = '\n'.join(links)
        return _indent_multi_line_string(links, indent_size=3, indent_level=indent_level)

    @staticmethod
    def _search_image_path(dataset_download_func_name: str):
        """Search the thumbnail directory and return its path.

        If no thumbnail is found, the path to a "not available" image is returned.
        """
        # Search directory and match:
        #     any word character(s), then function name, then any non-word character(s),
        #     then a 3character file extension, e.g.:
        #       'pyvista-examples...download_name...ext'
        #     or simply:
        #       'download_name.ext'
        all_filenames = '\n' + '\n'.join(os.listdir(DATASET_GALLERY_IMAGE_DIR)) + '\n'
        match = re.search(
            pattern=r'\n([\w|\-]*' + dataset_download_func_name + r'(\-\w*\.|\.)[a-z]{3})\n',
            string=all_filenames,
            flags=re.MULTILINE,
        )

        if match:
            groups = match.groups()
            assert (
                sum(dataset_download_func_name in grp for grp in groups) <= 1
            ), f"More than one thumbnail image was found for {dataset_download_func_name}, got:\n{groups}"
            img_fname = groups[0]
            img_path = os.path.join(DATASET_GALLERY_IMAGE_DIR, img_fname)
            assert os.path.isfile(img_path)
        else:
            print(f"WARNING: Missing thumbnail image file for \'{dataset_download_func_name}\'")
            img_path = os.path.join(DATASET_GALLERY_IMAGE_DIR, 'not_available.png')
        return img_path

    @staticmethod
    def _process_img(img_path):
        """Process the thumbnail image to ensure it's the right size."""
        from PIL import Image

        IMG_WIDTH, IMG_HEIGHT = 400, 300

        if os.path.basename(img_path) == 'not_available.png':
            not_available_mesh = pv.Text3D('Not Available')
            p = pv.Plotter(off_screen=True, window_size=(IMG_WIDTH, IMG_HEIGHT))
            p.background_color = 'white'
            p.add_mesh(not_available_mesh, color='black')
            p.view_xy()
            p.camera.up = (1, IMG_WIDTH / IMG_HEIGHT, 0)
            p.enable_parallel_projection()
            img_array = p.show(screenshot=True)

            # exit early if the image is the same
            if os.path.isfile(img_path) and pv.compare_images(img_path, img_path) < 1:
                return

            img = Image.fromarray(img_array)
            img.save(img_path)
        else:
            # Resize existing image if necessary
            img = Image.open(img_path)
            if img.width > IMG_WIDTH or img.height > IMG_HEIGHT:
                img.thumbnail(size=(IMG_WIDTH, IMG_HEIGHT))
                img.save(img_path)


def _get_doc(func: Callable[[], Any]) -> str:
    """Return the first line of the callable's docstring."""
    return func.__doc__.splitlines()[0]


def _get_fullname(typ: Type) -> str:
    """Return the fully qualified name of the given type object."""
    return f"{typ.__module__}.{typ.__qualname__}"


def _ljust_lines(lines: List[str], min_width=None) -> List[str]:
    """Left-justify a list of lines."""
    min_width = min_width if min_width else _max_width(lines)
    return [line.ljust(min_width) for line in lines]


def _max_width(lines: List[str]) -> int:
    """Compute the max line-width from a list of lines."""
    return max(map(len, lines))


def _repeat_string(string: str, num_repeat: int) -> str:
    """Repeat `string` `num_repeat` times."""
    return ''.join([string] * num_repeat)


def _pad_lines(
    lines: List[str], *, pad_left: str = '', pad_right: str = '', ljust=True, return_shape=False
):
    """Add padding to the left or right of each line with a specified string.

    By default, padding is only applied to left-justify the text such that the lines
    all have the same width.

    Optionally, the lines may be padded to the left or right using a specified string.

    Parameters
    ----------
    lines : list[str]
        Lines to be padded. If a tuple of lists is given, all lists are padded together
        as if they were one, but returned as separate lists.

    pad_left : str, default: ''
        String to pad the left of each line with.

    pad_right : str, default: ''
        String to pad the right of each line with.

    ljust : bool, default: True
        If ``True``, left-justify the lines such that they have equal width
        before applying any padding.

    return_shape : bool, default: False
        If ``True``, also return the width and height of the padded lines.

    """
    # Justify
    lines = _ljust_lines(lines) if ljust else lines
    # Pad
    lines = [pad_left + line + pad_right for line in lines]
    if return_shape:
        return lines, _max_width(lines), len(lines)
    return lines


def _indent_multi_line_string(
    string: str, indent_size=3, indent_level: int = 1, omit_first_line=True
) -> str:
    """Indent each line of a multi-line string by a specified indentation level.

    Optionally specify the indent size (e.g. 3 spaces for rst).
    Optionally omit indentation from the first line if it is already indented.

    This function is used to support de-denting formatted multi-line strings.
    E.g. for the following rst text with item {} indented by 3 levels:

        |      .. some_directive::
        |
        |         {}

    a multi-line string input such as 'line1\nline2\nline3' will be formatted as:

        |      .. some_directive::
        |
        |         line1\n         line2\n         line3
        |

    which will result in the correct indentation applied to all lines of the string.

    """
    lines = string.splitlines()
    indentation = _repeat_string(' ', num_repeat=indent_size * indent_level)
    first_line = lines.pop(0) if omit_first_line else ''
    lines = _pad_lines(lines, pad_left=indentation) if len(lines) > 0 else lines
    lines.insert(0, first_line)
    return '\n'.join(lines)


def make_all_tables():
    os.makedirs(CHARTS_IMAGE_DIR, exist_ok=True)
    os.makedirs(DATASET_GALLERY_TABLE_DIR, exist_ok=True)
    LineStyleTable.generate()
    MarkerStyleTable.generate()
    ColorSchemeTable.generate()
    ColorTable.generate()
    DownloadsMetadataTable.generate()
