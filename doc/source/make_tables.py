"""This is a helper module to generate tables that can be included in the documentation."""

import inspect
import io
import os
import re
import textwrap

import pyvista as pv
from pyvista.core.errors import VTKVersionError
from pyvista.examples import _example_loader, downloads

# Paths to directories in which resulting rst files and images are stored.
CHARTS_TABLE_DIR = "api/plotting/charts"
CHARTS_IMAGE_DIR = "images/charts"
COLORS_TABLE_DIR = "api/utilities"
DOWNLOADS_TABLE_DIR = "api/examples"
EXAMPLES_THUMBNAIL_IMAGES_DIR = "images/examples-thumbnails"
NOT_AVAILABLE_IMG_PATH = os.path.join(EXAMPLES_THUMBNAIL_IMAGES_DIR, 'not_available.png')


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


class DownloadsMetadataTable(DocTable):
    """Class to generate info about pyvista downloadable examples."""

    path = f"{DOWNLOADS_TABLE_DIR}/downloads_metadata_table.rst"

    # No hmain eader; each example is its own table
    header = ""
    row_template = _aligned_dedent(
        """
        |.. list-table:: :func:`~{}`
        |   :widths: 60 40
        |   :header-rows: 1
        |
        |   * - Info
        |     - Thumbnail
        |
        |   * - {}
        |
        |       - Size on Disk: {}
        |       - Num Files: {}
        |       - Extension: {}
        |       - Reader: {}
        |       - Dataset Type: {}
        |       - Representation: {}
        |
        |     - .. image:: /{}
        """
    )

    @classmethod
    def fetch_data(cls):

        # Collect all `_example_<name>` file loaders
        module_members = dict(inspect.getmembers(pv.examples.downloads))
        file_loaders_dict = {
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
        return sorted(file_loaders_dict.items())

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
            na = 'Not available'
            img_path = NOT_AVAILABLE_IMG_PATH
            (
                file_size_rst,
                num_files_rst,
                file_ext_rst,
                reader_type_rst,
                dataset_type_rst,
                repr_rst,
            ) = (na, na, na, na, na, na)
        else:
            # Get file info
            file_size_rst = '``' + loader.total_size + '``'
            num_files_rst = '``' + str(loader.num_files) + '``'
            # Format extension as single str with rst backticks
            file_ext = loader.unique_extension
            file_ext_str = (
                file_ext if isinstance(file_ext, str) else ' '.join(ext for ext in file_ext)
            )
            file_ext_rst = '``\'' + file_ext_str.replace(' ', '\'``, ``\'') + '\'``'
            # Format reader type as rst linked to class
            reader_type_rst = (
                repr(loader.unique_reader_type)
                .replace('<class \'', ':class:`~')
                .replace('\'>', '`')
                .replace('(', '')
                .replace(')', '')
            )

            # Get instance info
            dataset = loader.load()
            dataset_type_rst = (
                repr(loader.unique_dataset_type)
                .replace('<class \'', ':class:`~')
                .replace('\'>', '`')
                .replace('(', '')
                .replace(')', '')
            )
            # Format repr as literal block
            repr_rst = '::\n\n' + repr(dataset)
            repr_rst = repr_rst.replace('\n', '\n|           ')
            repr_rst = _aligned_dedent(repr_rst)
            # Replace any hex code memory addresses with ellipses
            repr_rst = re.sub(
                pattern=r'0x[0-9a-f]*',
                repl='...',
                string=repr_rst,
            )

            # Search for the file name from all images in the thumbnail directory.
            # Match:
            #     any word character(s), then function name, then any non-word character(s),
            #     then a 3 or 4 character file extension, e.g.:
            #       'pyvista-examples...download_name...ext'
            #     or simply:
            #       'download_name.ext'
            all_filenames = '\n' + '\n'.join(os.listdir(EXAMPLES_THUMBNAIL_IMAGES_DIR)) + '\n'
            match = re.search(
                pattern=r'\n([\w|\-]*' + download_name + r'(\-\w*\.|\.)[a-z]{3})\n',
                string=all_filenames,
                flags=re.MULTILINE,
            )

            if match:
                groups = match.groups()
                assert (
                    sum(download_name in grp for grp in groups) <= 1
                ), f"More than one thumbnail image was found for {download_name}, got:\n{groups}"
                img_fname = groups[0]
                img_path = os.path.join(EXAMPLES_THUMBNAIL_IMAGES_DIR, img_fname)
                assert os.path.isfile(img_path)
            else:
                print(f"WARNING: Missing thumbnail image file for \'{download_name}\'")
                img_path = os.path.join(EXAMPLES_THUMBNAIL_IMAGES_DIR, 'not_available.png')

        cls.process_img(img_path)

        return cls.row_template.format(
            func_fullname,
            doc_line,
            file_size_rst,
            num_files_rst,
            file_ext_rst,
            reader_type_rst,
            dataset_type_rst,
            repr_rst,
            img_path,
        )

    @staticmethod
    def process_img(img_path):
        """Process the thumbnail image to ensure it's the right size."""
        from PIL import Image

        IMG_WIDTH, IMG_HEIGHT = 300, 300

        if os.path.basename(img_path) == 'not_available.png':
            not_available_mesh = pv.Text3D('Not Available')
            p = pv.Plotter(off_screen=True, window_size=(IMG_WIDTH, IMG_HEIGHT))
            p.background_color = 'white'
            p.add_mesh(not_available_mesh, color='black')
            p.view_xy()
            p.camera.up = (1, 1, 0)
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


def _get_fullname(typ) -> str:
    return f"{typ.__module__}.{typ.__qualname__}"


def make_all_tables():
    os.makedirs(CHARTS_IMAGE_DIR, exist_ok=True)
    LineStyleTable.generate()
    MarkerStyleTable.generate()
    ColorSchemeTable.generate()
    ColorTable.generate()
    DownloadsMetadataTable.generate()
