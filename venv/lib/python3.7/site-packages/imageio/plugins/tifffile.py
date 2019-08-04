# -*- coding: utf-8 -*-
# imageio is distributed under the terms of the (new) BSD License.

""" Storage of image data in tiff format.
"""

from __future__ import absolute_import, print_function, division

import sys
import datetime

from .. import formats
from ..core import Format

import numpy as np

_tifffile = None  # Defer loading to lib() function.


def load_lib():
    if sys.version_info < (3,):
        try:
            import enum  # noqa - needs enum34
            import concurrent.futures  # noqa - needs futures
        except ImportError:
            raise ImportError(
                "The Imageio TIFF format has extra dependencies "
                "on Python 2.7. Install these using e.g. "
                '"pip install enum34 futures".'
            )

    global _tifffile
    try:
        import tifffile as _tifffile
    except ImportError:
        from . import _tifffile
    return _tifffile


TIFF_FORMATS = (".tif", ".tiff", ".stk", ".lsm")
WRITE_METADATA_KEYS = (
    "photometric",
    "planarconfig",
    "resolution",
    "description",
    "compress",
    "volume",
    "writeshape",
    "extratags",
    "datetime",
)
READ_METADATA_KEYS = (
    "planar_configuration",
    "is_fluoview",
    "is_nih",
    "is_contig",
    "is_micromanager",
    "is_ome",
    "is_lsm" "is_palette",
    "is_reduced",
    "is_rgb",
    "is_sgi",
    "is_shaped",
    "is_stk",
    "is_tiled",
    "is_mdgel" "resolution_unit",
    "compression",
    "is_mediacy",
    "orientation",
    "description",
    "description1",
    "is_imagej",
    "software",
)


class TiffFormat(Format):
    """ Provides support for a wide range of Tiff images.
    
    Images that contain multiple pages can be read using ``imageio.mimread()``
    to read the individual pages, or ``imageio.volread()`` to obtain a
    single (higher dimensional) array.

    Parameters for reading
    ----------------------
    offset : int
        Optional start position of embedded file. By default this is
        the current file position.
    size : int
        Optional size of embedded file. By default this is the number
        of bytes from the 'offset' to the end of the file.
    multifile : bool
        If True (default), series may include pages from multiple files.
        Currently applies to OME-TIFF only.
    multifile_close : bool
        If True (default), keep the handles of other files in multifile
        series closed. This is inefficient when few files refer to
        many pages. If False, the C runtime may run out of resources.

    Parameters for saving
    ---------------------
    bigtiff : bool
        If True, the BigTIFF format is used.
    byteorder : {'<', '>'}
        The endianness of the data in the file.
        By default this is the system's native byte order.
    software : str
        Name of the software used to create the image.
        Saved with the first page only.

    Metadata for reading
    --------------------
    planar_configuration : {'contig', 'planar'}
        Specifies if samples are stored contiguous or in separate planes.
        By default this setting is inferred from the data shape.
        'contig': last dimension contains samples.
        'planar': third last dimension contains samples.
    resolution_unit : (float, float) or ((int, int), (int, int))
        X and Y resolution in dots per inch as float or rational numbers.
    compression : int
        Values from 0 to 9 indicating the level of zlib compression.
        If 0, data is uncompressed.
    orientation : {'top_left', 'bottom_right', ...}
        Oriented of image array.
    is_rgb : bool
        True if page contains a RGB image.
    is_contig : bool
        True if page contains a contiguous image.
    is_tiled : bool
        True if page contains tiled image.
    is_palette : bool
        True if page contains a palette-colored image and not OME or STK.
    is_reduced : bool
        True if page is a reduced image of another image.
    is_shaped : bool
        True if page contains shape in image_description tag.
    is_fluoview : bool
        True if page contains FluoView MM_STAMP tag.
    is_nih : bool
        True if page contains NIH image header.
    is_micromanager : bool
        True if page contains Micro-Manager metadata.
    is_ome : bool
        True if page contains OME-XML in image_description tag.
    is_sgi : bool
        True if page contains SGI image and tile depth tags.
    is_stk : bool
        True if page contains UIC2Tag tag.
    is_mdgel : bool
        True if page contains md_file_tag tag.
    is_mediacy : bool
        True if page contains Media Cybernetics Id tag.
    is_stk : bool
        True if page contains UIC2Tag tag.
    is_lsm : bool
        True if page contains LSM CZ_LSM_INFO tag.
    description : str
        Image description
    description1 : str
        Additional description
    is_imagej : None or str
        ImageJ metadata
    software : str
        Software used to create the TIFF file
    datetime : datetime.datetime
        Creation date and time

    Metadata for writing
    --------------------
    photometric : {'minisblack', 'miniswhite', 'rgb'}
        The color space of the image data.
        By default this setting is inferred from the data shape.
    planarconfig : {'contig', 'planar'}
        Specifies if samples are stored contiguous or in separate planes.
        By default this setting is inferred from the data shape.
        'contig': last dimension contains samples.
        'planar': third last dimension contains samples.
    resolution : (float, float) or ((int, int), (int, int))
        X and Y resolution in dots per inch as float or rational numbers.
    description : str
        The subject of the image. Saved with the first page only.
    compress : int
        Values from 0 to 9 controlling the level of zlib compression.
        If 0, data are written uncompressed (default).
    volume : bool
        If True, volume data are stored in one tile (if applicable) using
        the SGI image_depth and tile_depth tags.
        Image width and depth must be multiple of 16.
        Few software can read this format, e.g. MeVisLab.
    writeshape : bool
        If True, write the data shape to the image_description tag
        if necessary and no other description is given.
    extratags: sequence of tuples
        Additional tags as [(code, dtype, count, value, writeonce)].

        code : int
            The TIFF tag Id.
        dtype : str
            Data type of items in 'value' in Python struct format.
            One of B, s, H, I, 2I, b, h, i, f, d, Q, or q.
        count : int
            Number of data values. Not used for string values.
        value : sequence
            'Count' values compatible with 'dtype'.
        writeonce : bool
            If True, the tag is written to the first page only.
    """

    def _can_read(self, request):
        # We support any kind of image data
        return request.extension in self.extensions

    def _can_write(self, request):
        # We support any kind of image data
        return request.extension in self.extensions

    # -- reader

    class Reader(Format.Reader):
        def _open(self, **kwargs):
            if not _tifffile:
                load_lib()
            # Allow loading from http; tiffile uses seek, so download first
            if self.request.filename.startswith(("http://", "https://")):
                self._f = f = open(self.request.get_local_filename(), "rb")
            else:
                self._f = None
                f = self.request.get_file()
            self._tf = _tifffile.TiffFile(f, **kwargs)

            # metadata is the same for all images
            self._meta = {}

        def _close(self):
            self._tf.close()
            if self._f is not None:
                self._f.close()

        def _get_length(self):
            if self.request.mode[1] in "vV":
                return 1  # or can there be pages in pages or something?
            else:
                return len(self._tf.pages)

        def _get_data(self, index):
            if self.request.mode[1] in "vV":
                # Read data as single 3D (+ color channels) array
                if index != 0:
                    raise IndexError('Tiff support no more than 1 "volume" per file')
                im = self._tf.asarray()  # request as singleton image
                meta = self._meta
            else:
                # Read as 2D image
                if index < 0 or index >= self._get_length():
                    raise IndexError("Index out of range while reading from tiff file")
                im = self._tf.pages[index].asarray()
                meta = self._meta or self._get_meta_data(index)
            # Return array and empty meta data
            return im, meta

        def _get_meta_data(self, index):
            page = self._tf.pages[index or 0]
            for key in READ_METADATA_KEYS:
                try:
                    self._meta[key] = getattr(page, key)
                except Exception:
                    pass

            # tifffile <= 0.12.1 use datetime, newer use DateTime
            for key in ("datetime", "DateTime"):
                try:
                    self._meta["datetime"] = datetime.datetime.strptime(
                        page.tags[key].value, "%Y:%m:%d %H:%M:%S"
                    )
                    break
                except Exception:
                    pass

            return self._meta

    # -- writer
    class Writer(Format.Writer):
        def _open(self, bigtiff=None, byteorder=None, software=None):
            if not _tifffile:
                load_lib()

            try:
                self._tf = _tifffile.TiffWriter(
                    self.request.get_file(), bigtiff, byteorder, software=software
                )
                self._software = None
            except TypeError:
                # In tifffile >= 0.15, the `software` arg is passed to
                # TiffWriter.save
                self._tf = _tifffile.TiffWriter(
                    self.request.get_file(), bigtiff, byteorder
                )
                self._software = software

            self._meta = {}

        def _close(self):
            self._tf.close()

        def _append_data(self, im, meta):
            if meta:
                self.set_meta_data(meta)
            # No need to check self.request.mode; tiffile figures out whether
            # this is a single page, or all page data at once.
            if self._software is None:
                self._tf.save(np.asanyarray(im), **self._meta)
            else:
                # tifffile >= 0.15
                self._tf.save(np.asanyarray(im), software=self._software, **self._meta)

        def set_meta_data(self, meta):
            self._meta = {}
            for (key, value) in meta.items():
                if key in WRITE_METADATA_KEYS:
                    self._meta[key] = value


# Register
format = TiffFormat("tiff", "TIFF format", TIFF_FORMATS, "iIvV")
formats.add_format(format)
