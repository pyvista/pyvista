# -*- coding: utf-8 -*-
# imageio is distributed under the terms of the (new) BSD License.

from __future__ import absolute_import, unicode_literals

from .tifffile import TiffFormat

from .. import formats


class FEISEMFormat(TiffFormat):
    """Provide read support for TIFFs produced by an FEI SEM microscope.

    This format is based on TIFF, and supports the same parameters.

    FEI microscopes append metadata as ASCII text at the end of the file,
    which this reader correctly extracts.

    Parameters for get_data
    -----------------------
    discard_watermark : bool
        If True (default), discard the bottom rows of the image, which
        contain no image data, only a watermark with metadata.
    watermark_height : int
        The height in pixels of the FEI watermark. The default is 70.
    """

    def _can_write(self, request):
        return False  # FEI-SEM only supports reading

    class Reader(TiffFormat.Reader):
        def _get_data(self, index=0, discard_watermark=True, watermark_height=70):
            """Get image and metadata from given index.

            FEI images usually (always?) contain a watermark at the
            bottom of the image, 70 pixels high. We discard this by
            default as it does not contain any information not present
            in the metadata.
            """
            im, meta = super(FEISEMFormat.Reader, self)._get_data(index)
            if discard_watermark:
                im = im[:-watermark_height]
            return im, meta

        def _get_meta_data(self, index=None):
            """Read the metadata from an FEI SEM TIFF.

            This metadata is included as ASCII text at the end of the file.

            The index, if provided, is ignored.

            Returns
            -------
            metadata : dict
                Dictionary of metadata.
            """
            md = {"root": {}}
            current_tag = "root"
            reading_metadata = False
            filename = self.request.get_local_filename()
            with open(filename, "rb") as fin:
                for line in fin:
                    if not reading_metadata:
                        if not line.startswith(b"Date="):
                            continue
                        else:
                            reading_metadata = True
                    line = line.rstrip().decode()
                    if line.startswith("["):
                        current_tag = line.lstrip("[").rstrip("]")
                        md[current_tag] = {}
                    else:
                        if line and line != "\x00":  # ignore blank lines
                            key, val = line.split("=")
                            for tag_type in (int, float):
                                try:
                                    val = tag_type(val)
                                except ValueError:
                                    continue
                                else:
                                    break
                            md[current_tag][key] = val
            if not md["root"] and len(md) == 1:
                raise ValueError("Input file %s contains no FEI metadata." % filename)
            self._meta.update(md)
            return md


# Register plugin
format = FEISEMFormat(
    "fei", "FEI-SEM TIFF format", extensions=[".tif", ".tiff"], modes="iv"
)
formats.add_format(format)
