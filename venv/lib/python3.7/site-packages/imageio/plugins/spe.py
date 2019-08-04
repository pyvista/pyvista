# -*- coding: utf-8 -*-
# imageio is distributed under the terms of the (new) BSD License.

""" SPE file reader
"""

from __future__ import absolute_import, print_function, division

import os
import logging

import numpy as np

from .. import formats
from ..core import Format


logger = logging.getLogger(__name__)


class Spec:
    """SPE file specification data

    Tuples of (offset, datatype, count), where offset is the offset in the SPE
    file and datatype is the datatype as used in `numpy.fromfile`()

    `data_start` is the offset of actual image data.

    `dtypes` translates SPE datatypes (0...4) to numpy ones, e. g. dtypes[0]
    is dtype("<f") (which is np.float32).

    `controllers` maps the `type` metadata to a human readable name

    `readout_modes` maps the `readoutMode` metadata to something human readable
    although this may not be accurate since there is next to no documentation
    to be found.
    """

    basic = {
        "datatype": (108, "<h"),  # dtypes
        "xdim": (42, "<H"),
        "ydim": (656, "<H"),
        "NumFrames": (1446, "<i"),
    }

    metadata = {
        # ROI information
        "NumROI": (1510, "<h"),
        "ROIs": (
            1512,
            np.dtype(
                [
                    ("startx", "<H"),
                    ("endx", "<H"),
                    ("groupx", "<H"),
                    ("starty", "<H"),
                    ("endy", "<H"),
                    ("groupy", "<H"),
                ]
            ),
            10,
        ),
        # chip-related sizes
        "xDimDet": (6, "<H"),
        "yDimDet": (18, "<H"),
        "VChipXdim": (14, "<h"),
        "VChipYdim": (16, "<h"),
        # other stuff
        "controller_version": (0, "<h"),
        "logic_output": (2, "<h"),
        "amp_high_cap_low_noise": (4, "<H"),  # enum?
        "mode": (8, "<h"),  # enum?
        "exposure_sec": (10, "<f"),
        "date": (20, "<10S"),
        "detector_temp": (36, "<f"),
        "detector_type": (40, "<h"),
        "st_diode": (44, "<h"),
        "delay_time": (46, "<f"),
        # shutter_control: normal, disabled open, disabled closed
        # But which one is which?
        "shutter_control": (50, "<H"),
        "absorb_live": (52, "<h"),
        "absorb_mode": (54, "<H"),
        "can_do_virtual_chip": (56, "<h"),
        "threshold_min_live": (58, "<h"),
        "threshold_min_val": (60, "<f"),
        "threshold_max_live": (64, "<h"),
        "threshold_max_val": (66, "<f"),
        "time_local": (172, "<7S"),
        "time_utc": (179, "<7S"),
        "adc_offset": (188, "<H"),
        "adc_rate": (190, "<H"),
        "adc_type": (192, "<H"),
        "adc_resolution": (194, "<H"),
        "adc_bit_adjust": (196, "<H"),
        "gain": (198, "<H"),
        "comments": (200, "<80S", 5),
        "geometric": (600, "<H"),  # flags
        "sw_version": (688, "<16S"),
        "spare_4": (742, "<436S"),
        "XPrePixels": (98, "<h"),
        "XPostPixels": (100, "<h"),
        "YPrePixels": (102, "<h"),
        "YPostPixels": (104, "<h"),
        "readout_time": (672, "<f"),
        "type": (704, "<h"),  # controllers
        "clockspeed_us": (1428, "<f"),
        "readout_mode": (1480, "<H"),  # readout_modes
        "window_size": (1482, "<H"),
        "file_header_ver": (1992, "<f"),
    }

    data_start = 4100

    dtypes = [
        np.dtype("<f"),
        np.dtype("<i"),
        np.dtype("<h"),
        np.dtype("<H"),
        np.dtype("<I"),
    ]

    controllers = [
        "new120 (Type II)",
        "old120 (Type I)",
        "ST130",
        "ST121",
        "ST138",
        "DC131 (PentaMax)",
        "ST133 (MicroMax/Roper)",
        "ST135 (GPIB)",
        "VTCCD",
        "ST116 (GPIB)",
        "OMA3 (GPIB)",
        "OMA4",
    ]

    # This was gathered from random places on the internet and own experiments
    # with the camera. May not be accurate.
    readout_modes = ["full frame", "frame transfer", "kinetics"]

    # Do not decode the following metadata keys into strings, but leave them
    # as byte arrays
    no_decode = ["spare_4"]


class SpeFormat(Format):
    """ Some CCD camera software produces images in the Princeton Instruments
    SPE file format. This plugin supports reading such files.

    Parameters for reading
    ----------------------
    char_encoding : str
        Character encoding used to decode strings in the metadata. Defaults
        to "latin1".
    check_filesize : bool
        The number of frames in the file is stored in the file header. However,
        this number may be wrong for certain software. If this is `True`
        (default), derive the number of frames also from the file size and
        raise a warning if the two values do not match.

    Metadata for reading
    --------------------
    ROIs : list of dict
        Regions of interest used for recording images. Each dict has the
        "top_left" key containing x and y coordinates of the top left corner,
        the "bottom_right" key with x and y coordinates of the bottom right
        corner, and the "bin" key with number of binned pixels in x and y
        directions.
    comments : list of str
        The SPE format allows for 5 comment strings of 80 characters each.
    controller_version : int
        Hardware version
    logic_output : int
        Definition of output BNC
    amp_hi_cap_low_noise : int
        Amp switching mode
    mode : int
        Timing mode
    exp_sec : float
        Alternative exposure in seconds
    date : str
        Date string
    detector_temp : float
        Detector temperature
    detector_type : int
        CCD / diode array type
    st_diode : int
        Trigger diode
    delay_time : float
        Used with async mode
    shutter_control : int
        Normal, disabled open, or disabled closed
    absorb_live : bool
        on / off
    absorb_mode : int
        Reference strip or file
    can_do_virtual_chip : bool
        True or False whether chip can do virtual chip
    threshold_min_live : bool
        on / off
    threshold_min_val : float
        Threshold minimum value
    threshold_max_live : bool
        on / off
    threshold_max_val : float
        Threshold maximum value
    time_local : str
        Experiment local time
    time_utc : str
        Experiment UTC time
    adc_offset : int
        ADC offset
    adc_rate : int
        ADC rate
    adc_type : int
        ADC type
    adc_resolution : int
        ADC resolution
    adc_bit_adjust : int
        ADC bit adjust
    gain : int
        gain
    sw_version : str
        Version of software which created this file
    spare_4 : bytes
        Reserved space
    readout_time : float
        Experiment readout time
    type : str
        Controller type
    clockspeed_us : float
        Vertical clock speed in microseconds
    readout_mode : {"full frame", "frame transfer", "kinetics", ""}
        Readout mode. Empty string means that this was not set by the
        Software.
    window_size : int
        Window size for Kinetics mode
    file_header_ver : float
        File header version
    chip_size : [int, int]
        x and y dimensions of the camera chip
    virt_chip_size : [int, int]
        Virtual chip x and y dimensions
    pre_pixels : [int, int]
        Pre pixels in x and y dimensions
    post_pixels : [int, int],
        Post pixels in x and y dimensions
    geometric : list of {"rotate", "reverse", "flip"}
        Geometric operations
    """

    def _can_read(self, request):
        return (
            request.mode[1] in self.modes + "?" and request.extension in self.extensions
        )

    def _can_write(self, request):
        return False

    class Reader(Format.Reader):
        def _open(self, char_encoding="latin1", check_filesize=True):
            self._file = self.request.get_file()
            self._char_encoding = char_encoding

            info = self._parse_header(Spec.basic)
            self._dtype = Spec.dtypes[info["datatype"]]
            self._shape = (info["ydim"], info["xdim"])
            self._len = info["NumFrames"]

            if check_filesize:
                # Some software writes incorrecet `NumFrames` metadata
                # Use the file size to determine the number of frames
                fsz = os.path.getsize(self.request.get_local_filename())
                l = fsz - Spec.data_start
                l //= self._shape[0] * self._shape[1] * self._dtype.itemsize
                if l != self._len:
                    logger.warning(
                        "Number of frames according to file header "
                        "does not match the size of file %s." % self.request.filename
                    )
                    self._len = min(l, self._len)

            self._meta = None

        def _get_meta_data(self, index):
            if self._meta is None:
                self._meta = self._parse_header(Spec.metadata)

                nr = self._meta.pop("NumROI", None)
                nr = 1 if nr < 1 else nr
                self._meta["ROIs"] = roi_array_to_dict(self._meta["ROIs"][:nr])

                # chip sizes
                self._meta["chip_size"] = [
                    self._meta.pop("xDimDet", None),
                    self._meta.pop("yDimDet", None),
                ]
                self._meta["virt_chip_size"] = [
                    self._meta.pop("VChipXdim", None),
                    self._meta.pop("VChipYdim", None),
                ]
                self._meta["pre_pixels"] = [
                    self._meta.pop("XPrePixels", None),
                    self._meta.pop("YPrePixels", None),
                ]
                self._meta["post_pixels"] = [
                    self._meta.pop("XPostPixels", None),
                    self._meta.pop("YPostPixels", None),
                ]

                # comments
                self._meta["comments"] = [str(c) for c in self._meta["comments"]]

                # geometric operations
                g = []
                f = self._meta.pop("geometric", 0)
                if f & 1:
                    g.append("rotate")
                if f & 2:
                    g.append("reverse")
                if f & 4:
                    g.append("flip")
                self._meta["geometric"] = g

                # Make some additional information more human-readable
                t = self._meta["type"]
                if 1 <= t <= len(Spec.controllers):
                    self._meta["type"] = Spec.controllers[t - 1]
                else:
                    self._meta["type"] = ""
                m = self._meta["readout_mode"]
                if 1 <= m <= len(Spec.readout_modes):
                    self._meta["readout_mode"] = Spec.readout_modes[m - 1]
                else:
                    self._meta["readout_mode"] = ""

                # bools
                for k in (
                    "absorb_live",
                    "can_do_virtual_chip",
                    "threshold_min_live",
                    "threshold_max_live",
                ):
                    self._meta[k] = bool(self._meta[k])

                # frame shape
                self._meta["frame_shape"] = self._shape
            return self._meta

        def _close(self):
            # The file should be closed by `self.request`
            pass

        def _parse_header(self, spec):
            ret = {}
            # Decode each string from the numpy array read by np.fromfile
            decode = np.vectorize(lambda x: x.decode(self._char_encoding))

            for name, sp in spec.items():
                self._file.seek(sp[0])
                cnt = 1 if len(sp) < 3 else sp[2]
                v = np.fromfile(self._file, dtype=sp[1], count=cnt)
                if v.dtype.kind == "S" and name not in Spec.no_decode:
                    # Silently ignore string decoding failures
                    try:
                        v = decode(v)
                    except Exception:
                        logger.warning(
                            'Failed to decode "{}" metadata '
                            "string. Check `char_encoding` "
                            "parameter.".format(name)
                        )

                try:
                    # For convenience, if the array contains only one single
                    # entry, return this entry itself.
                    v = np.asscalar(v)
                except ValueError:
                    v = np.squeeze(v)
                ret[name] = v
            return ret

        def _get_length(self):
            if self.request.mode[1] in "vV":
                return 1
            else:
                return self._len

        def _get_data(self, index):
            if index < 0:
                raise IndexError("Image index %i < 0" % index)
            if index >= self._len:
                raise IndexError("Image index %i > %i" % (index, self._len))

            if self.request.mode[1] in "vV":
                if index != 0:
                    raise IndexError("Index has to be 0 in v and V modes")
                self._file.seek(Spec.data_start)
                data = np.fromfile(
                    self._file,
                    dtype=self._dtype,
                    count=self._shape[0] * self._shape[1] * self._len,
                )
                data = data.reshape((self._len,) + self._shape)
            else:
                self._file.seek(
                    Spec.data_start
                    + index * self._shape[0] * self._shape[1] * self._dtype.itemsize
                )
                data = np.fromfile(
                    self._file, dtype=self._dtype, count=self._shape[0] * self._shape[1]
                )
                data = data.reshape(self._shape)
            return data, self._get_meta_data(index)


def roi_array_to_dict(a):
    """Convert the `ROIs` structured arrays to :py:class:`dict`

    Parameters
    ----------
    a : numpy.ndarray
        Structured array containing ROI data

    Returns
    -------
    list of dict
        One dict per ROI. Keys are "top_left", "bottom_right", and "bin",
        values are tuples whose first element is the x axis value and the
        second element is the y axis value.
    """
    l = []
    a = a[["startx", "starty", "endx", "endy", "groupx", "groupy"]]
    for sx, sy, ex, ey, gx, gy in a:
        d = {
            "top_left": [int(sx), int(sy)],
            "bottom_right": [int(ex), int(ey)],
            "bin": [int(gx), int(gy)],
        }
        l.append(d)
    return l


fmt = SpeFormat("spe", "SPE file format", ".spe", "iIvV")
formats.add_format(fmt, overwrite=True)
