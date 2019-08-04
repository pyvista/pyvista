# -*- coding: utf-8 -*-
# imageio is distributed under the terms of the (new) BSD License.

""" Plugin for reading videos via AvBin

Would be nice if we could capture webcam with this, but unfortunately,
avbin does not currently support this.
"""

from __future__ import absolute_import, print_function, division

import numpy as np
import logging
import ctypes
import sys
import os

from .. import formats
from ..core import (
    Format,
    get_platform,
    get_remote_file,
    InternetNotAllowedError,
    NeedDownloadError,
)


logger = logging.getLogger(__name__)


FNAME_PER_PLATFORM = {
    "osx64": "libavbin-10-osx.dylib",
    "win32": "avbin-10-win32.dll",
    "win64": "avbin-10-win64.dll",
    "linux32": "libavbin-10-linux32.so",
    "linux64": "libavbin-10-linux64.so",
}


AVBIN_RESULT_ERROR = -1
AVBIN_RESULT_OK = 0
# AVbinResult = ctypes.c_int


def AVbinResult(x):
    if x != AVBIN_RESULT_OK:
        raise RuntimeError("AVBin returned error code %i" % x)
    return x


AVBIN_STREAM_TYPE_UNKNOWN = 0
AVBIN_STREAM_TYPE_VIDEO = 1
AVBIN_STREAM_TYPE_AUDIO = 2
AVbinStreamType = ctypes.c_int

AVBIN_SAMPLE_FORMAT_U8 = 0
AVBIN_SAMPLE_FORMAT_S16 = 1
AVBIN_SAMPLE_FORMAT_S24 = 2
AVBIN_SAMPLE_FORMAT_S32 = 3
AVBIN_SAMPLE_FORMAT_FLOAT = 4
AVbinSampleFormat = ctypes.c_int

AVBIN_LOG_QUIET = -8
AVBIN_LOG_PANIC = 0
AVBIN_LOG_FATAL = 8
AVBIN_LOG_ERROR = 16
AVBIN_LOG_WARNING = 24
AVBIN_LOG_INFO = 32
AVBIN_LOG_VERBOSE = 40
AVBIN_LOG_DEBUG = 48
AVbinLogLevel = ctypes.c_int

AVbinFileP = ctypes.c_void_p
AVbinStreamP = ctypes.c_void_p

Timestamp = ctypes.c_int64


class AVbinFileInfo(ctypes.Structure):
    _fields_ = [
        ("structure_size", ctypes.c_size_t),
        ("n_streams", ctypes.c_int),
        ("start_time", Timestamp),
        ("duration", Timestamp),
        ("title", ctypes.c_char * 512),
        ("author", ctypes.c_char * 512),
        ("copyright", ctypes.c_char * 512),
        ("comment", ctypes.c_char * 512),
        ("album", ctypes.c_char * 512),
        ("year", ctypes.c_int),
        ("track", ctypes.c_int),
        ("genre", ctypes.c_char * 32),
    ]


class _AVbinStreamInfoVideo8(ctypes.Structure):
    _fields_ = [
        ("width", ctypes.c_uint),
        ("height", ctypes.c_uint),
        ("sample_aspect_num", ctypes.c_uint),
        ("sample_aspect_den", ctypes.c_uint),
        ("frame_rate_num", ctypes.c_uint),
        ("frame_rate_den", ctypes.c_uint),
    ]


class _AVbinStreamInfoAudio8(ctypes.Structure):
    _fields_ = [
        ("sample_format", ctypes.c_int),
        ("sample_rate", ctypes.c_uint),
        ("sample_bits", ctypes.c_uint),
        ("channels", ctypes.c_uint),
    ]


class _AVbinStreamInfoUnion8(ctypes.Union):
    _fields_ = [("video", _AVbinStreamInfoVideo8), ("audio", _AVbinStreamInfoAudio8)]


class AVbinStreamInfo8(ctypes.Structure):
    _fields_ = [
        ("structure_size", ctypes.c_size_t),
        ("type", ctypes.c_int),
        ("u", _AVbinStreamInfoUnion8),
    ]


class AVbinPacket(ctypes.Structure):
    _fields_ = [
        ("structure_size", ctypes.c_size_t),
        ("timestamp", Timestamp),
        ("stream_index", ctypes.c_int),
        ("data", ctypes.POINTER(ctypes.c_uint8)),
        ("size", ctypes.c_size_t),
    ]


AVbinLogCallback = ctypes.CFUNCTYPE(
    None, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p
)


def timestamp_from_avbin(timestamp):
    return float(timestamp) / 1000000


def download(directory=None, force_download=False):
    """ Download the avbin library to your computer.

    Parameters
    ----------
    directory : str | None
        The directory where the file will be cached if a download was
        required to obtain the file. By default, the appdata directory
        is used. This is also the first directory that is checked for
        a local version of the file.
    force_download : bool | str
        If True, the file will be downloaded even if a local copy exists
        (and this copy will be overwritten). Can also be a YYYY-MM-DD date
        to ensure a file is up-to-date (modified date of a file on disk,
        if present, is checked).
    """
    plat = get_platform()
    if not (plat and plat in FNAME_PER_PLATFORM):
        raise RuntimeError("AVBIN lib is not available for platform %s" % plat)
    fname = "avbin/" + FNAME_PER_PLATFORM[plat]
    get_remote_file(fname=fname, directory=directory, force_download=force_download)


def get_avbin_lib():
    """ Get avbin .dll/.dylib/.so
    """

    lib = os.getenv("IMAGEIO_AVBIN_LIB", None)
    if lib:  # pragma: no cover
        return lib

    platform = get_platform()

    try:
        lib = FNAME_PER_PLATFORM[platform]
    except KeyError:  # pragma: no cover
        raise RuntimeError("Avbin plugin is not supported on platform %s" % platform)

    try:
        return get_remote_file("avbin/" + lib, auto=False)
    except NeedDownloadError:
        raise NeedDownloadError(
            "Need avbin library. "
            "You can obtain it with either:\n"
            "  - download using the command: "
            "imageio_download_bin avbin\n"
            "  - download by calling (in Python): "
            "imageio.plugins.avbin.download()\n"
        )
    except InternetNotAllowedError as err:
        raise IOError("Could not download avbin lib:\n%s" % str(err))
        # in this case we raise. Can we try finding the system lib?


class AvBinFormat(Format):
    """ 
    The AvBinFormat uses the AvBin library (based on libav) to read
    video files.
    
    This plugin is more efficient than the ffmpeg plugin, because it
    uses ctypes (rather than a pipe like the ffmpeg plugin does).
    Further, it supports reading images into a given numpy array.
    
    The limitations of this plugin are that seeking, writing and camera
    feeds are not supported. See the ffmpeg format for these features.

    The avbin plugin requires an `avbin` binary. If this binary is
    not available on the system, it can be downloaded manually from
    <https://github.com/imageio/imageio-binaries> by either
    
    - the command line script ``imageio_download_bin avbin``
    - the Python method ``imageio.plugins.avbin.download()``.

    Parameters for reading
    ----------------------
    loop : bool
        If True, the video will rewind as soon as a frame is requested
        beyond the last frame. Otherwise, IndexError is raised. Default False.
    stream : int
        Specifies which video stream to read. Default 0.
    videoformat : str | None
        Specifies the video format (e.g. 'avi', or 'mp4'). If this is None
        (default) the format is auto-detected.
    
    Parameters for get_data
    -----------------------
    out : np.ndarray
        destination for the data retrieved. This can be used to save 
        time-consuming memory allocations when reading multiple image
        sequntially. The shape of out must be (width, height, 3), the
        dtype must be np.uint8 and it must be C-contiguous.
        
        Use the create_empty_image() method of the reader object
        to create an array that is suitable for get_data.
    """

    def __init__(self, *args, **kwargs):
        self._avbin = None
        Format.__init__(self, *args, **kwargs)

    def _can_read(self, request):
        if request.mode[1] in (self.modes + "?"):
            if request.extension in self.extensions:
                return True

    def _can_write(self, request):
        return False  # AvBin does not support writing videos

    def avbinlib(self, libpath=None):
        if self._avbin is not None and libpath is None:
            # Already loaded
            return self._avbin

        logger.warning(
            "The imageio avbin plugin is deprecated and will be removed in "
            "a future version. Please use ffmpeg instead."
        )

        if libpath is None:
            libpath = get_avbin_lib()

        self._avbin = avbin = ctypes.cdll.LoadLibrary(libpath)

        avbin.avbin_get_version.restype = ctypes.c_int
        avbin.avbin_get_ffmpeg_revision.restype = ctypes.c_int
        avbin.avbin_get_audio_buffer_size.restype = ctypes.c_size_t
        avbin.avbin_have_feature.restype = ctypes.c_int
        avbin.avbin_have_feature.argtypes = [ctypes.c_char_p]

        avbin.avbin_init.restype = AVbinResult
        avbin.avbin_set_log_level.restype = AVbinResult
        avbin.avbin_set_log_level.argtypes = [AVbinLogLevel]
        avbin.avbin_set_log_callback.argtypes = [AVbinLogCallback]

        avbin.avbin_open_filename.restype = AVbinFileP
        avbin.avbin_open_filename.argtypes = [ctypes.c_char_p]
        avbin.avbin_open_filename_with_format.restype = AVbinFileP
        avbin.avbin_open_filename_with_format.argtypes = [
            ctypes.c_char_p,
            ctypes.c_char_p,
        ]
        avbin.avbin_close_file.argtypes = [AVbinFileP]
        avbin.avbin_seek_file.argtypes = [AVbinFileP, Timestamp]
        avbin.avbin_file_info.argtypes = [AVbinFileP, ctypes.POINTER(AVbinFileInfo)]
        avbin.avbin_stream_info.argtypes = [
            AVbinFileP,
            ctypes.c_int,
            ctypes.POINTER(AVbinStreamInfo8),
        ]

        avbin.avbin_open_stream.restype = ctypes.c_void_p
        avbin.avbin_open_stream.argtypes = [AVbinFileP, ctypes.c_int]
        avbin.avbin_close_stream.argtypes = [AVbinStreamP]

        avbin.avbin_read.argtypes = [AVbinFileP, ctypes.POINTER(AVbinPacket)]
        avbin.avbin_read.restype = AVbinResult
        avbin.avbin_decode_audio.restype = ctypes.c_int
        avbin.avbin_decode_audio.argtypes = [
            AVbinStreamP,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int),
        ]
        avbin.avbin_decode_video.restype = ctypes.c_int
        avbin.avbin_decode_video.argtypes = [
            AVbinStreamP,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_void_p,
        ]

        avbin.avbin_init()
        avbin.avbin_set_log_level(AVBIN_LOG_QUIET)

        return self._avbin

    # -- reader

    class Reader(Format.Reader):
        def _open(self, loop=False, stream=0, videoformat=None, skipempty=False):

            # Init args
            self._arg_loop = bool(loop)
            self._arg_stream = int(stream)
            self._arg_videoformat = videoformat
            self._arg_skipempty = bool(skipempty)

            # Init other variables
            self._filename = self.request.get_local_filename()
            self._file = None
            self._meta = {"plugin": "avbin"}

            self._init_video()

        def _init_video(self):

            avbin = self.format.avbinlib()
            filename_bytes = self._filename.encode(sys.getfilesystemencoding())

            # Open file
            if self._arg_videoformat is not None:
                self._file = avbin.avbin_open_filename_with_format(
                    filename_bytes, self._arg_videoformat.encode("ascii")
                )
            else:
                self._file = avbin.avbin_open_filename(filename_bytes)
            if not self._file:
                raise IOError('Could not open "%s"' % self._filename)

            # Get info
            self._info = AVbinFileInfo()
            self._info.structure_size = ctypes.sizeof(self._info)
            avbin.avbin_file_info(self._file, ctypes.byref(self._info))

            # Store some info in meta dict
            self._meta["avbin_version"] = str(avbin.avbin_get_version())
            self._meta["title"] = self._info.title.decode("utf-8")
            self._meta["author"] = self._info.author.decode("utf-8")
            # The reported duration is different from what we get from ffmpeg,
            # and using it as is will yielf a wrong nframes. We correct below
            self._meta["duration"] = timestamp_from_avbin(self._info.duration)

            # Parse through the available streams in the file and find
            # the video stream specified by stream

            video_stream_counter = 0

            for i in range(self._info.n_streams):
                info = AVbinStreamInfo8()
                info.structure_size = ctypes.sizeof(info)
                avbin.avbin_stream_info(self._file, i, info)

                if info.type != AVBIN_STREAM_TYPE_VIDEO:
                    continue

                if video_stream_counter != self._arg_stream:
                    video_stream_counter += 1
                    continue

                # We have the n-th (n=stream number specified) video stream
                self._stream = avbin.avbin_open_stream(self._file, i)

                # Store info specific to this stream
                self._stream_info = info
                self._width = info.u.video.width
                self._height = info.u.video.height
                # Store meta info
                self._meta["size"] = self._width, self._height
                self._meta["source_size"] = self._width, self._height
                self._meta["fps"] = float(info.u.video.frame_rate_num) / float(
                    info.u.video.frame_rate_den
                )
                self._meta["duration"] -= 1.0 / self._meta["fps"]  # correct
                self._meta["nframes"] = int(self._meta["duration"] * self._meta["fps"])

                self._stream_index = i
                break
            else:
                raise IOError(
                    "Stream #%d not found in %r" % (self._arg_stream, self._filename)
                )

            self._packet = AVbinPacket()
            self._packet.structure_size = ctypes.sizeof(self._packet)

            self._framecounter = 0

        def _close(self):
            if self._file is not None:
                avbin = self.format.avbinlib()
                avbin.avbin_close_file(self._file)
                self._file = None

        def _get_length(self):
            # Return the number of images. Can be np.inf
            # Note that nframes is an estimate that can be a few frames off
            # for very large video files
            return self._meta["nframes"]

        def create_empty_image(self):
            return np.zeros((self._height, self._width, 3), dtype=np.uint8)

        def _get_data(self, index, out=None):
            avbin = self.format.avbinlib()

            # Modulo index (for looping)
            if self._meta["nframes"] and self._meta["nframes"] < float("inf"):
                if self._arg_loop:
                    index = index % self._meta["nframes"]

            # Check index
            if index < 0:
                raise IndexError("Frame index must be > 0")
            elif index >= self._meta["nframes"]:
                raise IndexError("Reached end of video")
            elif index != self._framecounter:
                if index == 0:  # Rewind
                    self._close()
                    self._init_video()
                    return self._get_data(0)
                raise IndexError("Avbin format cannot seek")

            self._framecounter += 1

            if out is None:
                out = self.create_empty_image()

            assert (
                out.dtype == np.uint8
                and out.flags.c_contiguous
                and out.shape == (self._height, self._width, 3)
            )

            # Read from the file until the next packet of our video
            # stream is found
            while True:
                try:
                    avbin.avbin_read(self._file, ctypes.byref(self._packet))
                except RuntimeError:  # todo: I hope we can fix this ...
                    raise IndexError("Reached end of video too soon")
                if self._packet.stream_index != self._stream_index:
                    continue

                # Decode the image, storing data in the out array
                try:
                    ptr = out.ctypes.data
                except Exception:  # pragma: no cover - IS_PYPY
                    ptr = out.__array_interface__["data"][0]
                result = avbin.avbin_decode_video(
                    self._stream, self._packet.data, self._packet.size, ptr
                )

                # Check for success. If not, continue reading the file stream
                # AK: disabled for now, because this will make the file
                # shorter; you're just dropping frames! We need to think
                # of a better solution ...
                if (not self._arg_skipempty) or result != -1:
                    break

            # Return array and dummy meta data
            return out, dict(timestamp=self._packet.timestamp)

        def _get_meta_data(self, index):
            return self._meta


# Register. You register an *instance* of a Format class. Here specify:
format = AvBinFormat(
    "avbin",  # short name
    "Many video formats (via AvBin, i.e. libav library)",
    "mov avi mp4 mpg mpeg mkv",  # list of extensions
    "I",  # modes, characters in iIvV
)
formats.add_format(format)
