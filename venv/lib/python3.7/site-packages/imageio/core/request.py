# -*- coding: utf-8 -*-
# imageio is distributed under the terms of the (new) BSD License.

"""
Definition of the Request object, which acts as a kind of bridge between
what the user wants and what the plugins can.
"""

from __future__ import absolute_import, print_function, division

import sys
import os
from io import BytesIO
import zipfile
import tempfile
import shutil

from ..core import string_types, binary_type, urlopen, get_remote_file

if sys.version_info < (3,):
    FileNotFoundError = OSError
try:
    from pathlib import Path
except ImportError:
    Path = None

# URI types
URI_BYTES = 1
URI_FILE = 2
URI_FILENAME = 3
URI_ZIPPED = 4
URI_HTTP = 5
URI_FTP = 6

SPECIAL_READ_URIS = "<video", "<screen>", "<clipboard>"

# The user can use this string in a write call to get the data back as bytes.
RETURN_BYTES = "<bytes>"

# Example images that will be auto-downloaded
EXAMPLE_IMAGES = {
    "astronaut.png": "Image of the astronaut Eileen Collins",
    "camera.png": "Classic grayscale image of a photographer",
    "checkerboard.png": "Black and white image of a chekerboard",
    "clock.png": "Photo of a clock with motion blur (Stefan van der Walt)",
    "coffee.png": "Image of a cup of coffee (Rachel Michetti)",
    "chelsea.png": "Image of Stefan's cat",
    "wikkie.png": "Image of Almar's cat",
    "coins.png": "Image showing greek coins from Pompeii",
    "horse.png": "Image showing the silhouette of a horse (Andreas Preuss)",
    "hubble_deep_field.png": "Photograph taken by Hubble telescope (NASA)",
    "immunohistochemistry.png": "Immunohistochemical (IHC) staining",
    "moon.png": "Image showing a portion of the surface of the moon",
    "page.png": "A scanned page of text",
    "text.png": "A photograph of handdrawn text",
    "chelsea.zip": "The chelsea.png in a zipfile (for testing)",
    "chelsea.bsdf": "The chelsea.png in a BSDF file(for testing)",
    "newtonscradle.gif": "Animated GIF of a newton's cradle",
    "cockatoo.mp4": "Video file of a cockatoo",
    "stent.npz": "Volumetric image showing a stented abdominal aorta",
}


class Request(object):
    """ Request(uri, mode, **kwargs)

    Represents a request for reading or saving an image resource. This
    object wraps information to that request and acts as an interface
    for the plugins to several resources; it allows the user to read
    from filenames, files, http, zipfiles, raw bytes, etc., but offer
    a simple interface to the plugins via ``get_file()`` and
    ``get_local_filename()``.

    For each read/write operation a single Request instance is used and passed
    to the can_read/can_write method of a format, and subsequently to
    the Reader/Writer class. This allows rudimentary passing of
    information between different formats and between a format and
    associated reader/writer.

    parameters
    ----------
    uri : {str, bytes, file}
        The resource to load the image from.
    mode : str
        The first character is "r" or "w", indicating a read or write
        request. The second character is used to indicate the kind of data:
        "i" for an image, "I" for multiple images, "v" for a volume,
        "V" for multiple volumes, "?" for don't care.
    """

    def __init__(self, uri, mode, **kwargs):

        # General
        self._uri_type = None
        self._filename = None
        self._extension = None
        self._kwargs = kwargs
        self._result = None  # Some write actions may have a result

        # To handle the user-side
        self._filename_zip = None  # not None if a zipfile is used
        self._bytes = None  # Incoming bytes
        self._zipfile = None  # To store a zipfile instance (if used)

        # To handle the plugin side
        self._file = None  # To store the file instance
        self._file_is_local = False  # whether the data needs to be copied at end
        self._filename_local = None  # not None if using tempfile on this FS
        self._firstbytes = None  # For easy header parsing

        # To store formats that may be able to fulfil this request
        # self._potential_formats = []

        # Check mode
        self._mode = mode
        if not isinstance(mode, string_types):
            raise ValueError("Request requires mode must be a string")
        if not len(mode) == 2:
            raise ValueError("Request requires mode to have two chars")
        if mode[0] not in "rw":
            raise ValueError('Request requires mode[0] to be "r" or "w"')
        if mode[1] not in "iIvV?":
            raise ValueError('Request requires mode[1] to be in "iIvV?"')

        # Parse what was given
        self._parse_uri(uri)

        # Set extension
        if self._filename is not None:
            ext = self._filename
            if self._filename.startswith(("http://", "https://", "ftp://", "ftps://")):
                ext = ext.split("?")[0]
            self._extension = "." + ext.split(".")[-1].lower()

    def _parse_uri(self, uri):
        """ Try to figure our what we were given
        """
        py3k = sys.version_info[0] == 3
        is_read_request = self.mode[0] == "r"
        is_write_request = self.mode[0] == "w"

        if isinstance(uri, string_types):
            # Explicit
            if uri.startswith("imageio:"):
                if is_write_request:
                    raise RuntimeError("Cannot write to the standard images.")
                fn = uri.split(":", 1)[-1].lower()
                fn, _, zip_part = fn.partition(".zip/")
                if zip_part:
                    fn += ".zip"
                if fn not in EXAMPLE_IMAGES:
                    raise ValueError("Unknown standard image %r." % fn)
                self._uri_type = URI_FILENAME
                self._filename = get_remote_file("images/" + fn, auto=True)
                if zip_part:
                    self._filename += "/" + zip_part
            elif uri.startswith("http://") or uri.startswith("https://"):
                self._uri_type = URI_HTTP
                self._filename = uri
            elif uri.startswith("ftp://") or uri.startswith("ftps://"):
                self._uri_type = URI_FTP
                self._filename = uri
            elif uri.startswith("file://"):
                self._uri_type = URI_FILENAME
                self._filename = uri[7:]
            elif uri.startswith(SPECIAL_READ_URIS) and is_read_request:
                self._uri_type = URI_BYTES
                self._filename = uri
            elif uri.startswith(RETURN_BYTES) and is_write_request:
                self._uri_type = URI_BYTES
                self._filename = uri
            # Less explicit (particularly on py 2.x)
            elif py3k:
                self._uri_type = URI_FILENAME
                self._filename = uri
            else:  # pragma: no cover - our ref for coverage is py3k
                try:
                    isfile = os.path.isfile(uri)
                except Exception:
                    isfile = False  # If checking does not even work ...
                if isfile:
                    self._uri_type = URI_FILENAME
                    self._filename = uri
                elif len(uri) < 256:  # Can go wrong with veeery tiny images
                    self._uri_type = URI_FILENAME
                    self._filename = uri
                elif isinstance(uri, binary_type) and is_read_request:
                    self._uri_type = URI_BYTES
                    self._filename = "<bytes>"
                    self._bytes = uri
                else:
                    self._uri_type = URI_FILENAME
                    self._filename = uri
        elif isinstance(uri, memoryview) and is_read_request:
            self._uri_type = URI_BYTES
            self._filename = "<bytes>"
            self._bytes = uri.tobytes()
        elif isinstance(uri, binary_type) and is_read_request:
            self._uri_type = URI_BYTES
            self._filename = "<bytes>"
            self._bytes = uri
        elif Path is not None and isinstance(uri, Path):
            self._uri_type = URI_FILENAME
            self._filename = str(uri)
        # Files
        elif is_read_request:
            if hasattr(uri, "read") and hasattr(uri, "close"):
                self._uri_type = URI_FILE
                self._filename = "<file>"
                self._file = uri  # Data must be read from here
        elif is_write_request:
            if hasattr(uri, "write") and hasattr(uri, "close"):
                self._uri_type = URI_FILE
                self._filename = "<file>"
                self._file = uri  # Data must be written here

        # Expand user dir
        if self._uri_type == URI_FILENAME and self._filename.startswith("~"):
            self._filename = os.path.expanduser(self._filename)

        # Check if a zipfile
        if self._uri_type == URI_FILENAME:
            # Search for zip extension followed by a path separater
            for needle in [".zip/", ".zip\\"]:
                zip_i = self._filename.lower().find(needle)
                if zip_i > 0:
                    zip_i += 4
                    self._uri_type = URI_ZIPPED
                    self._filename_zip = (
                        self._filename[:zip_i],
                        self._filename[zip_i:].lstrip("/\\"),
                    )
                    break

        # Check if we could read it
        if self._uri_type is None:
            uri_r = repr(uri)
            if len(uri_r) > 60:
                uri_r = uri_r[:57] + "..."
            raise IOError("Cannot understand given URI: %s." % uri_r)

        # Check if this is supported
        noWriting = [URI_HTTP, URI_FTP]
        if is_write_request and self._uri_type in noWriting:
            raise IOError("imageio does not support writing to http/ftp.")

        # Deprecated way to load standard images, give a sensible error message
        if is_read_request and self._uri_type in [URI_FILENAME, URI_ZIPPED]:
            fn = self._filename
            if self._filename_zip:
                fn = self._filename_zip[0]
            if (not os.path.exists(fn)) and (fn in EXAMPLE_IMAGES):
                raise IOError(
                    "No such file: %r. This file looks like one of "
                    "the standard images, but from imageio 2.1, "
                    "standard images have to be specified using "
                    '"imageio:%s".' % (fn, fn)
                )

        # Make filename absolute
        if self._uri_type in [URI_FILENAME, URI_ZIPPED]:
            if self._filename_zip:
                self._filename_zip = (
                    os.path.abspath(self._filename_zip[0]),
                    self._filename_zip[1],
                )
            else:
                self._filename = os.path.abspath(self._filename)

        # Check whether file name is valid
        if self._uri_type in [URI_FILENAME, URI_ZIPPED]:
            fn = self._filename
            if self._filename_zip:
                fn = self._filename_zip[0]
            if is_read_request:
                # Reading: check that the file exists (but is allowed a dir)
                if not os.path.exists(fn):
                    raise FileNotFoundError("No such file: '%s'" % fn)
            else:
                # Writing: check that the directory to write to does exist
                dn = os.path.dirname(fn)
                if not os.path.exists(dn):
                    raise FileNotFoundError("The directory %r does not exist" % dn)

    @property
    def filename(self):
        """ The uri for which reading/saving was requested. This
        can be a filename, an http address, or other resource
        identifier. Do not rely on the filename to obtain the data,
        but use ``get_file()`` or ``get_local_filename()`` instead.
        """
        return self._filename

    @property
    def extension(self):
        """ The (lowercase) extension of the requested filename.
        Suffixes in url's are stripped. Can be None if the request is
        not based on a filename.
        """
        return self._extension

    @property
    def mode(self):
        """ The mode of the request. The first character is "r" or "w",
        indicating a read or write request. The second character is
        used to indicate the kind of data:
        "i" for an image, "I" for multiple images, "v" for a volume,
        "V" for multiple volumes, "?" for don't care.
        """
        return self._mode

    @property
    def kwargs(self):
        """ The dict of keyword arguments supplied by the user.
        """
        return self._kwargs

    ## For obtaining data

    def get_file(self):
        """ get_file()
        Get a file object for the resource associated with this request.
        If this is a reading request, the file is in read mode,
        otherwise in write mode. This method is not thread safe. Plugins
        should not close the file when done.

        This is the preferred way to read/write the data. But if a
        format cannot handle file-like objects, they should use
        ``get_local_filename()``.
        """
        want_to_write = self.mode[0] == "w"

        # Is there already a file?
        # Either _uri_type == URI_FILE, or we already opened the file,
        # e.g. by using firstbytes
        if self._file is not None:
            return self._file

        if self._uri_type == URI_BYTES:
            if want_to_write:
                # Create new file object, we catch the bytes in finish()
                self._file = BytesIO()
                self._file_is_local = True
            else:
                self._file = BytesIO(self._bytes)

        elif self._uri_type == URI_FILENAME:
            if want_to_write:
                self._file = open(self.filename, "wb")
            else:
                self._file = open(self.filename, "rb")

        elif self._uri_type == URI_ZIPPED:
            # Get the correct filename
            filename, name = self._filename_zip
            if want_to_write:
                # Create new file object, we catch the bytes in finish()
                self._file = BytesIO()
                self._file_is_local = True
            else:
                # Open zipfile and open new file object for specific file
                self._zipfile = zipfile.ZipFile(filename, "r")
                self._file = self._zipfile.open(name, "r")
                make_file_object_support_noop_seeks(self._file)

        elif self._uri_type in [URI_HTTP or URI_FTP]:
            assert not want_to_write  # This should have been tested in init
            self._file = urlopen(self.filename, timeout=5)
            make_file_object_support_noop_seeks(self._file)

        return self._file

    def get_local_filename(self):
        """ get_local_filename()
        If the filename is an existing file on this filesystem, return
        that. Otherwise a temporary file is created on the local file
        system which can be used by the format to read from or write to.
        """

        if self._uri_type == URI_FILENAME:
            return self._filename
        else:
            # Get filename
            if self._uri_type in (URI_HTTP, URI_FTP):
                ext = os.path.splitext(self._filename.split("?")[0])[1]
            else:
                ext = os.path.splitext(self._filename)[1]
            self._filename_local = tempfile.mktemp(ext, "imageio_")
            # Write stuff to it?
            if self.mode[0] == "r":
                with open(self._filename_local, "wb") as file:
                    shutil.copyfileobj(self.get_file(), file)
            return self._filename_local

    def finish(self):
        """ finish()
        For internal use (called when the context of the reader/writer
        exits). Finishes this request. Close open files and process
        results.
        """

        if self.mode[0] == "w":

            # See if we "own" the data and must put it somewhere
            bytes = None
            if self._filename_local:
                with open(self._filename_local, "rb") as file:
                    bytes = file.read()
            elif self._file_is_local:
                bytes = self._file.getvalue()

            # Put the data in the right place
            if bytes is not None:
                if self._uri_type == URI_BYTES:
                    self._result = bytes  # Picked up by imread function
                elif self._uri_type == URI_FILE:
                    self._file.write(bytes)
                elif self._uri_type == URI_ZIPPED:
                    zf = zipfile.ZipFile(self._filename_zip[0], "a")
                    zf.writestr(self._filename_zip[1], bytes)
                    zf.close()
                # elif self._uri_type == URI_FILENAME: -> is always direct
                # elif self._uri_type == URI_FTP/HTTP: -> write not supported

        # Close open files that we know of (and are responsible for)
        if self._file and self._uri_type != URI_FILE:
            self._file.close()
            self._file = None
        if self._zipfile:
            self._zipfile.close()
            self._zipfile = None

        # Remove temp file
        if self._filename_local:
            try:
                os.remove(self._filename_local)
            except Exception:  # pragma: no cover
                pass
            self._filename_local = None

        # Detach so gc can clean even if a reference of self lingers
        self._bytes = None

    def get_result(self):
        """ For internal use. In some situations a write action can have
        a result (bytes data). That is obtained with this function.
        """
        self._result, res = None, self._result
        return res

    @property
    def firstbytes(self):
        """ The first 256 bytes of the file. These can be used to
        parse the header to determine the file-format.
        """
        if self._firstbytes is None:
            self._read_first_bytes()
        return self._firstbytes

    def _read_first_bytes(self, N=256):
        if self._bytes is not None:
            self._firstbytes = self._bytes[:N]
        else:
            # Prepare
            try:
                f = self.get_file()
            except IOError:
                if os.path.isdir(self.filename):  # A directory, e.g. for DICOM
                    self._firstbytes = binary_type()
                    return
                raise
            try:
                i = f.tell()
            except Exception:
                i = None
            # Read
            self._firstbytes = read_n_bytes(f, N)
            # Set back
            try:
                if i is None:
                    raise Exception("cannot seek with None")
                f.seek(i)
            except Exception:
                # Prevent get_file() from reusing the file
                self._file = None
                # If the given URI was a file object, we have a problem,
                if self._uri_type == URI_FILE:
                    raise IOError("Cannot seek back after getting firstbytes!")


def read_n_bytes(f, N):
    """ read_n_bytes(file, n)

    Read n bytes from the given file, or less if the file has less
    bytes. Returns zero bytes if the file is closed.
    """
    bb = binary_type()
    while len(bb) < N:
        extra_bytes = f.read(N - len(bb))
        if not extra_bytes:
            break
        bb += extra_bytes
    return bb


def make_file_object_support_noop_seeks(f):
    """This fixes up an HTTPResponse object so that it can tell(), and also
    seek() will work if its effectively a no-op. This allows tools like Pillow
    to use the file object.
    """
    count = [0]

    def read(n=None):
        res = ori_read(n)
        count[0] += len(res)
        return res

    def tell():
        return count[0]

    def seek(i, mode=0):
        if not (mode == 0 and i == count[0]):
            ori_seek(i, mode)

    def fail_seek(i, mode=0):
        raise RuntimeError("No seeking allowed!")

    # Note, there is currently no protection from wrapping an object more than
    # once, it will (probably) work though, because closures.
    ori_read = f.read
    ori_seek = f.seek if hasattr(f, "seek") else fail_seek

    f.read = read
    f.tell = tell
    f.seek = seek
