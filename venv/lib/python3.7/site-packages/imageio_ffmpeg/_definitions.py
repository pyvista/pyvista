import sys
import struct

__version__ = "0.3.0"


def get_platform():
    bits = struct.calcsize("P") * 8
    if sys.platform.startswith("linux"):
        return "linux{}".format(bits)
    elif sys.platform.startswith("win"):
        return "win{}".format(bits)
    elif sys.platform.startswith("darwin"):
        return "osx{}".format(bits)
    else:  # pragma: no cover
        return None


# The Linux static builds (https://johnvansickle.com/ffmpeg/) are build
# for Linux kernels 2.6.32 and up (at the time of writing, ffmpeg v4.1).
# This corresponds to CentOS 6. This means we should use manylinux2010 and not
# manylinux1.
# manylinux1: https://www.python.org/dev/peps/pep-0513
# manylinux2010: https://www.python.org/dev/peps/pep-0571


# Platform string -> ffmpeg filename
FNAME_PER_PLATFORM = {
    "osx64": "ffmpeg-osx64-v4.1",  # 10.9+
    "win32": "ffmpeg-win32-v4.1.exe",  # Windows 7+
    "win64": "ffmpeg-win64-v4.1.exe",
    # "linux32": "ffmpeg-linux32-v4.1",
    "linux64": "ffmpeg-linux64-v4.1",  # Kernel 2.6.32+
}

osxplats = "macosx_10_9_intel.macosx_10_9_x86_64.macosx_10_10_intel.macosx_10_10_x86_64"

# Wheel tag -> platform string
WHEEL_BUILDS = {
    "py3-none-manylinux2010_x86_64": "linux64",
    "py3-none-" + osxplats: "osx64",
    "py3-none-win32": "win32",
    "py3-none-win_amd64": "win64",
}
