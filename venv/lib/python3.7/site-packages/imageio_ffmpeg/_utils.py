import os
from pkg_resources import resource_filename
import sys
import subprocess
import logging

from ._definitions import get_platform, FNAME_PER_PLATFORM

logger = logging.getLogger("imageio_ffmpeg")


def get_ffmpeg_exe():
    """
    Get the ffmpeg executable file. This can be the binary defined by 
    the IMAGEIO_FFMPEG_EXE environment variable, the binary distributed
    with imageio-ffmpeg, an ffmpeg binary installed with conda, or the
    system ffmpeg (in that order). A RuntimeError is raised if no valid
    ffmpeg could be found.
    """

    # 1. Try environment variable. - Dont test it: the user is explicit here!
    exe = os.getenv("IMAGEIO_FFMPEG_EXE", None)
    if exe:
        return exe

    plat = get_platform()

    # 2. Try from here
    bin_dir = resource_filename("imageio_ffmpeg", "binaries")
    exe = os.path.join(bin_dir, FNAME_PER_PLATFORM.get(plat, ""))
    if exe and os.path.isfile(exe) and _is_valid_exe(exe):
        return exe

    # 3. Try binary from conda package
    # (installed e.g. via `conda install ffmpeg -c conda-forge`)
    if plat.startswith("win"):
        exe = os.path.join(sys.prefix, "Library", "bin", "ffmpeg.exe")
    else:
        exe = os.path.join(sys.prefix, "bin", "ffmpeg")
    if exe and os.path.isfile(exe) and _is_valid_exe(exe):
        return exe

    # 4. Try system ffmpeg command
    exe = "ffmpeg"
    if _is_valid_exe(exe):
        return exe

    # Nothing was found
    raise RuntimeError(
        "No ffmpeg exe could be found. Install ffmpeg on your system, "
        "or set the IMAGEIO_FFMPEG_EXE environment variable."
    )


def _is_valid_exe(exe):
    cmd = [exe, "-version"]
    try:
        with open(os.devnull, "w") as null:
            subprocess.check_call(cmd, stdout=null, stderr=subprocess.STDOUT)
        return True
    except (OSError, ValueError, subprocess.CalledProcessError):
        return False


def get_ffmpeg_version():
    """
    Get the version of the used ffmpeg executable (as a string).
    """
    exe = get_ffmpeg_exe()
    line = subprocess.check_output([exe, "-version"]).split(b"\n", 1)[0]
    line = line.decode(errors="ignore").strip()
    version = line.split("version", 1)[-1].lstrip().split(" ", 1)[0].strip()
    return version
