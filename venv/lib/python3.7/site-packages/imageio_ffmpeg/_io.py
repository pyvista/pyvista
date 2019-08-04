import sys
import time
import signal
import subprocess

from ._utils import get_ffmpeg_exe, logger
from ._parsing import LogCatcher, parse_ffmpeg_header, cvsecs


ISWIN = sys.platform.startswith("win")


exe = None


def _get_exe():
    global exe
    if exe is None:
        exe = get_ffmpeg_exe()
    return exe


def count_frames_and_secs(path):
    """
    Get the number of frames and number of seconds for the given video
    file. Note that this operation can be quite slow for large files.
    
    Disclaimer: I've seen this produce different results from actually reading
    the frames with older versions of ffmpeg (2.x). Therefore I cannot say
    with 100% certainty that the returned values are always exact.
    """
    # https://stackoverflow.com/questions/2017843/fetch-frame-count-with-ffmpeg

    assert isinstance(path, str), "Video path must be a string"

    cmd = [_get_exe(), "-i", path, "-map", "0:v:0", "-c", "copy", "-f", "null", "-"]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=ISWIN)
    except subprocess.CalledProcessError as err:
        out = err.output.decode(errors="ignore")
        raise RuntimeError("FFMEG call failed with {}:\n{}".format(err.returncode, out))

    # Note that other than with the subprocess calls below, ffmpeg wont hang here.
    # Worst case Python will stop/crash and ffmpeg will continue running until done.

    nframes = nsecs = None
    for line in reversed(out.splitlines()):
        if line.startswith(b"frame="):
            line = line.decode(errors="ignore")
            i = line.find("frame=")
            if i >= 0:
                s = line[i:].split("=", 1)[-1].lstrip().split(" ", 1)[0].strip()
                nframes = int(s)
            i = line.find("time=")
            if i >= 0:
                s = line[i:].split("=", 1)[-1].lstrip().split(" ", 1)[0].strip()
                nsecs = cvsecs(*s.split(":"))
            return nframes, nsecs

    raise RuntimeError("Could not get number of frames")  # pragma: no cover


def read_frames(path, pix_fmt="rgb24", bpp=3, input_params=None, output_params=None):
    """
    Create a generator to iterate over the frames in a video file.
    
    It first yields a small metadata dictionary that contains:
    
    * ffmpeg_version: the ffmpeg version is use (as a string).
    * codec: a hint about the codec used to encode the video, e.g. "h264"
    * source_size: the width and height of the encoded video frames
    * size: the width and height of the frames that will be produced
    * fps: the frames per second. Can be zero if it could not be detected.
    * duration: duration in seconds. Can be zero if it could not be detected.
    
    After that, it yields frames until the end of the video is reached. Each
    frame is a bytes object.
    
    This function makes no assumptions about the number of frames in
    the data. For one because this is hard to predict exactly, but also
    because it may depend on the provided output_params. If you want
    to know the number of frames in a video file, use count_frames_and_secs().
    It is also possible to estimate the number of frames from the fps and
    duration, but note that even if both numbers are present, the resulting
    value is not always correct.
    
    Example:
        
        gen = read_frames(path)
        meta = gen.__next__()
        for frame in gen:
            print(len(frame))
    
    Parameters:
        path (str): the file to write to.
        pix_fmt (str): the pixel format of the frames to be read.
            The default is "rgb24" (frames are uint8 RGB images).
        bpp (int): The number of bytes per pixel in the output frames.
            This depends on the given pix_fmt. Default is 3 (RGB).
        input_params (list): Additional ffmpeg input command line parameters.
        output_params (list): Additional ffmpeg output command line parameters.
    """

    # ----- Input args

    assert isinstance(path, str), "Video path must be a string"
    # Note: Dont check whether it exists. The source could be e.g. a camera.

    pix_fmt = pix_fmt or "rgb24"
    bpp = bpp or 3
    input_params = input_params or []
    output_params = output_params or []

    assert isinstance(pix_fmt, str), "pix_fmt must be a string"
    assert isinstance(bpp, int), "bpp must be an int"
    assert isinstance(input_params, list), "input_params must be a list"
    assert isinstance(output_params, list), "output_params must be a list"

    # ----- Prepare

    pre_output_params = ["-pix_fmt", pix_fmt, "-vcodec", "rawvideo", "-f", "image2pipe"]

    cmd = [_get_exe()]
    cmd += input_params + ["-i", path]
    cmd += pre_output_params + output_params + ["-"]

    p = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=ISWIN,
    )

    log_catcher = LogCatcher(p.stderr)

    try:

        # ----- Load meta data

        # Wait for the log catcher to get the meta information
        etime = time.time() + 10.0
        while (not log_catcher.header) and time.time() < etime:
            time.sleep(0.01)

        # Check whether we have the information
        if not log_catcher.header:
            err2 = log_catcher.get_text(0.2)
            fmt = "Could not load meta information\n=== stderr ===\n{}"
            raise IOError(fmt.format(err2))
        elif "No such file or directory" in log_catcher.header:
            raise IOError("{} not found! Wrong path?".format(path))

        meta = parse_ffmpeg_header(log_catcher.header)
        yield meta

        # ----- Read frames

        w, h = meta["size"]
        framesize = w * h * bpp
        framenr = 0

        while True:
            framenr += 1
            try:
                bb = bytes()
                while len(bb) < framesize:
                    extra_bytes = p.stdout.read(framesize - len(bb))
                    if not extra_bytes:
                        if len(bb) == 0:
                            return
                        else:
                            raise RuntimeError(
                                "End of file reached before full frame could be read."
                            )
                    bb += extra_bytes
                yield bb
            except Exception as err:
                err1 = str(err)
                err2 = log_catcher.get_text(0.4)
                fmt = "Could not read frame {}:\n{}\n=== stderr ===\n{}"
                raise RuntimeError(fmt.format(framenr, err1, err2))

    finally:
        # Generators are automatically closed when they get deleted,
        # so this code is almost guaranteed to run.

        if p.poll() is None:

            # Ask ffmpeg to quit
            try:
                if True:
                    p.communicate(b"q")
                else:  # pragma: no cover
                    # I read somewhere that modern ffmpeg on Linux prefers a
                    # "ctrl-c", but tests so far suggests sending q is better.
                    p.send_signal(signal.SIGINT)
            except Exception as err:  # pragma: no cover
                logger.warning("Error while attempting stop ffmpeg: " + str(err))

            # Wait for it to stop
            etime = time.time() + 1.5
            while time.time() < etime and p.poll() is None:
                time.sleep(0.01)

            # Grr, we have to kill it
            if p.poll() is None:  # pragma: no cover
                logger.warning("We had to kill ffmpeg to stop it.")
                p.kill()


def write_frames(
    path,
    size,
    pix_fmt_in="rgb24",
    pix_fmt_out="yuv420p",
    fps=16,
    quality=5,
    bitrate=None,
    codec=None,
    macro_block_size=16,
    ffmpeg_log_level="warning",
    ffmpeg_timeout=20.0,
    input_params=None,
    output_params=None,
):
    """
    Create a generator to write frames (bytes objects) into a video file.
    
    The frames are written by using the generator's `send()` method. Frames
    can be anything that can be written to a file. Typically these are
    bytes objects, but c-contiguous Numpy arrays also work.
    
    Example:
    
        gen = write_frames(path, size)
        gen.send(None)  # seed the generator
        for frame in frames:
            gen.send(frame)
        gen.close()  # don't forget this
    
    Parameters:
        path (str): the file to write to.
        size (tuple): the width and height of the frames.
        pix_fmt_in (str): the pixel format of incoming frames.
            E.g. "gray", "gray8a", "rgb24", or "rgba". Default "rgb24".
        pix_fmt_out (str): the pixel format to store frames. Default yuv420p".
        fps (float): The frames per second. Default 16.
        quality (float): A measure for quality between 0 and 10. Default 5.
            Ignored if bitrate is given.
        bitrate (str): The bitrate, e.g. "192k". The defaults are pretty good.
        codec (str): The codec. Default "libx264" (or "msmpeg4" for .wmv).
        macro_block_size (int): You probably want to align the size of frames
            to this value to avoid image resizing. Default 16. Can be set
            to 1 to avoid block alignment, though this is not recommended.
        ffmpeg_log_level (str): The ffmpeg logging level. Default "warning".
        ffmpeg_timeout (float): Timeout in seconds to wait for ffmpeg process
            to finish. Value of 0 will wait forever. The time that ffmpeg needs
            depends on CPU speed, compression, and frame size. Default 20.0.
        input_params (list): Additional ffmpeg input command line parameters.
        output_params (list): Additional ffmpeg output command line parameters.
    """

    # ----- Input args

    assert isinstance(path, str), "Video path must be a string"

    # The pix_fmt_out yuv420p is the best for the outpur to work in
    # QuickTime and most other players. These players only support
    # the YUV planar color space with 4:2:0 chroma subsampling for
    # H.264 video. Otherwise, depending on the source, ffmpeg may
    # output to a pixel format that may be incompatible with these
    # players. See https://trac.ffmpeg.org/wiki/Encode/H.264#Encodingfordumbplayers

    pix_fmt_in = pix_fmt_in or "rgb24"
    pix_fmt_out = pix_fmt_out or "yuv420p"
    fps = fps or 16
    quality = quality or 5
    # bitrate, codec, macro_block_size can all be None or ...
    macro_block_size = macro_block_size or 16
    ffmpeg_log_level = ffmpeg_log_level or "warning"
    input_params = input_params or []
    output_params = output_params or []

    floatish = float, int
    if isinstance(size, (tuple, list)):
        assert len(size) == 2, "size must be a 2-tuple"
        assert isinstance(size[0], int) and isinstance(
            size[1], int
        ), "size must be ints"
        sizestr = "{:d}x{:d}".format(*size)
    # elif isinstance(size, str):
    #     assert "x" in size, "size as string must have format NxM"
    #     sizestr = size
    else:
        assert False, "size must be str or tuple"
    assert isinstance(pix_fmt_in, str), "pix_fmt_in must be str"
    assert isinstance(pix_fmt_out, str), "pix_fmt_out must be str"
    assert isinstance(fps, floatish), "fps must be float"
    assert isinstance(quality, floatish), "quality must be float"
    assert 1 <= quality <= 10, "quality must be between 1 and 10 inclusive"
    assert isinstance(macro_block_size, int), "macro_block_size must be int"
    assert isinstance(ffmpeg_log_level, str), "ffmpeg_log_level must be str"
    assert isinstance(ffmpeg_timeout, floatish), "ffmpeg_timeout must be float"
    assert isinstance(input_params, list), "input_params must be a list"
    assert isinstance(output_params, list), "output_params must be a list"

    # ----- Prepare

    # Get parameters
    default_codec = "libx264"
    if path.lower().endswith(".wmv"):
        # This is a safer default codec on windows to get videos that
        # will play in powerpoint and other apps. H264 is not always
        # available on windows.
        default_codec = "msmpeg4"
    codec = codec or default_codec

    # Get command
    cmd = [_get_exe(), "-y", "-f", "rawvideo", "-vcodec", "rawvideo", "-s", sizestr]
    cmd += ["-pix_fmt", pix_fmt_in, "-r", "{:.02f}".format(fps)] + input_params
    cmd += ["-i", "-"]
    cmd += ["-an", "-vcodec", codec, "-pix_fmt", pix_fmt_out]

    # Add fixed bitrate or variable bitrate compression flags
    if bitrate is not None:
        cmd += ["-b:v", str(bitrate)]
    elif quality is not None:  # If None, then we don't add anything
        quality = 1 - quality / 10.0
        if codec == "libx264":
            # crf ranges 0 to 51, 51 being worst.
            quality = int(quality * 51)
            cmd += ["-crf", str(quality)]  # for h264
        else:  # Many codecs accept q:v
            # q:v range can vary, 1-31, 31 being worst
            # But q:v does not always have the same range.
            # May need a way to find range for any codec.
            quality = int(quality * 30) + 1
            cmd += ["-qscale:v", str(quality)]  # for others

    # Note, for most codecs, the image dimensions must be divisible by
    # 16 the default for the macro_block_size is 16. Check if image is
    # divisible, if not have ffmpeg upsize to nearest size and warn
    # user they should correct input image if this is not desired.
    if macro_block_size > 1:
        if size[0] % macro_block_size > 0 or size[1] % macro_block_size > 0:
            out_w = size[0]
            out_h = size[1]
            if size[0] % macro_block_size > 0:
                out_w += macro_block_size - (size[0] % macro_block_size)
            if size[1] % macro_block_size > 0:
                out_h += macro_block_size - (size[1] % macro_block_size)
            cmd += ["-vf", "scale={}:{}".format(out_w, out_h)]
            logger.warning(
                "IMAGEIO FFMPEG_WRITER WARNING: input image is not"
                " divisible by macro_block_size={}, resizing from {} "
                "to {} to ensure video compatibility with most codecs "
                "and players. To prevent resizing, make your input "
                "image divisible by the macro_block_size or set the "
                "macro_block_size to 1 (risking incompatibility).".format(
                    macro_block_size, size[:2], (out_w, out_h)
                )
            )

    # Rather than redirect stderr to a pipe, just set minimal
    # output from ffmpeg by default. That way if there are warnings
    # the user will see them.
    cmd += ["-v", ffmpeg_log_level]
    cmd += output_params
    cmd.append(path)
    cmd_str = " ".join(cmd)
    if any(
        [level in ffmpeg_log_level for level in ("info", "verbose", "debug", "trace")]
    ):
        logger.info("RUNNING FFMPEG COMMAND: " + cmd_str)

    # Launch process
    p = subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None, shell=ISWIN
    )

    # For Windows, set `shell=True` in sp.Popen to prevent popup
    # of a command line window in frozen applications.
    # Note that directing stderr to a pipe on windows will cause ffmpeg
    # to hang if the buffer is not periodically cleared using
    # StreamCatcher or other means.
    # Setting bufsize to 0 or a small value does not seem to have much effect
    # (at least on Windows). I suspect that ffmpeg buffers # multiple frames
    # (before encoding in a batch).

    # ----- Write frames

    try:

        # Just keep going until the generator.close() is called (raises GeneratorExit).
        # This could also happen when the generator is deleted somehow.
        nframes = 0
        while True:

            # Get frame
            bb = (yield)

            # framesize = size[0] * size[1] * depth * bpp
            # assert isinstance(bb, bytes), "Frame must be send as bytes"
            # assert len(bb) == framesize, "Frame must have width*height*depth*bpp bytes"
            # Actually, we accept anything that can be written to file.
            # This e.g. allows writing numpy arrays without having to make a copy ...

            # Write
            try:
                p.stdin.write(bb)
            except Exception as err:
                # Show the command and stderr from pipe
                msg = (
                    "{0:}\n\nFFMPEG COMMAND:\n{1:}\n\nFFMPEG STDERR "
                    "OUTPUT:\n".format(err, cmd_str)
                )
                raise IOError(msg)

            nframes += 1

    except GeneratorExit:
        if nframes == 0:
            logger.warning("No frames have been written; the written video is invalid.")
    finally:

        if p.poll() is None:

            # Ask ffmpeg to quit - and wait for it to finish writing the file.
            # Depending on the frame size and encoding this can take a few
            # seconds (sometimes 10-20). Since a user may get bored and hit
            # Ctrl-C, we wrap this in a try-except.
            waited = False
            try:
                try:
                    p.stdin.close()
                except Exception:  # pragma: no cover
                    pass
                etime = time.time() + ffmpeg_timeout
                while (not ffmpeg_timeout or time.time() < etime) and p.poll() is None:
                    time.sleep(0.01)
                waited = True
            finally:
                # Grr, we have to kill it
                if p.poll() is None:  # pragma: no cover
                    more = " Consider increasing ffmpeg_timeout." if waited else ""
                    logger.warning("We had to kill ffmpeg to stop it." + more)
                    p.kill()
