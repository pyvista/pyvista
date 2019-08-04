""" This module attempts to make it easy to create VTK-Python
unittests.  The module uses unittest for the test interface.  For more
documentation on what unittests are and how to use them, please read
these:

   http://www.python.org/doc/current/lib/module-unittest.html

   http://www.diveintopython.org/roman_divein.html


This VTK-Python test module supports image based tests with multiple
images per test suite and multiple images per individual test as well.
It also prints information appropriate for CDash
(http://open.kitware.com/).

This module defines several useful classes and functions to make
writing tests easy.  The most important of these are:

class vtkTest:
   Subclass this for your tests.  It also has a few useful internal
   functions that can be used to do some simple blackbox testing.

compareImage(renwin, img_fname, threshold=10):
   Compares renwin with image and generates image if it does not
   exist.  The threshold determines how closely the images must match.
   The function also handles multiple images and finds the best
   matching image.

compareImageWithSavedImage(src_img, img_fname, threshold=10):
   Compares given source image (in the form of a vtkImageData) with
   saved image and generates the image if it does not exist.  The
   threshold determines how closely the images must match.  The
   function also handles multiple images and finds the best matching
   image.

getAbsImagePath(img_basename):
   Returns the full path to the image given the basic image name.

main(cases):
   Does the testing given a list of tuples containing test classes and
   the starting string of the functions used for testing.

interact():
    Interacts with the user if necessary.  The behavior of this is
    rather trivial and works best when using Tkinter.  It does not do
    anything by default and stops to interact with the user when given
    the appropriate command line arguments.

isInteractive():
    If interact() is not good enough, use this to find if the mode is
    interactive or not and do whatever is necessary to generate an
    interactive view.

Examples:

  The best way to learn on how to use this module is to look at a few
  examples.  The end of this file contains a trivial example.  Please
  also look at the following examples:

    Rendering/Testing/Python/TestTkRenderWidget.py,
    Rendering/Testing/Python/TestTkRenderWindowInteractor.py

Created: September, 2002

Prabhu Ramachandran <prabhu@aero.iitb.ac.in>
"""
from __future__ import absolute_import
import sys, os, time
import os.path
import unittest, getopt
import vtk
from . import BlackBox

# location of the VTK data files.  Set via command line args or
# environment variable.
VTK_DATA_ROOT = ""

# a list of paths to specific input data files
VTK_DATA_PATHS = []

# location of the VTK baseline images.  Set via command line args or
# environment variable.
VTK_BASELINE_ROOT = ""

# location of the VTK difference images for failed tests.  Set via
# command line args or environment variable.
VTK_TEMP_DIR = ""

# a list of paths to validated output files
VTK_BASELINE_PATHS = []

# Verbosity of the test messages (used by unittest)
_VERBOSE = 0

# Determines if it is necessary to interact with the user.  If zero
# don't interact if 1 interact.  Set via command line args
_INTERACT = 0

# This will be set to 1 when the image test will not be performed.
# This option is used internally by the script and set via command
# line arguments.
_NO_IMAGE = 0

def skip():
    '''Cause the test to be skipped due to insufficient requirements.'''
    sys.exit(0)
    # CMake 3.9 is required to make this not appear on CDash as a failure. See
    # issue #17031.
    #sys.exit(125)


class vtkTest(unittest.TestCase):
    """A simple default VTK test class that defines a few useful
    blackbox tests that can be readily used.  Derive your test cases
    from this class and use the following if you'd like to.

    Note: Unittest instantiates this class (or your subclass) each
    time it tests a method.  So if you do not want that to happen when
    generating VTK pipelines you should create the pipeline in the
    class definition as done below for _blackbox.
    """

    _blackbox = BlackBox.Tester(debug=0)

    # Due to what seems to be a bug in python some objects leak.
    # Avoid the exit-with-error in vtkDebugLeaks.
    dl = vtk.vtkDebugLeaks()
    dl.SetExitError(0)
    dl = None

    def _testParse(self, obj):
        """Does a blackbox test by attempting to parse the class for
        its various methods using vtkMethodParser.  This is a useful
        test because it gets all the methods of the vtkObject, parses
        them and sorts them into different classes of objects."""
        self._blackbox.testParse(obj)

    def _testGetSet(self, obj, excluded_methods=[]):
        """Checks the Get/Set method pairs by setting the value using
        the current state and making sure that it equals the value it
        was originally.  This effectively calls _testParse
        internally. """
        self._blackbox.testGetSet(obj, excluded_methods)

    def _testBoolean(self, obj, excluded_methods=[]):
        """Checks the Boolean methods by setting the value on and off
        and making sure that the GetMethod returns the the set value.
        This effectively calls _testParse internally. """
        self._blackbox.testBoolean(obj, excluded_methods)

    def pathToData(self, filename):
        """Given a filename with no path (i.e., no leading directories
        prepended), return the full path to a file as specified on the
        command line with a '-D' option.

        As an example, if a test is run with "-D /path/to/grid.vtu"
        then calling

            self.pathToData('grid.vtu')

        in your test will return "/path/to/grid.vtu". This is
        useful in combination with ExternalData, where data may be
        staged by CTest to a user-configured directory at build time.

        In order for this method to work, you must specify
        the JUST_VALID option for your test in CMake.
        """
        global VTK_DATA_PATHS
        if not filename:
            return VTK_DATA_PATHS
        for path in VTK_DATA_PATHS:
            if filename == os.path.split(path)[-1]:
                return path
        return filename

    def pathToValidatedOutput(self, filename):
        """Given a filename with no path (i.e., no leading directories
        prepended), return the full path to a file as specified on the
        command line with a '-V' option.

        As an example, if a test is run with
        "-V /path/to/validImage.png" then calling

            self.pathToData('validImage.png')

        in your test will return "/path/to/validImage.png". This is
        useful in combination with ExternalData, where data may be
        staged by CTest to a user-configured directory at build time.

        In order for this method to work, you must specify
        the JUST_VALID option for your test in CMake.
        """
        global VTK_BASELINE_PATHS
        if not filename:
            return VTK_BASELINE_PATHS
        for path in VTK_BASELINE_PATHS:
            if filename == os.path.split(path)[-1]:
                return path
        return filename

    def prepareTestImage(self, interactor, **kwargs):
        import time
        startTime = time.time()
        events = []

        def onKeyPress(caller, eventId):
            print('key is "' + caller.GetKeySym() + '"')
            events.append((time.time() - startTime, eventId, caller.GetKeySym()))

        def onButton(caller, eventId):
            events.append((time.time() - startTime, eventId))

        def onMovement(caller, eventId):
            events.append((time.time() - startTime, eventId, caller.GetEventPosition()))

        interactor.AddObserver(vtk.vtkCommand.KeyPressEvent, onKeyPress)
        interactor.AddObserver(vtk.vtkCommand.LeftButtonPressEvent, onButton)
        interactor.AddObserver(vtk.vtkCommand.LeftButtonReleaseEvent, onButton)
        interactor.AddObserver(vtk.vtkCommand.MouseMoveEvent, onMovement)
        interactor.Start()
        rw = interactor.GetRenderWindow()
        baseline = 'baselineFilename'
        if 'filename' in kwargs:
            # Render an image and save it to the given filename
            w2if = vtk.vtkWindowToImageFilter()
            w2if.ReadFrontBufferOff()
            w2if.SetInput(rw)
            w2if.Update()
            baselineWithPath = kwargs['filename']
            baseline = os.path.split(baselineWithPath)[-1]
            pngw = vtk.vtkPNGWriter()
            pngw.SetFileName(baselineWithPath)
            pngw.SetInputConnection(w2if.GetOutputPort())
            try:
                pngw.Write()
            except RuntimeError:
                w2if.ReadFrontBufferOn()
                pngw.Write()
        rsz = rw.GetSize()
        rrc = rw.GetRenderers()
        rrs = [rrc.GetItemAsObject(i) for i in range(rrc.GetNumberOfItems())]
        eye = [0,0,1]
        aim = [0,0,0]
        up  = [0,1,0]
        if len(rrs) > 0:
            cam = rrs[0].GetActiveCamera()
            eye = cam.GetPosition()
            aim = cam.GetFocalPoint()
            up  = cam.GetViewUp()
        print("""
        Replace prepareTestImage() in your script with the following to make a test:

            camera.SetPosition({eye[0]}, {eye[1]}, {eye[2]})
            camera.SetFocalPoint({aim[0]}, {aim[1]}, {aim[2]})
            camera.SetViewUp({up[0]}, {up[1]}, {up[2]})
            renwin.SetSize({rsz[0]}, {rsz[1]})
            self.assertImageMatch(renwin, '{baseline}')

        Be sure that "renwin" and "camera" are valid variables (or rename them in the
        snippet above) referencing the vtkRenderWindow and vtkCamera, respectively.
        """.format(eye=eye, aim=aim, up=up, rsz=rsz, baseline=baseline))
        return events

    def assertImageMatch(self, renwin, baseline, **kwargs):
        """Throw an error if a rendering in the render window does not match the baseline image.

        This method accepts a threshold keyword argument (with a default of 10)
        that specifies how different a baseline may be before causing a failure.
        """
        absoluteBaseline = baseline
        try:
            open(absoluteBaseline, 'r')
        except:
            absoluteBaseline = getAbsImagePath(baseline)
        compareImage(renwin, absoluteBaseline, **kwargs)

def interact():
    """Interacts with the user if necessary. """
    global _INTERACT
    if _INTERACT:
        raw_input("\nPress Enter/Return to continue with the testing. --> ")

def isInteractive():
    """Returns if the currently chosen mode is interactive or not
    based on command line options."""
    return _INTERACT

def getAbsImagePath(img_basename):
    """Returns the full path to the image given the basic image
    name."""
    global VTK_BASELINE_ROOT
    return os.path.join(VTK_BASELINE_ROOT, img_basename)

def _getTempImagePath(img_fname):
    x = os.path.join(VTK_TEMP_DIR, os.path.split(img_fname)[1])
    return os.path.abspath(x)

def compareImageWithSavedImage(src_img, img_fname, threshold=10):
    """Compares a source image (src_img, which is a vtkImageData) with
    the saved image file whose name is given in the second argument.
    If the image file does not exist the image is generated and
    stored.  If not the source image is compared to that of the
    figure.  This function also handles multiple images and finds the
    best matching image.
    """
    global _NO_IMAGE
    if _NO_IMAGE:
        return

    f_base, f_ext = os.path.splitext(img_fname)

    if not os.path.isfile(img_fname):
        # generate the image
        pngw = vtk.vtkPNGWriter()
        pngw.SetFileName(_getTempImagePath(img_fname))
        pngw.SetInputConnection(src_img.GetOutputPort())
        pngw.Write()
        _printCDashImageNotFoundError(img_fname)
        msg = "Missing baseline image: " + img_fname + "\nTest image created: " + _getTempImagePath(img_fname)
        sys.tracebacklimit = 0
        raise RuntimeError(msg)

    pngr = vtk.vtkPNGReader()
    pngr.SetFileName(img_fname)
    pngr.Update()

    idiff = vtk.vtkImageDifference()
    idiff.SetInputConnection(src_img.GetOutputPort())
    idiff.SetImageConnection(pngr.GetOutputPort())
    idiff.Update()

    min_err = idiff.GetThresholdedError()
    img_err = min_err

    err_index = 0
    count = 0
    if min_err > threshold:
        count = 1
        test_failed = 1
        err_index = -1
        while 1: # keep trying images till we get the best match.
            new_fname = f_base + "_%d.png"%count
            if not os.path.exists(new_fname):
                # no other image exists.
                break
            # since file exists check if it matches.
            pngr.SetFileName(new_fname)
            pngr.Update()
            idiff.Update()
            alt_err = idiff.GetThresholdedError()
            if alt_err < threshold:
                # matched,
                err_index = count
                test_failed = 0
                min_err = alt_err
                img_err = alt_err
                break
            else:
                if alt_err < min_err:
                    # image is a better match.
                    err_index = count
                    min_err = alt_err
                    img_err = alt_err

            count = count + 1
        # closes while loop.

        if test_failed:
            _handleFailedImage(idiff, pngr, img_fname)
            # Print for CDash.
            _printCDashImageError(img_err, err_index, f_base)
            msg = "Failed image test: %f\n"%idiff.GetThresholdedError()
            sys.tracebacklimit = 0
            raise RuntimeError(msg)
    # output the image error even if a test passed
    _printCDashImageSuccess(img_err, err_index)


def compareImage(renwin, img_fname, threshold=10):
    """Compares renwin's (a vtkRenderWindow) contents with the image
    file whose name is given in the second argument.  If the image
    file does not exist the image is generated and stored.  If not the
    image in the render window is compared to that of the figure.
    This function also handles multiple images and finds the best
    matching image.  """

    global _NO_IMAGE
    if _NO_IMAGE:
        return

    w2if = vtk.vtkWindowToImageFilter()
    w2if.ReadFrontBufferOff()
    w2if.SetInput(renwin)
    w2if.Update()
    try:
        compareImageWithSavedImage(w2if, img_fname, threshold)
    except RuntimeError:
        w2if.ReadFrontBufferOn()
        compareImageWithSavedImage(w2if, img_fname, threshold)
    return


def _printCDashImageError(img_err, err_index, img_base):
    """Prints the XML data necessary for CDash."""
    img_base = _getTempImagePath(img_base)
    print("Failed image test with error: %f"%img_err)
    print("<DartMeasurement name=\"ImageError\" type=\"numeric/double\"> "
          "%f </DartMeasurement>"%img_err)
    if err_index <= 0:
        print("<DartMeasurement name=\"BaselineImage\" type=\"text/string\">Standard</DartMeasurement>")
    else:
        print("<DartMeasurement name=\"BaselineImage\" type=\"numeric/integer\"> "
              "%d </DartMeasurement>"%err_index)

    print("<DartMeasurementFile name=\"TestImage\" type=\"image/png\"> "
          "%s </DartMeasurementFile>"%(img_base + '.png'))

    print("<DartMeasurementFile name=\"DifferenceImage\" type=\"image/png\"> "
          "%s </DartMeasurementFile>"%(img_base + '.diff.png'))
    print("<DartMeasurementFile name=\"ValidImage\" type=\"image/png\"> "
          "%s </DartMeasurementFile>"%(img_base + '.valid.png'))

def _printCDashImageNotFoundError(img_fname):
    """Prints the XML data necessary for Dart when the baseline image is not found."""
    print("<DartMeasurement name=\"ImageNotFound\" type=\"text/string\">" + img_fname + "</DartMeasurement>")

def _printCDashImageSuccess(img_err, err_index):
    "Prints XML data for Dart when image test succeeded."
    print("<DartMeasurement name=\"ImageError\" type=\"numeric/double\"> "
          "%f </DartMeasurement>"%img_err)
    if err_index <= 0:
        print("<DartMeasurement name=\"BaselineImage\" type=\"text/string\">Standard</DartMeasurement>")
    else:
       print("<DartMeasurement name=\"BaselineImage\" type=\"numeric/integer\"> "
             "%d </DartMeasurement>"%err_index)


def _handleFailedImage(idiff, pngr, img_fname):
    """Writes all the necessary images when an image comparison
    failed."""
    f_base, f_ext = os.path.splitext(img_fname)

    # write the difference image gamma adjusted for the dashboard.
    gamma = vtk.vtkImageShiftScale()
    gamma.SetInputConnection(idiff.GetOutputPort())
    gamma.SetShift(0)
    gamma.SetScale(10)

    pngw = vtk.vtkPNGWriter()
    pngw.SetFileName(_getTempImagePath(f_base + ".diff.png"))
    pngw.SetInputConnection(gamma.GetOutputPort())
    pngw.Write()

    # Write out the image that was generated.  Write it out as full so that
    # it may be used as a baseline image if the tester deems it valid.
    pngw.SetInputConnection(idiff.GetInputConnection(0,0))
    pngw.SetFileName(_getTempImagePath(f_base + ".png"))
    pngw.Write()

    # write out the valid image that matched.
    pngw.SetInputConnection(idiff.GetInputConnection(1,0))
    pngw.SetFileName(_getTempImagePath(f_base + ".valid.png"))
    pngw.Write()


def main(cases):
    """ Pass a list of tuples containing test classes and the starting
    string of the functions used for testing.

    Example:

    main ([(vtkTestClass, 'test'), (vtkTestClass1, 'test')])
    """

    processCmdLine()

    timer = vtk.vtkTimerLog()
    s_time = timer.GetCPUTime()
    s_wall_time = time.time()

    # run the tests
    result = test(cases)

    tot_time = timer.GetCPUTime() - s_time
    tot_wall_time = float(time.time() - s_wall_time)

    # output measurements for CDash
    print("<DartMeasurement name=\"WallTime\" type=\"numeric/double\"> "
          " %f </DartMeasurement>"%tot_wall_time)
    print("<DartMeasurement name=\"CPUTime\" type=\"numeric/double\"> "
          " %f </DartMeasurement>"%tot_time)

    # Delete these to eliminate debug leaks warnings.
    del cases, timer

    if result.wasSuccessful():
        sys.exit(0)
    else:
        sys.exit(1)


def test(cases):
    """ Pass a list of tuples containing test classes and the
    functions used for testing.

    It returns a unittest._TextTestResult object.

    Example:

      test = test_suite([(vtkTestClass, 'test'),
                        (vtkTestClass1, 'test')])
    """
    # Make the test suites from the arguments.
    suites = []
    for case in cases:
        suites.append(unittest.makeSuite(case[0], case[1]))
    test_suite = unittest.TestSuite(suites)

    # Now run the tests.
    runner = unittest.TextTestRunner(verbosity=_VERBOSE)
    result = runner.run(test_suite)

    return result


def usage():
    msg="""Usage:\nTestScript.py [options]\nWhere options are:\n

    -D /path/to/VTKData
    --data-dir /path/to/VTKData

          Directory containing VTK Data use for tests.  If this option
          is not set via the command line the environment variable
          VTK_DATA_ROOT is used.  If the environment variable is not
          set the value defaults to '../../../../../VTKData'.

    -B /path/to/valid/image_dir/
    --baseline-root /path/to/valid/image_dir/

          This is a path to the directory containing the valid images
          for comparison.  If this option is not set via the command
          line the environment variable VTK_BASELINE_ROOT is used.  If
          the environment variable is not set the value defaults to
          the same value set for -D (--data-dir).

    -T /path/to/valid/temporary_dir/
    --temp-dir /path/to/valid/temporary_dir/

          This is a path to the directory where the image differences
          are written.  If this option is not set via the command line
          the environment variable VTK_TEMP_DIR is used.  If the
          environment variable is not set the value defaults to
          '../../../../Testing/Temporary'.

    -V /path/to/validated/output.png
    --validated-output /path/to/valid/output.png

          This is a path to a file (usually but not always an image)
          which is compared to data generated by the test.

    -v level
    --verbose level

          Sets the verbosity of the test runner.  Valid values are 0,
          1, and 2 in increasing order of verbosity.

    -I
    --interact

          Interacts with the user when chosen.  If this is not chosen
          the test will run and exit as soon as it is finished.  When
          enabled, the behavior of this is rather trivial and works
          best when the test uses Tkinter.

    -n
    --no-image

          Does not do any image comparisons.  This is useful if you
          want to run the test and not worry about test images or
          image failures etc.

    -h
    --help

                 Prints this message.

"""
    return msg


def parseCmdLine():
    arguments = sys.argv[1:]

    options = "B:D:T:V:v:hnI"
    long_options = ['baseline-root=', 'data-dir=', 'temp-dir=',
                    'validated-output=', 'verbose=', 'help',
                    'no-image', 'interact']

    try:
        opts, args = getopt.getopt(arguments, options, long_options)
    except getopt.error as msg:
        print(usage())
        print('-'*70)
        print(msg)
        sys.exit (1)

    return opts, args


def processCmdLine():
    opts, args = parseCmdLine()

    global VTK_DATA_ROOT, VTK_BASELINE_ROOT, VTK_TEMP_DIR, VTK_BASELINE_PATHS
    global _VERBOSE, _NO_IMAGE, _INTERACT

    # setup defaults
    try:
        VTK_DATA_ROOT = os.environ['VTK_DATA_ROOT']
    except KeyError:
        VTK_DATA_ROOT = os.path.normpath("../../../../../VTKData")

    try:
        VTK_BASELINE_ROOT = os.environ['VTK_BASELINE_ROOT']
    except KeyError:
        pass

    try:
        VTK_TEMP_DIR = os.environ['VTK_TEMP_DIR']
    except KeyError:
        VTK_TEMP_DIR = os.path.normpath("../../../../Testing/Temporary")

    for o, a in opts:
        if o in ('-D', '--data-dir'):
            oa = os.path.abspath(a)
            if os.path.isfile(oa):
                VTK_DATA_PATHS.append(oa)
            else:
                VTK_DATA_ROOT = oa
        if o in ('-B', '--baseline-root'):
            VTK_BASELINE_ROOT = os.path.abspath(a)
        if o in ('-T', '--temp-dir'):
            VTK_TEMP_DIR = os.path.abspath(a)
        if o in ('-V', '--validated-output'):
            VTK_BASELINE_PATHS.append(os.path.abspath(a))
        if o in ('-n', '--no-image'):
            _NO_IMAGE = 1
        if o in ('-I', '--interact'):
            _INTERACT = 1
        if o in ('-v', '--verbose'):
            try:
                _VERBOSE = int(a)
            except:
                msg="Verbosity should be an integer.  0, 1, 2 are valid."
                print(msg)
                sys.exit(1)
        if o in ('-h', '--help'):
            print(usage())
            sys.exit()

    if not VTK_BASELINE_ROOT: # default value.
        VTK_BASELINE_ROOT = VTK_DATA_ROOT



if __name__ == "__main__":
    ######################################################################
    # A Trivial test case to illustrate how this module works.
    class SampleTest(vtkTest):
        obj = vtk.vtkActor()
        def testParse(self):
            "Test if class is parseable"
            self._testParse(self.obj)

        def testGetSet(self):
            "Testing Get/Set methods"
            self._testGetSet(self.obj)

        def testBoolean(self):
            "Testing Boolean methods"
            self._testBoolean(self.obj)

    # Test with the above trivial sample test.
    main( [ (SampleTest, 'test') ] )

