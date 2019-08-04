"""Miscellaneous functions and classes that don't fit into specific
categories."""

import sys, os, vtk

def calldata_type(type):
    """set_call_data_type(type) -- convenience decorator to easily set the CallDataType attribute
    for python function used as observer callback.
    For example:

    import vtk

    @vtk.calldata_type(vtk.VTK_STRING)
    def onError(caller, event, calldata):
        print("caller: %s - event: %s - msg: %s" % (caller.GetClassName(), event, calldata))

    lt = vtk.vtkLookupTable()
    lt.AddObserver(vtk.vtkCommand.ErrorEvent, onError)
    lt.SetTableRange(2,1)
    """
    supported_call_data_types = ['string0', vtk.VTK_STRING, vtk.VTK_OBJECT, vtk.VTK_INT, vtk.VTK_LONG, vtk.VTK_DOUBLE, vtk.VTK_FLOAT]
    if type not in supported_call_data_types:
        raise TypeError("'%s' is not a supported VTK call data type. Supported types are: %s" % (type, supported_call_data_types))
    def wrap(f):
        f.CallDataType = type
        return f
    return wrap

#----------------------------------------------------------------------
# the following functions are for the vtk regression testing and examples

def vtkGetDataRoot():
    """vtkGetDataRoot() -- return vtk example data directory
    """
    dataIndex=-1;
    for i in range(0, len(sys.argv)):
        if sys.argv[i] == '-D' and i < len(sys.argv)-1:
            dataIndex = i+1

    if dataIndex != -1:
        dataRoot = sys.argv[dataIndex]
    else:
        try:
            dataRoot = os.environ['VTK_DATA_ROOT']
        except KeyError:
            dataRoot = '../../../../VTKData'

    return dataRoot

def vtkGetTempDir():
    """vtkGetTempDir() -- return vtk testing temp dir
    """
    tempIndex=-1;
    for i in range(0, len(sys.argv)):
        if sys.argv[i] == '-T' and i < len(sys.argv)-1:
            tempIndex = i+1

    if tempIndex != -1:
        tempDir = sys.argv[tempIndex]
    else:
        tempDir = '.'

    return tempDir


def vtkRegressionTestImage( renWin ):
    """vtkRegressionTestImage(renWin) -- produce regression image for window

    This function writes out a regression .png file for a vtkWindow.
    Does anyone involved in testing care to elaborate?
    """
    imageIndex=-1;
    for i in range(0, len(sys.argv)):
        if sys.argv[i] == '-V' and i < len(sys.argv)-1:
            imageIndex = i+1

    if imageIndex != -1:
        fname = os.path.join(vtkGetDataRoot(), sys.argv[imageIndex])

        rt_w2if = vtk.vtkWindowToImageFilter()
        rt_w2if.SetInput(renWin)

        if os.path.isfile(fname):
            pass
        else:
            rt_pngw = vtk.vtkPNGWriter()
            rt_pngw.SetFileName(fname)
            rt_pngw.SetInputConnection(rt_w2if.GetOutputPort())
            rt_pngw.Write()
            rt_pngw = None

        rt_png = vtk.vtkPNGReader()
        rt_png.SetFileName(fname)

        rt_id = vtk.vtkImageDifference()
        rt_id.SetInputConnection(rt_w2if.GetOutputPort())
        rt_id.SetImageConnection(rt_png.GetOutputPort())
        rt_id.Update()

        if rt_id.GetThresholdedError() <= 10:
            return 1
        else:
            sys.stderr.write('Failed image test: %f\n'
                             % rt_id.GetThresholdedError())
            return 0
    return 2
