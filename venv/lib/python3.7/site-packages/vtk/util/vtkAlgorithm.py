from vtk.vtkCommonDataModel import vtkDataObject
from vtk.vtkCommonExecutionModel import vtkAlgorithm
from vtk.vtkCommonExecutionModel import vtkDemandDrivenPipeline
from vtk.vtkCommonExecutionModel import vtkStreamingDemandDrivenPipeline
from vtk.vtkFiltersPython import vtkPythonAlgorithm

class VTKAlgorithm(object):
    """This is a superclass which can be derived to implement
    Python classes that work with vtkPythonAlgorithm. It implements
    Initialize(), ProcessRequest(), FillInputPortInformation() and
    FillOutputPortInformation().

    Initialize() sets the input and output ports based on data
    members.

    ProcessRequest() calls RequestXXX() methods to implement
    various pipeline passes.

    FillInputPortInformation() and FillOutputPortInformation() set
    the input and output types based on data members.
    """

    def __init__(self, nInputPorts=1, inputType='vtkDataSet',
                       nOutputPorts=1, outputType='vtkPolyData'):
        """Sets up default NumberOfInputPorts, NumberOfOutputPorts,
        InputType and OutputType that are used by various initialization
        methods."""

        self.NumberOfInputPorts = nInputPorts
        self.NumberOfOutputPorts = nOutputPorts
        self.InputType = inputType
        self.OutputType = outputType

    def Initialize(self, vtkself):
        """Sets up number of input and output ports based on
        NumberOfInputPorts and NumberOfOutputPorts."""

        vtkself.SetNumberOfInputPorts(self.NumberOfInputPorts)
        vtkself.SetNumberOfOutputPorts(self.NumberOfOutputPorts)

    def GetInputData(self, inInfo, i, j):
        """Convenience method that returns an input data object
        given a vector of information objects and two indices."""

        return inInfo[i].GetInformationObject(j).Get(vtkDataObject.DATA_OBJECT())

    def GetOutputData(self, outInfo, i):
        """Convenience method that returns an output data object
        given an information object and an index."""
        return outInfo.GetInformationObject(i).Get(vtkDataObject.DATA_OBJECT())

    def RequestDataObject(self, vtkself, request, inInfo, outInfo):
        """Overwritten by subclass to manage data object creation.
        There is not need to overwrite this class if the output can
        be created based on the OutputType data member."""
        return 1

    def RequestInformation(self, vtkself, request, inInfo, outInfo):
        """Overwritten by subclass to provide meta-data to downstream
        pipeline."""
        return 1

    def RequestUpdateExtent(self, vtkself, request, inInfo, outInfo):
        """Overwritten by subclass to modify data request going
        to upstream pipeline."""
        return 1

    def RequestData(self, vtkself, request, inInfo, outInfo):
        """Overwritten by subclass to execute the algorithm."""
        raise NotImplementedError('RequestData must be implemented')

    def ProcessRequest(self, vtkself, request, inInfo, outInfo):
        """Splits a request to RequestXXX() methods."""
        if request.Has(vtkDemandDrivenPipeline.REQUEST_DATA_OBJECT()):
            return self.RequestDataObject(vtkself, request, inInfo, outInfo)
        elif request.Has(vtkDemandDrivenPipeline.REQUEST_INFORMATION()):
            return self.RequestInformation(vtkself, request, inInfo, outInfo)
        elif request.Has(vtkStreamingDemandDrivenPipeline.REQUEST_UPDATE_EXTENT()):
            return self.RequestUpdateExtent(vtkself, request, inInfo, outInfo)
        elif request.Has(vtkDemandDrivenPipeline.REQUEST_DATA()):
            return self.RequestData(vtkself, request, inInfo, outInfo)

        return 1

    def FillInputPortInformation(self, vtkself, port, info):
        """Sets the required input type to InputType."""
        info.Set(vtkAlgorithm.INPUT_REQUIRED_DATA_TYPE(), self.InputType)
        return 1

    def FillOutputPortInformation(self, vtkself, port, info):
        """Sets the default output type to OutputType."""
        info.Set(vtkDataObject.DATA_TYPE_NAME(), self.OutputType)
        return 1

class VTKPythonAlgorithmBase(vtkPythonAlgorithm):
    """This is a superclass which can be derived to implement
    Python classes that act as VTK algorithms in a VTK pipeline.
    It implements ProcessRequest(), FillInputPortInformation() and
    FillOutputPortInformation().

    ProcessRequest() calls RequestXXX() methods to implement
    various pipeline passes.

    FillInputPortInformation() and FillOutputPortInformation() set
    the input and output types based on data members.

    Common use is something like this:

    class HDF5Source(VTKPythonAlgorithmBase):
        def __init__(self):
            VTKPythonAlgorithmBase.__init__(self,
                nInputPorts=0,
                nOutputPorts=1, outputType='vtkImageData')

        def RequestInformation(self, request, inInfo, outInfo):
            f = h5py.File("foo.h5", 'r')
            dims = f['RTData'].shape[::-1]
            info = outInfo.GetInformationObject(0)
            info.Set(vtk.vtkStreamingDemandDrivenPipeline.WHOLE_EXTENT(),
                (0, dims[0]-1, 0, dims[1]-1, 0, dims[2]-1), 6)
            return 1

        def RequestData(self, request, inInfo, outInfo):
            f = h5py.File("foo.h5", 'r')
            data = f['RTData'][:]
            output = dsa.WrapDataObject(vtk.vtkImageData.GetData(outInfo))
            output.SetDimensions(data.shape)
            output.PointData.append(data.flatten(), 'RTData')
            output.PointData.SetActiveScalars('RTData')
            return 1

    alg = HDF5Source()

    cf = vtk.vtkContourFilter()
    cf.SetInputConnection(alg.GetOutputPort())
    cf.Update()
    """

    class InternalAlgorithm(object):
        "Internal class. Do not use."
        def Initialize(self, vtkself):
            pass

        def FillInputPortInformation(self, vtkself, port, info):
            return vtkself.FillInputPortInformation(port, info)

        def FillOutputPortInformation(self, vtkself, port, info):
            return vtkself.FillOutputPortInformation(port, info)

        def ProcessRequest(self, vtkself, request, inInfo, outInfo):
            return vtkself.ProcessRequest(request, inInfo, outInfo)

    def __init__(self, nInputPorts=1, inputType='vtkDataSet',
                       nOutputPorts=1, outputType='vtkPolyData'):
        """Sets up default NumberOfInputPorts, NumberOfOutputPorts,
        InputType and OutputType that are used by various methods.
        Make sure to call this method from any subclass' __init__"""

        self.SetPythonObject(VTKPythonAlgorithmBase.InternalAlgorithm())

        self.SetNumberOfInputPorts(nInputPorts)
        self.SetNumberOfOutputPorts(nOutputPorts)

        self.InputType = inputType
        self.OutputType = outputType

    def GetInputData(self, inInfo, i, j):
        """Convenience method that returns an input data object
        given a vector of information objects and two indices."""

        return inInfo[i].GetInformationObject(j).Get(vtkDataObject.DATA_OBJECT())

    def GetOutputData(self, outInfo, i):
        """Convenience method that returns an output data object
        given an information object and an index."""
        return outInfo.GetInformationObject(i).Get(vtkDataObject.DATA_OBJECT())

    def FillInputPortInformation(self, port, info):
        """Sets the required input type to InputType."""
        info.Set(vtkAlgorithm.INPUT_REQUIRED_DATA_TYPE(), self.InputType)
        return 1

    def FillOutputPortInformation(self, port, info):
        """Sets the default output type to OutputType."""
        info.Set(vtkDataObject.DATA_TYPE_NAME(), self.OutputType)
        return 1

    def ProcessRequest(self, request, inInfo, outInfo):
        """Splits a request to RequestXXX() methods."""
        if request.Has(vtkDemandDrivenPipeline.REQUEST_DATA_OBJECT()):
            return self.RequestDataObject(request, inInfo, outInfo)
        elif request.Has(vtkDemandDrivenPipeline.REQUEST_INFORMATION()):
            return self.RequestInformation(request, inInfo, outInfo)
        elif request.Has(vtkStreamingDemandDrivenPipeline.REQUEST_UPDATE_EXTENT()):
            return self.RequestUpdateExtent(request, inInfo, outInfo)
        elif request.Has(vtkDemandDrivenPipeline.REQUEST_DATA()):
            return self.RequestData(request, inInfo, outInfo)

        return 1

    def RequestDataObject(self, request, inInfo, outInfo):
        """Overwritten by subclass to manage data object creation.
        There is not need to overwrite this class if the output can
        be created based on the OutputType data member."""
        return 1

    def RequestInformation(self, request, inInfo, outInfo):
        """Overwritten by subclass to provide meta-data to downstream
        pipeline."""
        return 1

    def RequestUpdateExtent(self, request, inInfo, outInfo):
        """Overwritten by subclass to modify data request going
        to upstream pipeline."""
        return 1

    def RequestData(self, request, inInfo, outInfo):
        """Overwritten by subclass to execute the algorithm."""
        raise NotImplementedError('RequestData must be implemented')
