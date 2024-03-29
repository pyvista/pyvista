from vtkmodules import vtkCommonCore as cc, vtkCommonDataModel as dm, vtkCommonExecutionModel as em

import pyvista as pv
from pyvista.plotting.utilities.algorithms import PreserveTypeAlgorithmBase


class SimpleFilter(PreserveTypeAlgorithmBase):
    def __init__(self):
        PreserveTypeAlgorithmBase.__init__(self)
        self.InputType = "vtkDataSet"
        self.OutputType = "vtkDataObject"
        self.Counter = 0

    def FillInputPortInformation(self, port, info):
        """Sets the required input type to InputType."""
        info.Set(em.vtkAlgorithm.INPUT_REQUIRED_DATA_TYPE(), self.InputType)
        return 1

    def RequestDataObject(self, request, inInfo, outInfo):
        """Preserve data type.

        Parameters
        ----------
        _request : vtk.vtkInformation
            The request object for the filter.

        inInfo : vtk.vtkInformationVector
            The input information vector for the filter.

        outInfo : vtk.vtkInformationVector
            The output information vector for the filter.

        Returns
        -------
        int
            Returns 1 if successful.
        """
        inp = dm.vtkDataObject.GetData(inInfo[0])
        opt = dm.vtkDataObject.GetData(outInfo)

        if opt and opt.IsA(inp.GetClassName()):
            return 1

        opt = inp.NewInstance()
        outInfo.GetInformationObject(0).Set(dm.vtkDataObject.DATA_OBJECT(), opt)
        return 1

    def RequestData(self, _request, inInfo, outInfo):
        """Perform algorithm execution.

        Parameters
        ----------
        _request : vtk.vtkInformation
            The request object.

        inInfo : vtk.vtkInformationVector
            Information about the input data.

        outInfo : vtk.vtkInformationVector
            Information about the output data.

        Returns
        -------
        int
            1 on success.

        """
        inp = dm.vtkDataObject.GetData(inInfo[0])
        print("SimpleFilter iter %d: Input: %s" % (self.Counter, inp.GetClassName()))
        opt = dm.vtkDataObject.GetData(outInfo)
        a = cc.vtkTypeUInt8Array()
        a.SetName("counter")
        a.SetNumberOfTuples(1)
        a.SetValue(0, self.Counter)
        a.GetInformation().Set(dm.vtkDataObject.DATA_TYPE_NAME(), inp.GetClassName())
        opt.GetFieldData().AddArray(a)
        self.Counter += 1
        return 1


class PartitionAwareFilter(PreserveTypeAlgorithmBase):
    def __init__(self):
        PreserveTypeAlgorithmBase.__init__(self)
        self.InputType = "vtkDataSet"
        self.OutputType = "vtkDataObject"
        self.Counter = 0

    def FillInputPortInformation(self, port, info):
        """Sets the required input type to InputType."""
        info.Set(em.vtkAlgorithm.INPUT_REQUIRED_DATA_TYPE(), self.InputType)
        info.Append(em.vtkAlgorithm.INPUT_REQUIRED_DATA_TYPE(), "vtkPartitionedDataSet")
        return 1

    def RequestDataObject(self, request, inInfo, outInfo):
        inp = dm.vtkDataObject.GetData(inInfo[0])
        opt = dm.vtkDataObject.GetData(outInfo)

        if opt and opt.IsA(inp.GetClassName()):
            return 1

        opt = inp.NewInstance()
        outInfo.GetInformationObject(0).Set(dm.vtkDataObject.DATA_OBJECT(), opt)
        return 1

    def RequestData(self, request, inInfo, outInfo):
        inp = dm.vtkDataObject.GetData(inInfo[0])
        print("PartitionAwareFilter iter %d: Input: %s" % (self.Counter, inp.GetClassName()))
        opt = dm.vtkDataObject.GetData(outInfo)
        a = cc.vtkTypeUInt8Array()
        a.SetName("counter")
        a.SetNumberOfTuples(1)
        a.SetValue(0, self.Counter)
        a.GetInformation().Set(dm.vtkDataObject.DATA_TYPE_NAME(), inp.GetClassName())
        opt.GetFieldData().AddArray(a)
        self.Counter += 1
        return 1


class PartitionCollectionAwareFilter(PreserveTypeAlgorithmBase):
    def __init__(self):
        PreserveTypeAlgorithmBase.__init__(self)
        self.InputType = "vtkDataSet"
        self.OutputType = "vtkDataObject"
        self.Counter = 0

    def FillInputPortInformation(self, port, info):
        """Sets the required input type to InputType."""
        info.Set(em.vtkAlgorithm.INPUT_REQUIRED_DATA_TYPE(), self.InputType)
        info.Append(em.vtkAlgorithm.INPUT_REQUIRED_DATA_TYPE(), "vtkPartitionedDataSetCollection")
        return 1

    def RequestDataObject(self, request, inInfo, outInfo):
        inp = dm.vtkDataObject.GetData(inInfo[0])
        opt = dm.vtkDataObject.GetData(outInfo)

        if opt and opt.IsA(inp.GetClassName()):
            return 1

        opt = inp.NewInstance()
        outInfo.GetInformationObject(0).Set(dm.vtkDataObject.DATA_OBJECT(), opt)
        return 1

    def RequestData(self, request, inInfo, outInfo):
        inp = dm.vtkDataObject.GetData(inInfo[0])
        print(
            "PartitionCollectionAwareFilter iter %d: Input: %s" % (self.Counter, inp.GetClassName())
        )
        opt = dm.vtkDataObject.GetData(outInfo)
        a = cc.vtkTypeUInt8Array()
        a.SetName("counter")
        a.SetNumberOfTuples(1)
        a.SetValue(0, self.Counter)
        a.GetInformation().Set(dm.vtkDataObject.DATA_TYPE_NAME(), inp.GetClassName())
        opt.GetFieldData().AddArray(a)
        self.Counter += 1
        return 1


class CompositeAwareFilter(PreserveTypeAlgorithmBase):
    def __init__(self):
        PreserveTypeAlgorithmBase.__init__(self)
        self.InputType = "vtkDataSet"
        self.OutputType = "vtkDataObject"
        self.Counter = 0

    def FillInputPortInformation(self, port, info):
        """Sets the required input type to InputType."""
        info.Set(em.vtkAlgorithm.INPUT_REQUIRED_DATA_TYPE(), self.InputType)
        info.Append(em.vtkAlgorithm.INPUT_REQUIRED_DATA_TYPE(), "vtkCompositeDataSet")
        return 1

    def RequestDataObject(self, request, inInfo, outInfo):
        inp = dm.vtkDataObject.GetData(inInfo[0])
        opt = dm.vtkDataObject.GetData(outInfo)

        if opt and opt.IsA(inp.GetClassName()):
            return 1

        opt = inp.NewInstance()
        outInfo.GetInformationObject(0).Set(dm.vtkDataObject.DATA_OBJECT(), opt)
        return 1

    def RequestData(self, request, inInfo, outInfo):
        inp = dm.vtkDataObject.GetData(inInfo[0])
        print("CompositeAwareFilter iter %d: Input: %s" % (self.Counter, inp.GetClassName()))
        opt = dm.vtkDataObject.GetData(outInfo)
        a = cc.vtkTypeUInt8Array()
        a.SetName("counter")
        a.SetNumberOfTuples(1)
        a.SetValue(0, self.Counter)
        a.GetInformation().Set(dm.vtkDataObject.DATA_TYPE_NAME(), inp.GetClassName())
        opt.GetFieldData().AddArray(a)
        self.Counter += 1
        return 1


if __name__ == "__main__":
    p = dm.vtkPartitionedDataSet()

    wavelet1 = pv.Wavelet(extent=(0, 10, 0, 10, 0, 5))

    p1 = pv.ImageData()
    p1.ShallowCopy(wavelet1)

    wavelet2 = pv.Wavelet(extent=(0, 10, 0, 10, 5, 10))

    p2 = pv.ImageData()
    p2.ShallowCopy(wavelet2)

    p.SetPartition(0, p1)
    p.SetPartition(1, p2)

    p2 = dm.vtkPartitionedDataSet()
    p2.ShallowCopy(p)

    c = dm.vtkPartitionedDataSetCollection()
    c.SetPartitionedDataSet(0, p)
    c.SetPartitionedDataSet(1, p2)

    # SimpleFilter:
    sf = SimpleFilter()
    sf.SetInputDataObject(c)
    sf.Update()
    assert sf.GetOutputDataObject(0).GetNumberOfPartitionedDataSets() == 2
    for i in range(2):
        pdsc = sf.GetOutputDataObject(0)
        assert pdsc.GetClassName() == "vtkPartitionedDataSetCollection"
        pds = pdsc.GetPartitionedDataSet(i)
        assert pds.GetClassName() == "vtkPartitionedDataSet"
        assert pds.GetNumberOfPartitions() == 2
        for j in range(2):
            part = pds.GetPartition(j)
            countArray = part.GetFieldData().GetArray("counter")
            info = countArray.GetInformation()
            assert countArray.GetValue(0) == i * 2 + j
            assert info.Get(dm.vtkDataObject.DATA_TYPE_NAME()) == "vtkImageData"

    # PartitionAwareFilter
    pf = PartitionAwareFilter()
    pf.SetInputDataObject(c)
    pf.Update()
    assert pf.GetOutputDataObject(0).GetNumberOfPartitionedDataSets() == 2
    for i in range(2):
        pdsc = pf.GetOutputDataObject(0)
        assert pdsc.GetClassName() == "vtkPartitionedDataSetCollection"
        pds = pdsc.GetPartitionedDataSet(i)
        assert pds.GetClassName() == "vtkPartitionedDataSet"
        assert pds.GetNumberOfPartitions() == 0
        countArray = pds.GetFieldData().GetArray("counter")
        info = countArray.GetInformation()
        assert countArray.GetValue(0) == i
        assert info.Get(dm.vtkDataObject.DATA_TYPE_NAME()) == "vtkPartitionedDataSet"

    # PartitionCollectionAwareFilter
    pcf = PartitionCollectionAwareFilter()
    pcf.SetInputDataObject(c)
    pcf.Update()
    assert pcf.GetOutputDataObject(0).GetNumberOfPartitionedDataSets() == 0
    pdsc = pcf.GetOutputDataObject(0)
    assert pdsc.GetClassName() == "vtkPartitionedDataSetCollection"
    countArray = pdsc.GetFieldData().GetArray("counter")
    info = countArray.GetInformation()
    assert countArray.GetValue(0) == 0
    assert info.Get(dm.vtkDataObject.DATA_TYPE_NAME()) == "vtkPartitionedDataSetCollection"

    # CompositeAwareFilter
    cf = CompositeAwareFilter()
    cf.SetInputDataObject(c)
    cf.Update()
    assert pcf.GetOutputDataObject(0).GetNumberOfPartitionedDataSets() == 0
    pdsc = pcf.GetOutputDataObject(0)
    assert pdsc.GetClassName() == "vtkPartitionedDataSetCollection"
    countArray = pdsc.GetFieldData().GetArray("counter")
    info = countArray.GetInformation()
    assert countArray.GetValue(0) == 0
    assert info.Get(dm.vtkDataObject.DATA_TYPE_NAME()) == "vtkPartitionedDataSetCollection"
