from vtkmodules import (
    vtkCommonCore as cc,
    vtkCommonDataModel as dm,
    vtkCommonExecutionModel as em,
    vtkImagingCore as ic,
)
from vtkmodules.test import Testing
from vtkmodules.util.vtkAlgorithm import VTKPythonAlgorithmBase

import pyvista as pv


class SimpleFilter(VTKPythonAlgorithmBase):
    def __init__(self):
        VTKPythonAlgorithmBase.__init__(self)
        self.InputType = "vtkDataSet"
        self.OutputType = "vtkDataObject"
        self.Counter = 0

    def FillInputPortInformation(self, port, info):
        """Sets the required input type to InputType."""
        info.Set(em.vtkAlgorithm.INPUT_REQUIRED_DATA_TYPE(), self.InputType)
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


class PartitionAwareFilter(VTKPythonAlgorithmBase):
    def __init__(self):
        VTKPythonAlgorithmBase.__init__(self)
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


class PartitionCollectionAwareFilter(VTKPythonAlgorithmBase):
    def __init__(self):
        VTKPythonAlgorithmBase.__init__(self)
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


class CompositeAwareFilter(VTKPythonAlgorithmBase):
    def __init__(self):
        VTKPythonAlgorithmBase.__init__(self)
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


class TestPartitionedData(Testing.vtkTest):

    def test(self):

        p = dm.vtkPartitionedDataSet()

        s = ic.vtkRTAnalyticSource()
        s.SetWholeExtent(0, 10, 0, 10, 0, 5)
        s.Update()

        p1 = pv.ImageData()
        p1.ShallowCopy(s.GetOutput())

        s.SetWholeExtent(0, 10, 0, 10, 5, 10)
        s.Update()

        p2 = pv.ImageData()
        p2.ShallowCopy(s.GetOutput())

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
        for i in (0, 1):
            pdsc = sf.GetOutputDataObject(0)
            assert pdsc.GetClassName() == "vtkPartitionedDataSetCollection"
            pds = pdsc.GetPartitionedDataSet(i)
            assert pds.GetClassName() == "vtkPartitionedDataSet"
            assert pds.GetNumberOfPartitions() == 2
            for j in (0, 1):
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
        for i in (0, 1):
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


if __name__ == "__main__":
    Testing.main([(TestPartitionedData, 'test')])
