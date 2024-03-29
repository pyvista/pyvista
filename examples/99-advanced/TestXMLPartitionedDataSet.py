from pathlib import Path

from vtkmodules import vtkCommonDataModel as dm, vtkImagingCore as ic
from vtkmodules.test import Testing
from vtkmodules.util.misc import vtkGetTempDir
from vtkmodules.vtkIOParallelXML import vtkXMLPartitionedDataSetWriter
from vtkmodules.vtkIOXML import vtkXMLPartitionedDataSetReader


class TestXMLPartitionedDataSet(Testing.vtkTest):

    def test(self):

        p = dm.vtkPartitionedDataSet()

        s = ic.vtkRTAnalyticSource()
        s.SetWholeExtent(0, 10, 0, 10, 0, 5)
        s.Update()

        p1 = dm.vtkImageData()
        p1.ShallowCopy(s.GetOutput())

        s.SetWholeExtent(0, 10, 0, 10, 5, 10)
        s.Update()

        p2 = dm.vtkImageData()
        p2.ShallowCopy(s.GetOutput())

        p.SetPartition(0, p1)
        p.SetPartition(1, p2)

        tmpdir = vtkGetTempDir()
        fname = tmpdir + "/testxmlpartds.vtpd"
        w = vtkXMLPartitionedDataSetWriter()
        w.SetInputData(p)
        w.SetFileName(fname)
        w.Write()

        r = vtkXMLPartitionedDataSetReader()
        r.SetFileName(fname)
        r.Update()
        o = r.GetOutputDataObject(0)

        print(o.IsA("vtkPartitionedDataSet"))
        np = o.GetNumberOfPartitions()
        assert np == 2

        for i in range(np):
            d = o.GetPartition(i)
            d2 = p.GetPartition(i)
            assert d.IsA("vtkImageData")
            assert d.GetNumberOfCells() == d2.GetNumberOfCells()
        Path(fname).unlink()


if __name__ == "__main__":
    Testing.main([(TestXMLPartitionedDataSet, 'test')])
