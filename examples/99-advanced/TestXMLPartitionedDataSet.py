import tempfile

from vtkmodules import vtkCommonDataModel as dm
from vtkmodules.vtkIOParallelXML import vtkXMLPartitionedDataSetWriter
from vtkmodules.vtkIOXML import vtkXMLPartitionedDataSetReader

import pyvista as pv

p = dm.vtkPartitionedDataSet()

partition1 = pv.Wavelet(extent=(0, 10, 0, 10, 0, 5))
partition2 = pv.Wavelet(extent=(0, 10, 0, 10, 5, 10))

p.SetPartition(0, partition1)
p.SetPartition(1, partition2)

with tempfile.TemporaryDirectory() as tmpdir:
    fname = tmpdir + "/testxmlpartds.vtpd"
    w = vtkXMLPartitionedDataSetWriter()
    w.SetInputData(p)
    w.SetFileName(fname)
    w.Write()

    r = vtkXMLPartitionedDataSetReader()
    r.SetFileName(fname)
    r.Update()
    o = r.GetOutputDataObject(0)

    assert o.IsA("vtkPartitionedDataSet")
    np = o.GetNumberOfPartitions()
    assert np == 2

    for i in range(np):
        d = o.GetPartition(i)
        d2 = p.GetPartition(i)
        assert d.IsA("vtkImageData")
        assert d.GetNumberOfCells() == d2.GetNumberOfCells()
