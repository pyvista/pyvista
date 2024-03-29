import tempfile

from vtkmodules.vtkCommonDataModel import vtkPartitionedDataSet
from vtkmodules.vtkIOParallelXML import vtkXMLPartitionedDataSetWriter
from vtkmodules.vtkIOXML import vtkXMLPartitionedDataSetReader

import pyvista as pv

p = vtkPartitionedDataSet()

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

number_of_partitions = o.GetNumberOfPartitions()
assert o.IsA("vtkPartitionedDataSet")
assert number_of_partitions == 2

for i in range(number_of_partitions):
    partition = o.GetPartition(i)
    partition_ = p.GetPartition(i)
    assert partition.IsA("vtkImageData")
    assert partition.GetNumberOfCells() == partition_.GetNumberOfCells()
