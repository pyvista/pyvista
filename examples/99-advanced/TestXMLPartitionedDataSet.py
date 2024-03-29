import tempfile

from vtkmodules.vtkCommonDataModel import vtkPartitionedDataSet
from vtkmodules.vtkIOParallelXML import vtkXMLPartitionedDataSetWriter
from vtkmodules.vtkIOXML import vtkXMLPartitionedDataSetReader

import pyvista as pv

dataset = vtkPartitionedDataSet()

partition1 = pv.Wavelet(extent=(0, 10, 0, 10, 0, 5))
partition2 = pv.Wavelet(extent=(0, 10, 0, 10, 5, 10))

dataset.SetPartition(0, partition1)
dataset.SetPartition(1, partition2)

with tempfile.TemporaryDirectory() as tmpdir:
    file_name = tmpdir + "/testxmlpartds.vtpd"
    w = vtkXMLPartitionedDataSetWriter()
    w.SetInputData(dataset)
    w.SetFileName(file_name)
    w.Write()

    r = vtkXMLPartitionedDataSetReader()
    r.SetFileName(file_name)
    r.Update()

dataset_ = r.GetOutputDataObject(0)

number_of_partitions = dataset_.GetNumberOfPartitions()
assert dataset_.IsA("vtkPartitionedDataSet")
assert number_of_partitions == 2

for i in range(number_of_partitions):
    partition = dataset.GetPartition(i)
    partition_ = dataset_.GetPartition(i)
    assert partition_.IsA("vtkImageData")
    assert partition_.GetNumberOfCells() == partition.GetNumberOfCells()
