.. _create_partitioned_data_set:

Creating a PartitionedDataSet
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a PartitionedDataSet.

"""
import tempfile

from vtkmodules.vtkIOParallelXML import vtkXMLPartitionedDataSetWriter
from vtkmodules.vtkIOXML import vtkXMLPartitionedDataSetReader

import pyvista as pv

input_data = pv.PartitionedDataSet()

partition1 = pv.Wavelet(extent=(0, 10, 0, 10, 0, 5))
partition2 = pv.Wavelet(extent=(0, 10, 0, 10, 5, 10))

input_data.SetPartition(0, partition1)
input_data.SetPartition(1, partition2)

with tempfile.TemporaryDirectory() as tmpdir:
    file_name = tmpdir + "/testxmlpartds.vtpd"
    w = vtkXMLPartitionedDataSetWriter()
    w.SetInputData(input_data)
    w.SetFileName(file_name)
    w.Write()

    r = vtkXMLPartitionedDataSetReader()
    r.SetFileName(file_name)
    r.Update()
    output_data = pv.wrap(r.GetOutputDataObject(0))

assert isinstance(output_data, pv.PartitionedDataSet)
assert output_data.GetNumberOfPartitions() == 2

for i in range(output_data.GetNumberOfPartitions()):
    assert isinstance(pv.wrap(output_data.GetPartition(i)), pv.ImageData)
    assert (
        output_data.GetPartition(i).GetNumberOfCells()
        == input_data.GetPartition(i).GetNumberOfCells()
    )
