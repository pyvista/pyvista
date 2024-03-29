"""Contains the PartitionedDataSet class."""

from . import _vtk_core as _vtk


class PartitionedDataSet(_vtk.vtkPartitionedDataSet):
    """DataSet which composite dataset to encapsulates a dataset consisting of partitions."""
