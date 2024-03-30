"""Contains the PartitionedDataSet class."""

import collections.abc

from . import _vtk_core as _vtk
from .dataset import DataObject
from .utilities.helpers import is_pyvista_dataset, wrap


class PartitionedDataSet(_vtk.vtkPartitionedDataSet, DataObject, collections.abc.MutableSequence):  # type: ignore[type-arg]
    """Wrapper for the ``vtkPartitionedDataSet`` class.

    DataSet which composite dataset to encapsulates a dataset consisting of partitions.

    Examples
    --------
    >>> import pyvista as pv

    """

    if _vtk.vtk_version_info >= (9, 1):
        _WRITERS = {".vtpd": _vtk.vtkXMLPartitionedDataSetWriter}

    def __init__(self, *args, **kwargs):
        """Initialize the PartitionedDataSet."""
        super().__init__(*args, **kwargs)
        if len(args) == 1:
            if isinstance(args[0], _vtk.vtkPartitionedDataSet):
                deep = kwargs.get('deep', True)
                if deep:
                    self.deep_copy(args[0])
                else:
                    self.shallow_copy(args[0])

        # Upon creation make sure all nested structures are wrapped
        self.wrap_nested()

    def wrap_nested(self):
        """Ensure that all nested data structures are wrapped as PyVista datasets.

        This is performed in place.

        """
        for i in range(self.n_partitions):
            block = self.GetPartition(i)
            if not is_pyvista_dataset(block):
                self.SetPartition(i, wrap(block))

    def __len__(self) -> int:
        """Return the number of partitions."""
        return self.n_partitions

    @property
    def n_partitions(self) -> int:
        """Return the number of partitions.

        Returns
        -------
        int
            The number of partitions.
        """
        return self.GetNumberOfPartitions()
