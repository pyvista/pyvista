"""Contains the PartitionedDataSet class."""

import collections.abc
from typing import Iterable, Optional, Union, overload

from . import _vtk_core as _vtk
from .dataset import DataObject, DataSet
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
            elif isinstance(args[0], (list, tuple)):
                for partition in args[0]:
                    self.append(partition)

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

    @overload
    def __getitem__(self, index: int) -> Optional[DataSet]:  # noqa: D105
        ...

    @overload
    def __getitem__(self, index: slice) -> 'PartitionedDataSet':  # noqa: D105
        ...

    def __getitem__(self, index):
        """Get a block by its index."""
        if isinstance(index, slice):
            partitions = PartitionedDataSet()
            for i in range(self.n_partitions)[index]:
                partitions.append(self[i])
            return partitions
        else:
            return wrap(self.GetPartition(index))

    @overload
    def __setitem__(self, index: int, data: Optional[DataSet]):  # noqa: D105
        ...

    @overload
    def __setitem__(self, index: slice, data: Iterable[Optional[DataSet]]):  # noqa: D105
        ...

    def __setitem__(
        self,
        index: Union[int, slice],
        data,
    ):
        """Set a block with a VTK data object."""
        self.SetPartition(index, data)

    @property
    def n_partitions(self) -> int:
        """Return the number of partitions.

        Returns
        -------
        int
            The number of partitions.
        """
        return self.GetNumberOfPartitions()

    @n_partitions.setter
    def n_partitions(self, n):  # numpydoc ignore=GL08
        self.SetNumberOfPartitions(n)
        self.Modified()

    def append(self, dataset):
        """Add a data set to the next partition index.

        Parameters
        ----------
        dataset : pyvista.DataSet
            Dataset to append to this partitioned dataset.
        """
        index = self.n_partitions
        self.n_partitions += 1
        self[index] = dataset
