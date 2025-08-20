"""Contains the PartitionedDataSet class."""

from __future__ import annotations

from collections.abc import MutableSequence
from typing import TYPE_CHECKING
from typing import overload

from pyvista._deprecate_positional_args import _deprecate_positional_args

from . import _vtk_core as _vtk
from .dataobject import DataObject
from .errors import PartitionedDataSetsNotSupported
from .utilities.helpers import is_pyvista_dataset
from .utilities.helpers import wrap

if TYPE_CHECKING:
    from collections.abc import Iterable
    from typing import ClassVar

    from typing_extensions import Self

    from .dataset import DataSet
    from .utilities.arrays import FieldAssociation


class PartitionedDataSet(DataObject, MutableSequence, _vtk.vtkPartitionedDataSet):  # type: ignore[type-arg]
    """Wrapper for the :vtk:`vtkPartitionedDataSet` class.

    DataSet which composite dataset to encapsulates a dataset consisting of partitions.

    Examples
    --------
    >>> import pyvista as pv
    >>> data = [
    ...     pv.Sphere(center=(2, 0, 0)),
    ...     pv.Cube(center=(0, 2, 0)),
    ...     pv.Cone(),
    ... ]
    >>> partitions = pv.PartitionedDataSet(data)
    >>> len(partitions)
    3

    """

    _WRITERS: ClassVar[str, _vtk.vtkAlgorithm] = {'.vtpd': _vtk.vtkXMLPartitionedDataSetWriter}

    if _vtk.vtk_version_info >= (9, 4):
        _WRITERS['.vtkhdf'] = _vtk.vtkHDFWriter

    def __init__(self, *args, **kwargs):
        """Initialize the PartitionedDataSet."""
        super().__init__()
        if len(args) == 1:
            if isinstance(args[0], _vtk.vtkPartitionedDataSet):
                deep = kwargs.get('deep', True)
                if deep:
                    self.deep_copy(args[0])
                else:
                    raise PartitionedDataSetsNotSupported
            elif isinstance(args[0], (list, tuple)):
                for partition in args[0]:
                    self.append(partition)
        self.wrap_nested()

    def wrap_nested(self) -> None:
        """Ensure that all nested data structures are wrapped as PyVista datasets.

        This is performed in place.

        """
        for i in range(self.n_partitions):
            partition = self.GetPartition(i)
            if not is_pyvista_dataset(partition):
                self.SetPartition(i, wrap(partition))

    @overload
    def __getitem__(self, index: int) -> DataSet | None: ...  # pragma: no cover

    @overload
    def __getitem__(self, index: slice) -> PartitionedDataSet: ...  # pragma: no cover

    def __getitem__(self, index):
        """Get a partition by its index."""
        if isinstance(index, slice):
            return PartitionedDataSet([self[i] for i in range(self.n_partitions)[index]])
        else:
            if index < -self.n_partitions or index >= self.n_partitions:
                msg = f'index ({index}) out of range for this dataset.'
                raise IndexError(msg)
            if index < 0:
                index = self.n_partitions + index
            return wrap(self.GetPartition(index))

    @overload
    def __setitem__(self, index: int, data: DataSet | None) -> None: ...  # pragma: no cover

    @overload
    def __setitem__(
        self, index: slice, data: Iterable[DataSet | None]
    ) -> None: ...  # pragma: no cover

    def __setitem__(
        self,
        index: int | slice,
        data,
    ):
        """Set a partition with a VTK data object."""
        if isinstance(index, slice):
            for i, d in zip(range(self.n_partitions)[index], data):
                self.SetPartition(i, d)
        else:
            if index < -self.n_partitions or index >= self.n_partitions:
                msg = f'index ({index}) out of range for this dataset.'
                raise IndexError(msg)
            if index < 0:
                index = self.n_partitions + index
            self.SetPartition(index, data)

    def __delitem__(self, index: int | slice) -> None:
        """Remove a partition at the specified index are not supported."""
        raise PartitionedDataSetsNotSupported

    def insert(self, index: int, dataset: DataSet) -> None:  # numpydoc ignore=PR01
        """Insert data before index."""
        index = range(self.n_partitions)[index]
        self.n_partitions += 1
        for i in reversed(range(index, self.n_partitions - 1)):
            self[i + 1] = self[i]
        self[index] = dataset

    def pop(self, index: int = -1) -> None:  # numpydoc ignore=PR01  # noqa: ARG002
        """Pop off a partition at the specified index are not supported."""
        raise PartitionedDataSetsNotSupported

    def _get_attrs(self):
        """Return the representation methods (internal helper)."""
        attrs = []
        attrs.append(('N Partitions', self.n_partitions, '{}'))
        return attrs

    def _repr_html_(self) -> str:
        """Define a pretty representation for Jupyter notebooks."""
        fmt = ''
        fmt += "<table style='width: 100%;'>"
        fmt += '<tr><th>Information</th><th>Partitions</th></tr>'
        fmt += '<tr><td>'
        fmt += '\n'
        fmt += '<table>\n'
        fmt += f'<tr><th>{type(self).__name__}</th><th>Values</th></tr>\n'
        row = '<tr><td>{}</td><td>{}</td></tr>\n'
        for attr in self._get_attrs():
            try:
                fmt += row.format(attr[0], attr[2].format(*attr[1]))
            except TypeError:
                fmt += row.format(attr[0], attr[2].format(attr[1]))
        fmt += '</table>\n'
        fmt += '\n'
        fmt += '</td><td>'
        fmt += '\n'
        fmt += '<table>\n'
        row = '<tr><th>{}</th><th>{}</th></tr>\n'
        fmt += row.format('Index', 'Type')
        for i in range(self.n_partitions):
            data = self[i]
            fmt += row.format(i, type(data).__name__)
        fmt += '</table>\n'
        fmt += '\n'
        fmt += '</td></tr> </table>'
        return fmt

    def __repr__(self) -> str:
        """Define an adequate representation."""
        fmt = f'{type(self).__name__} ({hex(id(self))})\n'
        max_len = max(len(attr[0]) for attr in self._get_attrs()) + 4
        row = f'  {{:{max_len}s}}' + '{}\n'
        for attr in self._get_attrs():
            try:
                fmt += row.format(attr[0], attr[2].format(*attr[1]))
            except TypeError:
                fmt += row.format(attr[0], attr[2].format(attr[1]))
        return fmt.strip()

    def __str__(self) -> str:
        """Return the str representation of the multi partition."""
        return PartitionedDataSet.__repr__(self)

    def __len__(self) -> int:
        """Return the number of partitions."""
        return self.n_partitions

    def copy_meta_from(self, ido, deep) -> None:  # numpydoc ignore=PR01
        """Copy pyvista meta data onto this object from another object."""

    @_deprecate_positional_args
    def copy(self, deep: bool = True):  # noqa: FBT001, FBT002
        """Return a copy of the PartitionedDataSet.

        Parameters
        ----------
        deep : bool, default: True
            When ``True``, make a full copy of the object.

        Returns
        -------
        pyvista.PartitionedDataSet
           Deep or shallow copy of the ``PartitionedDataSet``.

        Examples
        --------
        >>> import pyvista as pv
        >>> data = [
        ...     pv.Sphere(center=(2, 0, 0)),
        ...     pv.Cube(center=(0, 2, 0)),
        ...     pv.Cone(),
        ... ]
        >>> partitions = pv.PartitionedDataSet(data)
        >>> new_partitions = partitions.copy()
        >>> len(new_partitions)
        3

        """
        thistype = type(self)
        newobject = thistype()
        if deep:
            newobject.deep_copy(self)
        else:
            raise PartitionedDataSetsNotSupported
        newobject.copy_meta_from(self, deep)
        newobject.wrap_nested()
        return newobject

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
    def n_partitions(self, n) -> None:
        self.SetNumberOfPartitions(n)
        self.Modified()

    @property
    def is_empty(self) -> bool:  # numpydoc ignore=RT01
        """Return ``True`` if there are no partitions.

        .. versionadded:: 0.46

        Examples
        --------
        >>> import pyvista as pv
        >>> mesh = pv.PartitionedDataSet()
        >>> mesh.is_empty
        True

        >>> mesh.append(pv.Sphere())
        >>> mesh.is_empty
        False

        """
        return self.n_partitions == 0

    def append(self, dataset) -> None:
        """Add a data set to the next partition index.

        Parameters
        ----------
        dataset : pyvista.DataSet
            Dataset to append to this partitioned dataset.

        """
        index = self.n_partitions
        self.n_partitions += 1
        self[index] = dataset

    def get_data_range(  # numpydoc ignore=RT01
        self: Self, name: str | None, preference: FieldAssociation | str
    ) -> tuple[float, float]:  # pragma: no cover
        """Get the non-NaN min and max of a named array."""
        return DataObject.get_data_range(self, name=name, preference=preference)
