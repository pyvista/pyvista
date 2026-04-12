"""Contains the PartitionedDataSet and PartitionedDataSetCollection classes."""

from __future__ import annotations

from collections.abc import MutableSequence
import pathlib
from typing import TYPE_CHECKING
from typing import Any
from typing import overload

import numpy as np

import pyvista as pv
from pyvista._deprecate_positional_args import _deprecate_positional_args
from pyvista.core._vtk_utilities import vtk_version_info

from . import _vtk_core as _vtk
from ._typing_core import BoundsTuple
from .dataobject import DataObject
from .formatting_html import _children_section
from .formatting_html import build_repr_html
from .utilities.helpers import is_pyvista_dataset
from .utilities.helpers import wrap
from .utilities.misc import _BoundsSizeMixin
from .utilities.writer import HDFWriter
from .utilities.writer import XMLPartitionedDataSetCollectionWriter
from .utilities.writer import XMLPartitionedDataSetWriter

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Iterator
    from typing import ClassVar
    from typing import TypeAlias

    from typing_extensions import Self

    from .dataset import DataSet
    from .utilities.arrays import FieldAssociation
    from .utilities.writer import BaseWriter

    _PartitionedLeaf: TypeAlias = 'PartitionedDataSet | None'


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

    plot = pv._plot.plot

    _WRITERS: ClassVar[dict[str, type[BaseWriter]]] = {'.vtpd': XMLPartitionedDataSetWriter}

    if vtk_version_info >= (9, 4):
        _WRITERS['.vtkhdf'] = HDFWriter

    def __init__(self, *args, **kwargs) -> None:
        """Initialize the PartitionedDataSet."""
        super().__init__()
        deep = kwargs.pop('deep', True)
        if len(args) == 1:
            arg = args[0]
            if isinstance(arg, _vtk.vtkPartitionedDataSet):
                if deep:
                    self.deep_copy(arg)
                else:
                    self.ShallowCopy(arg)
            elif isinstance(arg, (list, tuple)):
                for partition in arg:
                    self.append(partition)
            else:
                msg = f'Type {type(arg).__name__} is not supported by pyvista.PartitionedDataSet'
                raise TypeError(msg)
        elif len(args) > 1:
            msg = 'pyvista.PartitionedDataSet supports 0 or 1 positional arguments.'
            raise ValueError(msg)
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
            for i, d in zip(range(self.n_partitions)[index], data, strict=True):
                self.SetPartition(i, d)
        else:
            if index < -self.n_partitions or index >= self.n_partitions:
                msg = f'index ({index}) out of range for this dataset.'
                raise IndexError(msg)
            if index < 0:
                index = self.n_partitions + index
            self.SetPartition(index, data)

    def __delitem__(self, index: int | slice) -> None:
        """Remove a partition at the specified index."""
        if isinstance(index, slice):
            for i in sorted(range(*index.indices(self.n_partitions)), reverse=True):
                del self[i]
            return
        if index < -self.n_partitions or index >= self.n_partitions:
            msg = f'index ({index}) out of range for this dataset.'
            raise IndexError(msg)
        if index < 0:
            index += self.n_partitions
        for i in range(index, self.n_partitions - 1):
            self.SetPartition(i, self.GetPartition(i + 1))
        self.n_partitions -= 1

    def insert(self, index: int, dataset: DataSet | None) -> None:  # numpydoc ignore=PR01
        """Insert data before index."""
        index = range(self.n_partitions + 1)[index]
        self.n_partitions += 1
        for i in reversed(range(index, self.n_partitions - 1)):
            self[i + 1] = self[i]
        self[index] = dataset

    def pop(self, index: int = -1) -> DataSet | None:  # numpydoc ignore=PR01,RT01
        """Pop off a partition at the specified index."""
        data = self[index]
        del self[index]
        return data

    def _get_attrs(self):
        """Return the representation methods (internal helper)."""
        attrs = []
        attrs.append(('N Partitions', self.n_partitions, '{}'))
        return attrs

    def _repr_html_(self) -> str:
        """Define a pretty representation for Jupyter notebooks."""
        sections: list[str] = []
        children = [
            (f'{i}', type(p).__name__ if (p := self[i]) is not None else 'None', '')
            for i in range(self.n_partitions)
        ]
        if children:
            sections.append(_children_section('Partitions', children))

        return build_repr_html(
            obj_type=type(self).__name__,
            mesh_type='MultiBlock',
            header_badges=[f'{self.n_partitions} partitions'],
            sections=sections,
            text_repr=repr(self),
        )

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
            newobject.ShallowCopy(self)
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

    def replace(self, index: int, dataset: DataSet | None) -> None:
        """Replace a partition at the given index.

        Parameters
        ----------
        index : int
            Partition index to overwrite.
        dataset : pyvista.DataSet
            Replacement dataset.

        """
        self[index] = dataset

    def get_data_range(  # numpydoc ignore=RT01
        self: Self, name: str | None, preference: FieldAssociation | str
    ) -> tuple[float, float]:  # pragma: no cover
        """Get the non-NaN min and max of a named array."""
        return DataObject.get_data_range(self, name=name, preference=preference)


class PartitionedDataSetCollection(
    _BoundsSizeMixin,
    DataObject,
    MutableSequence,  # type: ignore[type-arg]
    _vtk.vtkPartitionedDataSetCollection,
):
    """Wrapper for the :vtk:`vtkPartitionedDataSetCollection` class.

    A composite container that holds an ordered collection of
    :class:`pyvista.PartitionedDataSet` instances. Each entry of the collection
    can in turn contain multiple partitions of a single logical mesh. The
    collection also carries an optional :vtk:`vtkDataAssembly` describing a
    hierarchical view of its members.

    The class mirrors :class:`pyvista.MultiBlock` where it makes sense and
    behaves like a :class:`collections.abc.MutableSequence` of
    :class:`pyvista.PartitionedDataSet` objects.

    .. versionadded:: 0.48.0

    Parameters
    ----------
    *args : list, tuple, dict, str, pathlib.Path or vtkPartitionedDataSetCollection
        Initial data. May be another collection (shallow- or deep-copied), a
        sequence of :class:`pyvista.PartitionedDataSet` or
        :class:`pyvista.DataSet` instances, a ``dict`` mapping names to
        datasets, or a path to a file readable by :func:`pyvista.read`.

    deep : bool, default: False
        When initializing from another :vtk:`vtkPartitionedDataSetCollection`,
        perform a deep copy if ``True``, otherwise a shallow copy.

    **kwargs : dict, optional
        Forwarded to :func:`pyvista.read` when constructing from a filename.

    Examples
    --------
    >>> import pyvista as pv
    >>> sphere = pv.PartitionedDataSet([pv.Sphere()])
    >>> cube = pv.PartitionedDataSet([pv.Cube()])
    >>> col = pv.PartitionedDataSetCollection([sphere, cube])
    >>> len(col)
    2

    Construct from a dictionary to assign names.

    >>> col = pv.PartitionedDataSetCollection(
    ...     {'sphere': pv.Sphere(), 'cube': pv.Cube()}
    ... )
    >>> col.keys()
    ['sphere', 'cube']

    """

    plot = pv._plot.plot

    _WRITERS: ClassVar[dict[str, type[BaseWriter]]] = {
        '.vtpc': XMLPartitionedDataSetCollectionWriter,
    }
    if vtk_version_info >= (9, 4):
        _WRITERS['.vtkhdf'] = HDFWriter

    def __init__(self, *args, **kwargs) -> None:
        """Initialize the collection."""
        super().__init__()
        deep = kwargs.pop('deep', False)

        # Keep python references to children to avoid premature GC, mirroring
        # MultiBlock.  See https://github.com/pyvista/pyvista/pull/1805
        self._refs: dict[str, PartitionedDataSet] = {}

        if len(args) == 1:
            arg = args[0]
            if isinstance(arg, _vtk.vtkPartitionedDataSetCollection):
                if deep:
                    self.deep_copy(arg)
                else:
                    self.shallow_copy(arg)
            elif isinstance(arg, (list, tuple)):
                for item in arg:
                    self.append(item)
            elif isinstance(arg, (str, pathlib.Path)):
                self._from_file(arg, **kwargs)
            elif isinstance(arg, dict):
                for key, item in arg.items():
                    self.append(item, key)
            else:
                msg = f'Type {type(arg)} is not supported by pyvista.PartitionedDataSetCollection'
                raise TypeError(msg)
        elif len(args) > 1:
            msg = (
                'Invalid number of arguments:\n'
                '``pyvista.PartitionedDataSetCollection`` supports 0 or 1 arguments.'
            )
            raise ValueError(msg)

        self.wrap_nested()

    @staticmethod
    def _coerce_partitioned(
        dataset: _PartitionedLeaf | _vtk.vtkDataObject,
    ) -> PartitionedDataSet | None:
        """Promote a raw dataset to a single-partition ``PartitionedDataSet``."""
        if dataset is None:
            return None
        if isinstance(dataset, PartitionedDataSet):
            return dataset
        if isinstance(dataset, _vtk.vtkPartitionedDataSet):
            return PartitionedDataSet(dataset)
        wrapped = wrap(dataset)
        pds = PartitionedDataSet()
        pds.append(wrapped)
        return pds

    def wrap_nested(self) -> None:
        """Ensure that all nested data structures are wrapped as PyVista datasets.

        This is performed in place.

        """
        for i in range(self.n_partitioned_datasets):
            partitioned = self.GetPartitionedDataSet(i)
            if isinstance(partitioned, PartitionedDataSet):
                continue
            coerced = PartitionedDataSet(partitioned)
            self.SetPartitionedDataSet(i, coerced)
            self._refs[coerced.memory_address] = coerced

    @overload
    def __getitem__(self, index: int | str) -> PartitionedDataSet | None: ...  # pragma: no cover

    @overload
    def __getitem__(self, index: slice) -> PartitionedDataSetCollection: ...  # pragma: no cover

    def __getitem__(self, index):
        """Get a partitioned dataset by index, name or slice."""
        if isinstance(index, slice):
            new = PartitionedDataSetCollection()
            for i in range(self.n_partitioned_datasets)[index]:
                new.append(self[i], self.get_block_name(i))
            return new
        if isinstance(index, str):
            index = self.get_index_by_name(index)
        n = self.n_partitioned_datasets
        if index < -n or index >= n:
            msg = f'index ({index}) out of range for this dataset.'
            raise IndexError(msg)
        if index < 0:
            index += n
        return wrap(self.GetPartitionedDataSet(index))

    @overload
    def __setitem__(
        self,
        index: int | str,
        data: _PartitionedLeaf | _vtk.vtkDataObject,
    ) -> None: ...  # pragma: no cover

    @overload
    def __setitem__(
        self,
        index: slice,
        data: Iterable[_PartitionedLeaf | _vtk.vtkDataObject],
    ) -> None: ...  # pragma: no cover

    def __setitem__(self, index, data) -> None:
        """Set a partitioned dataset by index, name or slice."""
        name: str | None = None
        if isinstance(index, str):
            try:
                i = self.get_index_by_name(index)
            except KeyError:
                self.append(data, index)
                return
            name = index
            index = i
        elif isinstance(index, slice):
            indices = list(range(self.n_partitioned_datasets)[index])
            data_list = list(data)
            if len(indices) != len(data_list):
                msg = (
                    f'attempt to assign sequence of size {len(data_list)} '
                    f'to slice of size {len(indices)}'
                )
                raise ValueError(msg)
            for idx, d in zip(indices, data_list, strict=True):
                self[idx] = d
            return

        i = range(self.n_partitioned_datasets)[index]
        coerced = self._coerce_partitioned(data)
        self._remove_ref(i)
        # ``SetPartitionedDataSet`` accepts ``None`` at runtime (clearing the
        # slot) but the VTK stubs type it as non-``None``.
        self.SetPartitionedDataSet(i, coerced)  # type: ignore[arg-type]
        if coerced is not None:
            self._refs[coerced.memory_address] = coerced
        if name is not None:
            self.set_block_name(i, name)
        self.Modified()

    def __delitem__(self, index: int | str | slice) -> None:
        """Remove a partitioned dataset at the specified index."""
        if isinstance(index, slice):
            for i in sorted(range(*index.indices(self.n_partitioned_datasets)), reverse=True):
                del self[i]
            return
        if isinstance(index, str):
            index = self.get_index_by_name(index)
        n = self.n_partitioned_datasets
        if index < -n or index >= n:
            msg = f'index ({index}) out of range for this dataset.'
            raise IndexError(msg)
        if index < 0:
            index += n
        names = [self.get_block_name(i) for i in range(n)]
        self._remove_ref(index)
        # Shift subsequent entries down
        for i in range(index, n - 1):
            self.SetPartitionedDataSet(i, self.GetPartitionedDataSet(i + 1))
        self.SetNumberOfPartitionedDataSets(n - 1)
        # Restore names for shifted entries (drop the deleted name)
        del names[index]
        for i, nm in enumerate(names):
            if nm is not None:
                self.set_block_name(i, nm)
        self.Modified()

    def __len__(self) -> int:
        """Return the number of partitioned datasets."""
        return self.n_partitioned_datasets

    def __iter__(self) -> Iterator[PartitionedDataSet | None]:
        """Iterate over the partitioned datasets in the collection."""
        for i in range(self.n_partitioned_datasets):
            yield self[i]

    def __eq__(self, other: object) -> bool:
        """Equality comparison."""
        if self is other:
            return True
        if (
            not isinstance(other, PartitionedDataSetCollection)
            or len(self) != len(other)
            or self.keys() != other.keys()
        ):
            return False
        for a, b in zip(self, other, strict=True):
            if a is None and b is None:
                continue
            if a is None or b is None or len(a) != len(b):
                return False
            if any(pa != pb for pa, pb in zip(a, b, strict=True)):
                return False
        return True

    # Composite datasets are mutable and therefore unhashable.  See
    # https://github.com/pyvista/pyvista/pull/7671.
    __hash__ = None  # type: ignore[assignment]

    def append(
        self,
        dataset: _PartitionedLeaf | _vtk.vtkDataObject,
        name: str | None = None,
    ) -> None:
        """Append a partitioned dataset to the collection.

        Parameters
        ----------
        dataset : pyvista.PartitionedDataSet or pyvista.DataSet
            Dataset to append. Raw datasets are wrapped into a single-partition
            :class:`pyvista.PartitionedDataSet`.

        name : str, optional
            Name to assign to the new entry. A default name
            ``'Block-{i:02}'`` is used when omitted.

        """
        if dataset is self:
            msg = 'Cannot nest a composite dataset in itself.'
            raise ValueError(msg)
        index = self.n_partitioned_datasets
        self.SetNumberOfPartitionedDataSets(index + 1)
        self[index] = dataset
        if name is None:
            name = f'Block-{index:02}'
        self.set_block_name(index, name)

    def extend(self, datasets: Iterable[_PartitionedLeaf | _vtk.vtkDataObject]) -> None:
        """Extend the collection with an iterable.

        If a :class:`PartitionedDataSetCollection` is supplied, names are preserved.

        Parameters
        ----------
        datasets : Iterable
            Datasets to add.

        """
        if isinstance(datasets, PartitionedDataSetCollection):
            for key, data in zip(datasets.keys(), datasets, strict=True):
                self.append(data, key)
        else:
            for v in datasets:
                self.append(v)

    def insert(
        self,
        index: int,
        dataset: _PartitionedLeaf | _vtk.vtkDataObject,
        name: str | None = None,
    ) -> None:
        """Insert a partitioned dataset before ``index``.

        Parameters
        ----------
        index : int
            Index before which to insert the dataset.
        dataset : pyvista.PartitionedDataSet or pyvista.DataSet
            Data to insert.
        name : str, optional
            Name for the new entry. A default name ``'Block-{i:02}'`` is used
            when omitted.

        """
        new_n = self.n_partitioned_datasets + 1
        index = range(new_n)[index]
        names = self.keys()
        self.SetNumberOfPartitionedDataSets(new_n)
        for i in reversed(range(index, new_n - 1)):
            self.SetPartitionedDataSet(i + 1, self.GetPartitionedDataSet(i))
        # Clear the slot so __setitem__ treats it as unoccupied, then assign.
        self.SetPartitionedDataSet(index, _vtk.vtkPartitionedDataSet())
        self[index] = dataset
        names.insert(index, name if name is not None else f'Block-{index:02}')
        for i, nm in enumerate(names):
            if nm is not None:
                self.GetMetaData(i).Set(_vtk.vtkCompositeDataSet.NAME(), nm)
        self.Modified()

    def pop(self, index: int | str = -1) -> PartitionedDataSet | None:
        """Pop a partitioned dataset off the collection.

        Parameters
        ----------
        index : int or str, default: -1
            Index or name of the dataset to remove.

        Returns
        -------
        pyvista.PartitionedDataSet or None
            The dataset that was removed.

        """
        if isinstance(index, int):
            index = range(self.n_partitioned_datasets)[index]
        data = self[index]
        del self[index]
        return data

    def reverse(self) -> None:
        """Reverse the collection in-place."""
        names = self.keys()
        n = len(self)
        for i in range(n // 2):
            self[i], self[n - i - 1] = self[n - i - 1], self[i]
        for i, name in enumerate(reversed(names)):
            if name is not None:
                self.set_block_name(i, name)

    def replace(
        self,
        index: int | str,
        dataset: _PartitionedLeaf | _vtk.vtkDataObject,
    ) -> None:
        """Replace a dataset at index while preserving its name.

        Parameters
        ----------
        index : int or str
            Index or name of the entry to replace.

        dataset : pyvista.PartitionedDataSet or pyvista.DataSet
            New dataset.

        """
        name = index if isinstance(index, str) else self.get_block_name(index)
        self[index] = dataset
        if name is not None:
            self.set_block_name(index, name)

    def _remove_ref(self, index: int) -> None:
        """Drop the python reference bookkeeping for the entry at ``index``."""
        dataset = self[index]
        if dataset is not None:
            self._refs.pop(dataset.memory_address, None)

    def set_block_name(self, index: int | str, name: str | None) -> None:
        """Set a block name at the specified index.

        Parameters
        ----------
        index : int or str
            Index of the entry.
        name : str, optional
            Name to assign. ``None`` is a no-op.

        """
        if name is None:
            return
        if isinstance(index, str):
            index = self.get_index_by_name(index)
        index = range(self.n_partitioned_datasets)[index]
        self.GetMetaData(index).Set(_vtk.vtkCompositeDataSet.NAME(), name)
        self.Modified()

    def get_block_name(self, index: int) -> str | None:
        """Return the string name of the block at ``index``.

        Parameters
        ----------
        index : int
            Index of the entry.

        Returns
        -------
        str or None
            The name of the block, or ``None`` if not set.

        """
        index = range(self.n_partitioned_datasets)[index]
        meta = self.GetMetaData(index)
        if meta is None:
            return None
        return meta.Get(_vtk.vtkCompositeDataSet.NAME())

    def keys(self) -> list[str | None]:  # numpydoc ignore=RT01
        """Return all block names in the collection."""
        return [self.get_block_name(i) for i in range(self.n_partitioned_datasets)]

    def _ipython_key_completions_(self) -> list[str]:
        return [k for k in self.keys() if k is not None]

    def get_index_by_name(self, name: str) -> int:
        """Find the index of a block by name.

        Parameters
        ----------
        name : str
            Name of the block to look up.

        Returns
        -------
        int
            Index of the block.

        """
        for i in range(self.n_partitioned_datasets):
            if self.get_block_name(i) == name:
                return i
        msg = f'Block name ({name}) not found'
        raise KeyError(msg)

    def get(
        self,
        index: int | str,
        default: PartitionedDataSet | None = None,
    ) -> PartitionedDataSet | None:
        """Get a block by index or name, returning ``default`` if missing.

        Parameters
        ----------
        index : int or str
            Index or name of the block to look up.

        default : pyvista.PartitionedDataSet, optional
            Value to return when the lookup fails.

        Returns
        -------
        pyvista.PartitionedDataSet or None
            The block, or ``default`` if not found.

        """
        try:
            return self[index]
        except (KeyError, IndexError):
            return default

    def get_block(self, index: int | str) -> PartitionedDataSet | None:
        """Get a block by index or name; raise if not found.

        Parameters
        ----------
        index : int or str
            Index or name of the block to look up.

        Returns
        -------
        pyvista.PartitionedDataSet or None
            The block at the given index.

        """
        return self[index]

    @property
    def n_partitioned_datasets(self) -> int:  # numpydoc ignore=RT01
        """Return the number of partitioned datasets in the collection."""
        return self.GetNumberOfPartitionedDataSets()

    @n_partitioned_datasets.setter
    def n_partitioned_datasets(self, n: int) -> None:
        self.SetNumberOfPartitionedDataSets(n)
        self.Modified()

    @property
    def n_blocks(self) -> int:  # numpydoc ignore=RT01
        """Alias for :attr:`n_partitioned_datasets` for MultiBlock parity."""
        return self.n_partitioned_datasets

    @n_blocks.setter
    def n_blocks(self, n: int) -> None:
        self.n_partitioned_datasets = n

    @property
    def is_empty(self) -> bool:  # numpydoc ignore=RT01
        """Return ``True`` if the collection has no entries."""
        return self.n_partitioned_datasets == 0

    @property
    def is_nested(self) -> bool:  # numpydoc ignore=RT01
        """Return ``True`` if any entry holds more than one partition."""
        return any(block is not None and block.n_partitions > 1 for block in self)

    @property
    def bounds(self) -> BoundsTuple:
        """Compute aggregate bounds across all leaf datasets.

        Returns
        -------
        BoundsTuple
            Aggregate min/max bounds.

        """
        all_bounds: list[list[float]] = []
        for block in self:
            if block is None:
                continue
            for i in range(block.n_partitions):
                leaf = block[i]
                if leaf is None:
                    continue
                all_bounds.append(list(leaf.bounds))
        if not all_bounds:
            return BoundsTuple(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        minima = np.minimum.reduce(all_bounds)[::2].tolist()
        maxima = np.maximum.reduce(all_bounds)[1::2].tolist()
        return BoundsTuple(minima[0], maxima[0], minima[1], maxima[1], minima[2], maxima[2])

    @property
    def center(self) -> tuple[float, float, float]:  # numpydoc ignore=RT01
        """Return the center of the aggregate bounding box."""
        return tuple(np.reshape(self.bounds, (3, 2)).mean(axis=1).tolist())

    @property
    def length(self) -> float:  # numpydoc ignore=RT01
        """Return the length of the diagonal of the aggregate bounding box."""
        return float(np.linalg.norm(np.subtract(self.bounds[1::2], self.bounds[::2])))

    @property
    def block_types(self) -> set[type]:  # numpydoc ignore=RT01
        """Return the set of types of all leaf datasets across the collection."""
        types: set[type] = set()
        for block in self:
            if block is None:
                continue
            for i in range(block.n_partitions):
                leaf = block[i]
                if leaf is not None:
                    types.add(type(leaf))
        return types

    @property
    def is_all_polydata(self) -> bool:  # numpydoc ignore=RT01
        """Return ``True`` if every non-``None`` leaf is :class:`pyvista.PolyData`."""
        return all(
            isinstance(leaf, pv.PolyData) for leaf in self.recursive_iterator(contents='blocks')
        )

    def outline(
        self,
        *,
        generate_faces: bool = False,
        progress_bar: bool = False,
    ) -> pv.PolyData:
        """Produce an outline of the aggregate bounding box of all leaves.

        Parameters
        ----------
        generate_faces : bool, default: False
            Generate solid faces for the box.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        Returns
        -------
        pyvista.PolyData
            Mesh containing the outline.

        """
        box = pv.Box(bounds=self.bounds)
        return box.outline(generate_faces=generate_faces, progress_bar=progress_bar)

    def outline_corners(
        self,
        factor: float = 0.2,
        *,
        progress_bar: bool = False,
    ) -> pv.PolyData:
        """Produce an outline of the corners of the aggregate bounding box.

        Parameters
        ----------
        factor : float, default: 0.2
            Controls the relative size of the corners to the length of the
            corresponding bounds.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        Returns
        -------
        pyvista.PolyData
            Mesh containing outlined corners.

        """
        box = pv.Box(bounds=self.bounds)
        return box.outline_corners(factor=factor, progress_bar=progress_bar)

    @_deprecate_positional_args
    def combine(
        self,
        merge_points: bool = False,  # noqa: FBT001, FBT002
        tolerance: float = 0.0,
    ) -> pv.UnstructuredGrid:
        """Combine all leaf datasets into a single :class:`pyvista.UnstructuredGrid`.

        Parameters
        ----------
        merge_points : bool, default: False
            Merge coincident points.

        tolerance : float, default: 0.0
            Absolute tolerance used to identify coincident points when
            ``merge_points=True``.

        Returns
        -------
        pyvista.UnstructuredGrid
            Combined unstructured grid containing every leaf dataset.

        """
        alg = _vtk.vtkAppendFilter()
        alg.SetMergePoints(merge_points)
        alg.SetTolerance(tolerance)
        for leaf in self.recursive_iterator(contents='blocks'):
            alg.AddInputData(leaf)
        alg.Update()
        return wrap(alg.GetOutputDataObject(0))  # type: ignore[return-value]

    @property
    def assembly(self) -> _vtk.vtkDataAssembly:  # numpydoc ignore=RT01
        """Return the underlying :vtk:`vtkDataAssembly`.

        If no assembly is currently attached, a new empty assembly is created
        and attached automatically so callers can build it up incrementally.

        """
        # ``GetDataAssembly`` is stubbed as non-``None`` but returns ``None``
        # at runtime when no assembly is attached.
        existing: _vtk.vtkDataAssembly | None = self.GetDataAssembly()
        if existing is None:
            existing = _vtk.vtkDataAssembly()
            self.SetDataAssembly(existing)
        return existing

    @assembly.setter
    def assembly(self, value: _vtk.vtkDataAssembly | None) -> None:
        # ``SetDataAssembly`` accepts ``None`` at runtime to clear the
        # assembly, but the VTK stubs reject it.
        self.SetDataAssembly(value)  # type: ignore[arg-type]
        self.Modified()

    def add_assembly_node(self, parent_path: str, name: str) -> int:
        """Add a child node to the data assembly.

        Parameters
        ----------
        parent_path : str
            Selector identifying the parent node. Either ``'/'`` for the
            root, an absolute XPath like ``'/assembly/group'``, or a relative
            XPath like ``'//group'`` matching the first node by name.
        name : str
            Name of the new node.

        Returns
        -------
        int
            Identifier of the newly created node.

        """
        assembly = self.assembly
        if parent_path == '/':
            parent_id = assembly.GetRootNode()
        else:
            matches = list(assembly.SelectNodes([parent_path]))
            if not matches:
                msg = f'Assembly node not found at path: {parent_path!r}'
                raise KeyError(msg)
            parent_id = int(matches[0])
        return assembly.AddNode(name, parent_id)

    def assign_dataset_to_node(self, node_id: int, dataset_index: int) -> None:
        """Attach a dataset index to an assembly node.

        Parameters
        ----------
        node_id : int
            The assembly node identifier.
        dataset_index : int
            Index of the partitioned dataset within the collection.

        """
        n = self.n_partitioned_datasets
        if dataset_index < 0 or dataset_index >= n:
            msg = f'dataset_index ({dataset_index}) out of range for this collection.'
            raise IndexError(msg)
        self.assembly.AddDataSetIndex(node_id, dataset_index)

    def select_datasets(self, selector: str) -> list[int]:
        """Return dataset indices selected by an assembly XPath-like selector.

        Parameters
        ----------
        selector : str
            XPath-like selector understood by :vtk:`vtkDataAssembly`.

        Returns
        -------
        list[int]
            Indices of partitioned datasets matching the selector.

        """
        assembly: _vtk.vtkDataAssembly | None = self.GetDataAssembly()
        if assembly is None:
            return []
        node_ids = assembly.SelectNodes([selector])
        indices: list[int] = []
        for node_id in node_ids:
            indices.extend(int(i) for i in assembly.GetDataSetIndices(node_id))
        # Deduplicate while preserving order
        seen: set[int] = set()
        result: list[int] = []
        for i in indices:
            if i not in seen:
                seen.add(i)
                result.append(i)
        return result

    def assembly_to_dict(self) -> dict[str, Any]:  # numpydoc ignore=RT01
        """Return a recursive dict representation of the data assembly."""
        assembly: _vtk.vtkDataAssembly | None = self.GetDataAssembly()
        if assembly is None:
            return {}

        def _walk(node_id: int) -> dict[str, Any]:
            # Only direct dataset indices, not recursive into children
            datasets = [int(i) for i in assembly.GetDataSetIndices(node_id, False)]
            return {
                'name': assembly.GetNodeName(node_id),
                'id': int(node_id),
                'datasets': datasets,
                'children': [_walk(c) for c in assembly.GetChildNodes(node_id)],
            }

        return _walk(assembly.GetRootNode())

    def recursive_iterator(
        self,
        contents: str = 'blocks',
        *,
        skip_none: bool = True,
    ) -> Iterator[Any]:
        """Iterate over the leaf datasets of the collection.

        Parameters
        ----------
        contents : {'ids', 'names', 'blocks', 'items', 'all'}, default: 'blocks'
            What to yield for each leaf:

            * ``'ids'`` — ``(collection_index, partition_index)`` tuples.
            * ``'names'`` — block name of the parent partitioned dataset.
            * ``'blocks'`` — leaf datasets.
            * ``'items'`` — ``(name, leaf)`` tuples.
            * ``'all'`` — ``((collection_index, partition_index), name, leaf)``.

        skip_none : bool, default: True
            Skip leaves that are ``None``.

        Yields
        ------
        Any
            Items as described by ``contents``.

        """
        valid = {'ids', 'names', 'blocks', 'items', 'all'}
        if contents not in valid:
            msg = f'contents must be one of {sorted(valid)}, got {contents!r}'
            raise ValueError(msg)
        for i, block in enumerate(self):
            if block is None:
                continue
            name = self.get_block_name(i)
            for j in range(block.n_partitions):
                leaf = block[j]
                if leaf is None and skip_none:
                    continue
                if contents == 'ids':
                    yield (i, j)
                elif contents == 'names':
                    yield name
                elif contents == 'blocks':
                    yield leaf
                elif contents == 'items':
                    yield (name, leaf)
                else:  # 'all'
                    yield ((i, j), name, leaf)

    def cast_to_multiblock(self) -> pv.MultiBlock:
        """Convert the collection to a :class:`pyvista.MultiBlock`.

        Equivalent to :meth:`flatten` — every leaf :class:`pyvista.DataSet`
        becomes a block in the returned ``MultiBlock``. The default
        :vtk:`vtkConvertToMultiBlockDataSet` path is overridden because it
        produces :vtk:`vtkMultiPieceDataSet` entries that PyVista cannot wrap.

        Returns
        -------
        pyvista.MultiBlock
            ``MultiBlock`` containing every leaf dataset in the collection.

        """
        return self.flatten()

    def flatten(self) -> pv.MultiBlock:
        """Flatten the collection into a :class:`pyvista.MultiBlock` of leaf datasets.

        Returns
        -------
        pyvista.MultiBlock
            A ``MultiBlock`` containing every leaf dataset in the collection,
            labelled as ``'{block_name}/partition-{j:02}'``.

        """
        multi = pv.MultiBlock()
        for (i, j), name, leaf in self.recursive_iterator(contents='all', skip_none=False):
            label = f'{name or f"Block-{i:02}"}/partition-{j:02}'
            multi.append(leaf, label)
        return multi

    @_deprecate_positional_args(allowed=['name'])
    def set_active_scalars(
        self,
        name: str | None,
        preference: str = 'cell',
        *,
        allow_missing: bool = False,
    ) -> None:
        """Set active scalars across all leaf datasets.

        Parameters
        ----------
        name : str or None
            Name of the array to activate. Pass ``None`` to deactivate.

        preference : str, default: 'cell'
            Array association preference. Either ``'cell'`` or ``'point'``.

        allow_missing : bool, default: False
            If ``True``, silently skip leaves that do not contain an array
            with the given ``name``. If ``False``, a :class:`KeyError` is
            raised for the first missing leaf.

        """
        for leaf in self.recursive_iterator(contents='blocks'):
            try:
                leaf.set_active_scalars(name, preference=preference)
            except KeyError:
                if not allow_missing:
                    raise

    def clear_all_data(self) -> None:
        """Clear all point, cell and field data from every leaf dataset."""
        for leaf in self.recursive_iterator(contents='blocks'):
            leaf.clear_data()

    def clear_all_point_data(self) -> None:
        """Clear all point data from every leaf dataset."""
        for leaf in self.recursive_iterator(contents='blocks'):
            leaf.clear_point_data()

    def clear_all_cell_data(self) -> None:
        """Clear all cell data from every leaf dataset."""
        for leaf in self.recursive_iterator(contents='blocks'):
            leaf.clear_cell_data()

    @_deprecate_positional_args(allowed=['name'])
    def get_data_range(  # type: ignore[override]
        self,
        name: str | None,
        preference: str = 'cell',
        *,
        allow_missing: bool = False,
    ) -> tuple[float, float]:
        """Get the combined min/max of a named array across all leaf datasets.

        Parameters
        ----------
        name : str or None
            Name of the array to query.

        preference : str, default: 'cell'
            Array association preference. Either ``'cell'`` or ``'point'``.

        allow_missing : bool, default: False
            If ``True``, leaves without the requested array are skipped. If
            ``False``, a :class:`KeyError` is raised for the first missing
            leaf.

        Returns
        -------
        tuple[float, float]
            ``(min, max)`` values across all leaves. Returns
            ``(nan, nan)`` when no leaf contributes a finite value.

        """
        mini, maxi = np.inf, -np.inf
        for leaf in self.recursive_iterator(contents='blocks'):
            try:
                tmi, tma = leaf.get_data_range(name, preference=preference)
            except KeyError:
                if allow_missing:
                    continue
                raise
            if not np.isnan(tmi) and tmi < mini:
                mini = tmi
            if not np.isnan(tma) and tma > maxi:
                maxi = tma
        if not np.isfinite(mini) or not np.isfinite(maxi):
            return float('nan'), float('nan')
        return float(mini), float(maxi)

    def copy_meta_from(self, ido, deep) -> None:  # numpydoc ignore=PR01
        """Copy pyvista-side metadata onto this object from another object."""
        # No pyvista-side metadata is currently tracked.

    @_deprecate_positional_args
    def copy(self, deep: bool = True) -> Self:  # noqa: FBT001, FBT002
        """Return a copy of the collection.

        Parameters
        ----------
        deep : bool, default: True
            When ``True``, perform a deep copy.

        Returns
        -------
        pyvista.PartitionedDataSetCollection
            A new collection.

        """
        thistype = type(self)
        new = thistype()
        if deep:
            new.deep_copy(self)
        else:
            new.shallow_copy(self)
        new.copy_meta_from(self, deep=deep)
        return new

    def shallow_copy(  # type: ignore[override]
        self, to_copy: _vtk.vtkPartitionedDataSetCollection
    ) -> None:
        """Shallow copy from another collection."""
        if vtk_version_info >= (9, 3):
            self.CompositeShallowCopy(to_copy)
        else:
            self.ShallowCopy(to_copy)
        self.wrap_nested()

    def deep_copy(  # type: ignore[override]
        self, to_copy: _vtk.vtkPartitionedDataSetCollection
    ) -> None:
        """Deep copy from another collection."""
        super().deep_copy(to_copy)
        self.wrap_nested()

    def _get_attrs(self) -> list[tuple[str, Any, str]]:
        attrs: list[tuple[str, Any, str]] = []
        attrs.append(('N PartitionedDataSets:', self.n_partitioned_datasets, '{}'))
        bds = self.bounds
        attrs.append(('X Bounds:', (bds.x_min, bds.x_max), '{:.3e}, {:.3e}'))
        attrs.append(('Y Bounds:', (bds.y_min, bds.y_max), '{:.3e}, {:.3e}'))
        attrs.append(('Z Bounds:', (bds.z_min, bds.z_max), '{:.3e}, {:.3e}'))
        return attrs

    def __repr__(self) -> str:
        """Return a textual representation."""
        fmt = f'{type(self).__name__} ({hex(id(self))})\n'
        attrs = self._get_attrs()
        max_len = max(len(a[0]) for a in attrs) + 3
        row = f'  {{:{max_len}s}}' + '{}\n'
        for label, value, value_fmt in attrs:
            try:
                fmt += row.format(label, value_fmt.format(*value))
            except TypeError:
                fmt += row.format(label, value_fmt.format(value))
        return fmt.strip()

    def __str__(self) -> str:
        """Return a textual representation."""
        return PartitionedDataSetCollection.__repr__(self)

    def _repr_html_(self) -> str:
        """Return an HTML representation for Jupyter notebooks."""
        sections: list[str] = []
        children: list[tuple[str, str, str]] = []
        for i in range(self.n_partitioned_datasets):
            block = self[i]
            name = self.get_block_name(i) or f'Block {i}'
            if block is None:
                children.append((name, 'None', ''))
                continue
            ctype = type(block).__name__
            detail = f'{block.n_partitions} partitions'
            children.append((name, ctype, detail))
        if children:
            sections.append(_children_section('Partitioned Datasets', children))
        return build_repr_html(
            obj_type=type(self).__name__,
            mesh_type='MultiBlock',
            header_badges=[f'{self.n_partitioned_datasets} entries'],
            sections=sections,
            text_repr=repr(self),
        )
