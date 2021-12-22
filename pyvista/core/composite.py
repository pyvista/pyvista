"""Container to mimic ``vtkMultiBlockDataSet`` objects.

These classes hold many VTK datasets in one object that can be passed
to VTK algorithms and PyVista filtering/plotting routines.
"""
import collections.abc
import logging
import pathlib
from typing import Any, List, Optional, Tuple, Union, cast

import numpy as np

import pyvista
from pyvista import _vtk
from pyvista.utilities import is_pyvista_dataset, wrap

from .dataset import DataObject, DataSet
from .filters import CompositeFilters

log = logging.getLogger(__name__)
log.setLevel('CRITICAL')


class MultiBlock(_vtk.vtkMultiBlockDataSet, CompositeFilters, DataObject):
    """A composite class to hold many data sets which can be iterated over.

    This wraps/extends the ``vtkMultiBlockDataSet`` class in VTK so
    that we can easily plot these data sets and use the composite in a
    Pythonic manner.

    You can think of ``MultiBlock`` like lists or dictionaries as we
    can iterate over this data structure by index and we can also
    access blocks by their string name.

    Examples
    --------
    >>> import pyvista as pv

    Create empty composite dataset

    >>> blocks = pv.MultiBlock()

    Add a dataset to the collection.

    >>> sphere = pv.Sphere()
    >>> blocks.append(sphere)

    Add a named block.

    >>> blocks["cube"] = pv.Cube()

    Instantiate from a list of objects.

    >>> data = [pv.Sphere(center=(2, 0, 0)), pv.Cube(center=(0, 2, 0)),
    ...         pv.Cone()]
    >>> blocks = pv.MultiBlock(data)
    >>> blocks.plot()

    Instantiate from a dictionary.

    >>> data = {"cube": pv.Cube(), "sphere": pv.Sphere(center=(2, 2, 0))}
    >>> blocks = pv.MultiBlock(data)
    >>> blocks.plot()

    Iterate over the collection

    >>> for name in blocks.keys():
    ...     block = blocks[name]

    >>> for block in blocks:
    ...     surf = block.extract_surface()  # Do something with each dataset

    """

    # Bind pyvista.plotting.plot to the object
    plot = pyvista.plot
    _WRITERS = dict.fromkeys(['.vtm', '.vtmb'], _vtk.vtkXMLMultiBlockDataWriter)

    def __init__(self, *args, **kwargs) -> None:
        """Initialize multi block."""
        super().__init__()
        deep = kwargs.pop('deep', False)

        # keep a python reference to the dataset to avoid
        # unintentional garbage collections since python does not
        # add a reference to the dataset when it's added here in
        # MultiBlock.  See https://github.com/pyvista/pyvista/pull/1805
        self._refs: Any = {}

        if len(args) == 1:
            if isinstance(args[0], _vtk.vtkMultiBlockDataSet):
                if deep:
                    self.deep_copy(args[0])
                else:
                    self.shallow_copy(args[0])
            elif isinstance(args[0], (list, tuple)):
                for block in args[0]:
                    self.append(block)
            elif isinstance(args[0], (str, pathlib.Path)):
                self._from_file(args[0], **kwargs)
            elif isinstance(args[0], dict):
                idx = 0
                for key, block in args[0].items():
                    self[idx, key] = block
                    idx += 1
            else:
                raise TypeError(f'Type {type(args[0])} is not supported by pyvista.MultiBlock')

        elif len(args) > 1:
            raise ValueError('Invalid number of arguments:\n``pyvista.MultiBlock``'
                             'supports 0 or 1 arguments.')

        # Upon creation make sure all nested structures are wrapped
        self.wrap_nested()

    def wrap_nested(self):
        """Ensure that all nested data structures are wrapped as PyVista datasets.

        This is performed in place.

        """
        for i in range(self.n_blocks):
            block = self.GetBlock(i)
            if not is_pyvista_dataset(block):
                self.SetBlock(i, pyvista.wrap(block))

    @property
    def bounds(self) -> List[float]:
        """Find min/max for bounds across blocks.

        Returns
        -------
        tuple(float)
            length 6 tuple of floats containing min/max along each axis

        Examples
        --------
        Return the bounds across blocks.

        >>> import pyvista as pv
        >>> data = [pv.Sphere(center=(2, 0, 0)), pv.Cube(center=(0, 2, 0)), pv.Cone()]
        >>> blocks = pv.MultiBlock(data)
        >>> blocks.bounds
        [-0.5, 2.5, -0.5, 2.5, -0.5, 0.5]

        """
        # apply reduction of min and max over each block
        all_bounds = [block.bounds for block in self if block]
        # edge case where block has no bounds
        if not all_bounds:  # pragma: no cover
            minima = np.array([0, 0, 0])
            maxima = np.array([0, 0, 0])
        else:
            minima = np.minimum.reduce(all_bounds)[::2]
            maxima = np.maximum.reduce(all_bounds)[1::2]

        # interleave minima and maxima for bounds
        return np.stack([minima, maxima]).ravel('F').tolist()

    @property
    def center(self) -> Any:
        """Return the center of the bounding box.

        Examples
        --------
        >>> import pyvista as pv
        >>> data = [pv.Sphere(center=(2, 0, 0)), pv.Cube(center=(0, 2, 0)), pv.Cone()]
        >>> blocks = pv.MultiBlock(data)
        >>> blocks.center  # doctest:+SKIP
        array([1., 1., 0.])

        """
        return np.reshape(self.bounds, (3, 2)).mean(axis=1)

    @property
    def length(self) -> float:
        """Return the length of the diagonal of the bounding box.

        Examples
        --------
        >>> import pyvista as pv
        >>> data = [pv.Sphere(center=(2, 0, 0)), pv.Cube(center=(0, 2, 0)), pv.Cone()]
        >>> blocks = pv.MultiBlock(data)
        >>> blocks.length
        4.3584

        """
        return pyvista.Box(self.bounds).length

    @property
    def n_blocks(self) -> int:
        """Return the total number of blocks set.

        Examples
        --------
        >>> import pyvista as pv
        >>> data = [pv.Sphere(center=(2, 0, 0)), pv.Cube(center=(0, 2, 0)), pv.Cone()]
        >>> blocks = pv.MultiBlock(data)
        >>> blocks.n_blocks
        3

        """
        return self.GetNumberOfBlocks()

    @n_blocks.setter
    def n_blocks(self, n):
        """Change the total number of blocks set."""
        self.SetNumberOfBlocks(n)
        self.Modified()

    @property
    def volume(self) -> float:
        """Return the total volume of all meshes in this dataset.

        Returns
        -------
        float
            Total volume of the mesh.

        Examples
        --------
        >>> import pyvista as pv
        >>> data = [pv.Sphere(center=(2, 0, 0)), pv.Cube(center=(0, 2, 0)), pv.Cone()]
        >>> blocks = pv.MultiBlock(data)
        >>> blocks.volume
        1.7348

        """
        return sum(block.volume for block in self if block)

    def get_data_range(self, name: str) -> Tuple[float, float]:  # type: ignore
        """Get the min/max of an array given its name across all blocks.

        Parameters
        ----------
        name : str
            Name of the array.

        Returns
        -------
        tuple
            ``(min, max)`` of the named array.

        """
        mini, maxi = np.inf, -np.inf
        for i in range(self.n_blocks):
            data = self[i]
            if data is None:
                continue
            # get the scalars if available - recursive
            tmi, tma = data.get_data_range(name)
            if not np.isnan(tmi) and tmi < mini:
                mini = tmi
            if not np.isnan(tma) and tma > maxi:
                maxi = tma
        return mini, maxi

    def get_index_by_name(self, name: str) -> int:
        """Find the index number by block name.

        Parameters
        ----------
        name : str
            Name of the block.

        Returns
        -------
        int
            Index of the block.

        Examples
        --------
        >>> import pyvista as pv
        >>> data = {"cube": pv.Cube(), "sphere": pv.Sphere(center=(2, 2, 0))}
        >>> blocks = pv.MultiBlock(data)
        >>> blocks.get_index_by_name('sphere')
        1

        """
        for i in range(self.n_blocks):
            if self.get_block_name(i) == name:
                return i
        raise KeyError(f'Block name ({name}) not found')

    def __getitem__(self, index: Union[int, str]) -> Optional['MultiBlock']:
        """Get a block by its index or name.

        If the name is non-unique then returns the first occurrence.

        """
        if isinstance(index, slice):
            multi = MultiBlock()
            for i in range(self.n_blocks)[index]:
                multi[-1, self.get_block_name(i)] = self[i]
            return multi
        elif isinstance(index, (list, tuple, np.ndarray)):
            multi = MultiBlock()
            for i in index:
                name = i if isinstance(i, str) else self.get_block_name(i)
                multi[-1, name] = self[i]  # type: ignore
            return multi
        elif isinstance(index, str):
            index = self.get_index_by_name(index)
        ############################
        if index < 0:
            index = self.n_blocks + index
        if index < 0 or index >= self.n_blocks:
            raise IndexError(f'index ({index}) out of range for this dataset.')
        data = self.GetBlock(index)
        if data is None:
            return data
        if data is not None and not is_pyvista_dataset(data):
            data = wrap(data)
        return data

    def append(self, dataset: DataSet):
        """Add a data set to the next block index.

        Parameters
        ----------
        dataset : pyvista.DataSet
            Dataset to append to this multi-block.

        Examples
        --------
        >>> import pyvista as pv
        >>> data = {"cube": pv.Cube(), "sphere": pv.Sphere(center=(2, 2, 0))}
        >>> blocks = pv.MultiBlock(data)
        >>> blocks.append(pv.Cone())
        >>> len(blocks)
        3

        """
        index = self.n_blocks  # note off by one so use as index
        # always wrap since we may need to reference the VTK memory address
        if not pyvista.is_pyvista_dataset(dataset):
            dataset = pyvista.wrap(dataset)
        self[index] = dataset

    def get(self, index: Union[int, str]) -> Optional['MultiBlock']:
        """Get a block by its index or name.

        If the name is non-unique then returns the first occurrence.

        Parameters
        ----------
        index : int or str
            Index or name of the dataset within the multiblock.

        Returns
        -------
        pyvista.DataSet
            Dataset from the given index.

        """
        return self[index]

    def set_block_name(self, index: int, name: str):
        """Set a block's string name at the specified index.

        Parameters
        ----------
        index : int
            Index or the dataset within the multiblock.

        name : str
            Name to assign to the block at ``index``.

        Examples
        --------
        >>> import pyvista as pv
        >>> data = {"cube": pv.Cube(), "sphere": pv.Sphere(center=(2, 2, 0))}
        >>> blocks = pv.MultiBlock(data)
        >>> blocks.append(pv.Cone())
        >>> blocks.set_block_name(2, 'cone')
        >>> blocks.keys()
        ['cube', 'sphere', 'cone']

        """
        if name is None:
            return
        self.GetMetaData(index).Set(_vtk.vtkCompositeDataSet.NAME(), name)
        self.Modified()

    def get_block_name(self, index: int) -> Optional[str]:
        """Return the string name of the block at the given index.

        Parameters
        ----------
        index : int
            Index of the block to get the name of.

        Returns
        -------
        str
            Name of the block at the given index.

        Examples
        --------
        >>> import pyvista as pv
        >>> data = {"cube": pv.Cube(), "sphere": pv.Sphere(center=(2, 2, 0))}
        >>> blocks = pv.MultiBlock(data)
        >>> blocks.get_block_name(0)
        'cube'

        """
        meta = self.GetMetaData(index)
        if meta is not None:
            return meta.Get(_vtk.vtkCompositeDataSet.NAME())
        return None

    def keys(self) -> List[Optional[str]]:
        """Get all the block names in the dataset.

        Returns
        -------
        list
            List of block names.

        Examples
        --------
        >>> import pyvista as pv
        >>> data = {"cube": pv.Cube(), "sphere": pv.Sphere(center=(2, 2, 0))}
        >>> blocks = pv.MultiBlock(data)
        >>> blocks.keys()
        ['cube', 'sphere']

        """
        return [self.get_block_name(i) for i in range(self.n_blocks)]

    def _ipython_key_completions_(self) -> List[Optional[str]]:
        return self.keys()

    def __setitem__(self, index: Union[Tuple[int, Optional[str]], int, str], data: DataSet):
        """Set a block with a VTK data object.

        To set the name simultaneously, pass a string name as the 2nd index.

        Example
        -------
        >>> import pyvista
        >>> multi = pyvista.MultiBlock()
        >>> multi[0] = pyvista.PolyData()
        >>> multi[1, 'foo'] = pyvista.UnstructuredGrid()
        >>> multi['bar'] = pyvista.PolyData()
        >>> multi.n_blocks
        3

        """
        i: int = 0
        name: Optional[str] = None
        if isinstance(index, (np.ndarray, collections.abc.Sequence)) and not isinstance(index, str):
            i, name = index[0], index[1]
        elif isinstance(index, str):
            try:
                i = self.get_index_by_name(index)
            except KeyError:
                i = -1
            name = index
        else:
            i, name = cast(int, index), None
        if data is not None and not is_pyvista_dataset(data):
            data = wrap(data)

        if i == -1:
            self.append(data)
            i = self.n_blocks - 1
        else:
            # this is the only spot in the class where we actually add
            # data to the MultiBlock

            # check if we are overwriting a block
            existing_dataset = self.GetBlock(i)
            if existing_dataset is not None:
                self._remove_ref(i)

            self.SetBlock(i, data)
            if data is not None:
                self._refs[data.memory_address] = data

        if name is None:
            name = f'Block-{i:02}'
        self.set_block_name(i, name)  # Note that this calls self.Modified()

    def __delitem__(self, index: Union[int, str]):
        """Remove a block at the specified index."""
        if isinstance(index, str):
            index = self.get_index_by_name(index)
        self._remove_ref(index)
        self.RemoveBlock(index)

    def _remove_ref(self, index: int):
        """Remove python reference to the dataset."""
        dataset = self[index]
        if hasattr(dataset, 'memory_address'):
            self._refs.pop(dataset.memory_address, None)  # type: ignore

    def __iter__(self) -> 'MultiBlock':
        """Return the iterator across all blocks."""
        self._iter_n = 0
        return self

    def __eq__(self, other):
        """Equality comparison."""
        if not isinstance(other, MultiBlock):
            return False

        if self is other:
            return True

        if len(self) != len(other):
            return False

        if not self.keys() == other.keys():
            return False

        if any(self_mesh != other_mesh for self_mesh, other_mesh in zip(self, other)):
            return False

        return True

    def next(self) -> Optional['MultiBlock']:
        """Get the next block from the iterator."""
        if self._iter_n < self.n_blocks:
            result = self[self._iter_n]
            self._iter_n += 1
            return result
        else:
            raise StopIteration

    __next__ = next

    def pop(self, index: Union[int, str]) -> Optional['MultiBlock']:
        """Pop off a block at the specified index.

        Parameters
        ----------
        index : int or str
            Index or name of the dataset within the multiblock.

        Returns
        -------
        pyvista.DataSet
            Dataset from the given index.

        """
        data = self[index]
        del self[index]
        return data

    def clean(self, empty=True):
        """Remove any null blocks in place.

        Parameters
        ----------
        empty : bool
            Remove any meshes that are empty as well (have zero points).

        Examples
        --------
        >>> import pyvista as pv
        >>> data = {"cube": pv.Cube(), "empty": pv.PolyData()}
        >>> blocks = pv.MultiBlock(data)
        >>> blocks.clean(empty=True)
        >>> blocks.keys()
        ['cube']

        """
        null_blocks = []
        for i in range(self.n_blocks):
            if isinstance(self[i], MultiBlock):
                # Recursively move through nested structures
                self[i].clean()
                if self[i].n_blocks < 1:
                    null_blocks.append(i)
            elif self[i] is None:
                null_blocks.append(i)
            elif empty and self[i].n_points < 1:
                null_blocks.append(i)
        # Now remove the null/empty meshes
        null_blocks = np.array(null_blocks, dtype=int)
        for i in range(len(null_blocks)):
            # Cast as int because windows is super annoying
            del self[int(null_blocks[i])]
            null_blocks -= 1

    def _get_attrs(self):
        """Return the representation methods (internal helper)."""
        attrs = []
        attrs.append(("N Blocks", self.n_blocks, "{}"))
        bds = self.bounds
        attrs.append(("X Bounds", (bds[0], bds[1]), "{:.3f}, {:.3f}"))
        attrs.append(("Y Bounds", (bds[2], bds[3]), "{:.3f}, {:.3f}"))
        attrs.append(("Z Bounds", (bds[4], bds[5]), "{:.3f}, {:.3f}"))
        return attrs

    def _repr_html_(self) -> str:
        """Define a pretty representation for Jupyter notebooks."""
        fmt = ""
        fmt += "<table>"
        fmt += "<tr><th>Information</th><th>Blocks</th></tr>"
        fmt += "<tr><td>"
        fmt += "\n"
        fmt += "<table>\n"
        fmt += f"<tr><th>{type(self).__name__}</th><th>Values</th></tr>\n"
        row = "<tr><td>{}</td><td>{}</td></tr>\n"

        # now make a call on the object to get its attributes as a list of len 2 tuples
        for attr in self._get_attrs():
            try:
                fmt += row.format(attr[0], attr[2].format(*attr[1]))
            except:
                fmt += row.format(attr[0], attr[2].format(attr[1]))

        fmt += "</table>\n"
        fmt += "\n"
        fmt += "</td><td>"
        fmt += "\n"
        fmt += "<table>\n"
        row = "<tr><th>{}</th><th>{}</th><th>{}</th></tr>\n"
        fmt += row.format("Index", "Name", "Type")

        for i in range(self.n_blocks):
            data = self[i]
            fmt += row.format(i, self.get_block_name(i), type(data).__name__)

        fmt += "</table>\n"
        fmt += "\n"
        fmt += "</td></tr> </table>"
        return fmt

    def __repr__(self) -> str:
        """Define an adequate representation."""
        # return a string that is Python console friendly
        fmt = f"{type(self).__name__} ({hex(id(self))})\n"
        # now make a call on the object to get its attributes as a list of len 2 tuples
        row = "  {}:\t{}\n"
        for attr in self._get_attrs():
            try:
                fmt += row.format(attr[0], attr[2].format(*attr[1]))
            except:
                fmt += row.format(attr[0], attr[2].format(attr[1]))
        return fmt

    def __str__(self) -> str:
        """Return the str representation of the multi block."""
        return MultiBlock.__repr__(self)

    def __len__(self) -> int:
        """Return the number of blocks."""
        return self.n_blocks

    def copy_meta_from(self, ido):
        """Copy pyvista meta data onto this object from another object."""
        # Note that `pyvista.MultiBlock` datasets currently don't have any meta.
        # This method is here for consistency with the rest of the API and
        # in case we add meta data to this pbject down the road.
        pass

    def copy(self, deep=True):
        """Return a copy of the multiblock.

        Parameters
        ----------
        deep : bool, optional
            When ``True``, make a full copy of the object.

        Returns
        -------
        pyvista.MultiBlock
           Deep or shallow copy of the ``MultiBlock``.

        Examples
        --------
        >>> import pyvista as pv
        >>> data = [pv.Sphere(center=(2, 0, 0)), pv.Cube(center=(0, 2, 0)), pv.Cone()]
        >>> blocks = pv.MultiBlock(data)
        >>> new_blocks = blocks.copy()
        >>> len(new_blocks)
        3

        """
        thistype = type(self)
        newobject = thistype()
        if deep:
            newobject.deep_copy(self)
        else:
            newobject.shallow_copy(self)
        newobject.copy_meta_from(self)
        newobject.wrap_nested()
        return newobject
