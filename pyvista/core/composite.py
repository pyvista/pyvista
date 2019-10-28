"""
Container to mimic ``vtkMultiBlockDataSet`` objects. These classes hold many
VTK datasets in one object that can be passed to VTK algorithms and PyVista
filtering/plotting routines.
"""
import collections
import logging
import os

import numpy as np
import vtk
from vtk import vtkMultiBlockDataSet

import pyvista
from pyvista.utilities import get_array, is_pyvista_dataset, wrap

from .common import DataObject
from .filters import CompositeFilters

log = logging.getLogger(__name__)
log.setLevel('CRITICAL')


class MultiBlock(vtkMultiBlockDataSet, CompositeFilters, DataObject):
    """
    A composite class to hold many data sets which can be iterated over.
    This wraps/extends the ``vtkMultiBlockDataSet`` class in VTK so that we can
    easily plot these data sets and use the composite in a Pythonic manner.

    You can think of ``MultiBlock`` like lists or dictionaries as we can
    iterate over this data structure by index and we can also access blocks
    by their string name.

    Examples
    --------
    >>> import pyvista as pv

    >>> # Create empty composite dataset
    >>> blocks = pv.MultiBlock()
    >>> # Add a dataset to the collection
    >>> sphere = pv.Sphere()
    >>> blocks.append(sphere)
    >>> # Or add a named block
    >>> blocks["cube"] = pv.Cube()

    >>> # instantiate from a list of objects
    >>> data = [pv.Sphere(), pv.Cube(), pv.Cone()]
    >>> blocks = pv.MultiBlock(data)

    >>> # instantiate from a disctionary
    >>> data = {"cube": pv.Cube(), "sphere": pv.Sphere()}
    >>> blocks = pv.MultiBlock(data)

    >>> # now iterate over the collection
    >>> for name in blocks.keys():
    ...     block = blocks[name] # do something!

    >>> for block in blocks:
    ...     surf = block.extract_surface() # Do something with each dataset
    """

    # Bind pyvista.plotting.plot to the object
    plot = pyvista.plot

    def __init__(self, *args, **kwargs):
        super(MultiBlock, self).__init__()
        deep = kwargs.pop('deep', False)
        self.refs = []

        if len(args) == 1:
            if isinstance(args[0], vtk.vtkMultiBlockDataSet):
                if deep:
                    self.deep_copy(args[0])
                else:
                    self.shallow_copy(args[0])
            elif isinstance(args[0], (list, tuple)):
                for block in args[0]:
                    self.append(block)
            elif isinstance(args[0], str):
                self._load_file(args[0])
            elif isinstance(args[0], dict):
                idx = 0
                for key, block in args[0].items():
                    self[idx, key] = block
                    idx += 1

            # keep a reference of the args
            self.refs.append(args)

        # Upon creation make sure all nested structures are wrapped
        self.wrap_nested()


    def wrap_nested(self):
        """A helper to ensure all nested data structures are wrapped as PyVista
        datasets. This is perfrom inplace"""
        for i in range(self.n_blocks):
            block = self.GetBlock(i)
            if not is_pyvista_dataset(block):
                self.SetBlock(i, pyvista.wrap(block))
        return


    def _load_file(self, filename):
        """Load a vtkMultiBlockDataSet from a file (extension ``.vtm`` or
        ``.vtmb``)
        """
        filename = os.path.abspath(os.path.expanduser(filename))
        # test if file exists
        if not os.path.isfile(filename):
            raise Exception('File %s does not exist' % filename)

        # Get extension
        ext = pyvista.get_ext(filename)
        # Extensions: .vtm and .vtmb

        # Select reader
        if ext in ['.vtm', '.vtmb']:
            reader = vtk.vtkXMLMultiBlockDataReader()
        else:
            raise IOError('File extension must be either "vtm" or "vtmb"')

        # Load file
        reader.SetFileName(filename)
        reader.Update()
        self.shallow_copy(reader.GetOutput())


    def save(self, filename, binary=True):
        """
        Writes a ``MultiBlock`` dataset to disk.

        Written file may be an ASCII or binary vtm file.

        Parameters
        ----------
        filename : str
            Filename of mesh to be written.  File type is inferred from
            the extension of the filename unless overridden with
            ftype.  Can be one of the following types (.vtm or .vtmb)

        binary : bool, optional
            Writes the file as binary when True and ASCII when False.

        Notes
        -----
        Binary files write much faster than ASCII and have a smaller
        file size.
        """
        filename = os.path.abspath(os.path.expanduser(filename))
        ext = pyvista.get_ext(filename)
        if ext in ['.vtm', '.vtmb']:
            writer = vtk.vtkXMLMultiBlockDataWriter()
        else:
            raise Exception('File extension must be either "vtm" or "vtmb"')

        writer.SetFileName(filename)
        writer.SetInputDataObject(self)
        if binary:
            writer.SetDataModeToBinary()
        else:
            writer.SetDataModeToAscii()
        writer.Write()
        return

    @property
    def bounds(self):
        """Finds min/max for bounds across blocks

        Returns:
            tuple(float):
                length 6 tuple of floats containing min/max along each axis
        """
        bounds = [np.inf,-np.inf, np.inf,-np.inf, np.inf,-np.inf]

        def update_bounds(ax, nb, bounds):
            """internal helper to update bounds while keeping track"""
            if nb[2*ax] < bounds[2*ax]:
                bounds[2*ax] = nb[2*ax]
            if nb[2*ax+1] > bounds[2*ax+1]:
                bounds[2*ax+1] = nb[2*ax+1]
            return bounds

        # get bounds for each block and update
        for i in range(self.n_blocks):
            if self[i] is None:
                continue
            bnds = self[i].bounds
            for a in range(3):
                bounds = update_bounds(a, bnds, bounds)

        return bounds


    @property
    def center(self):
        """ Center of the bounding box """
        return np.array(self.bounds).reshape(3,2).mean(axis=1)


    @property
    def length(self):
        """the length of the diagonal of the bounding box"""
        return pyvista.Box(self.bounds).length


    @property
    def n_blocks(self):
        """The total number of blocks set"""
        return self.GetNumberOfBlocks()


    @n_blocks.setter
    def n_blocks(self, n):
        """The total number of blocks set"""
        self.SetNumberOfBlocks(n)
        self.Modified()


    @property
    def volume(self):
        """
        Total volume of all meshes in this dataast

        Returns
        -------
        volume : float
            Total volume of the mesh.

        """
        volume = 0.0
        for block in self:
            if block is None:
                continue
            volume += block.volume
        return volume


    def get_data_range(self, name):
        """Gets the min/max of a scalar given its name across all blocks"""
        mini, maxi = np.inf, -np.inf
        for i in range(self.n_blocks):
            data = self[i]
            if data is None:
                continue
            # get the scalar if availble
            arr = get_array(data, name)
            if arr is None or not np.issubdtype(arr.dtype, np.number):
                continue
            tmi, tma = np.nanmin(arr), np.nanmax(arr)
            if tmi < mini:
                mini = tmi
            if tma > maxi:
                maxi = tma
        return mini, maxi


    def get_index_by_name(self, name):
        """Find the index number by block name"""
        for i in range(self.n_blocks):
            if self.get_block_name(i) == name:
                return i
        raise KeyError('Block name ({}) not found'.format(name))


    def __getitem__(self, index):
        """Get a block by its index or name (if the name is non-unique then
        returns the first occurence)"""
        if isinstance(index, str):
            index = self.get_index_by_name(index)
        if index < 0:
            index = self.n_blocks + index
        if index < 0 or index >= self.n_blocks:
            raise IndexError('index ({}) out of range for this dataset.'.format(index))
        data = self.GetBlock(index)
        if data is None:
            return data
        if data is not None and not is_pyvista_dataset(data):
            data = wrap(data)
        if data not in self.refs:
            self.refs.append(data)
        return data


    def append(self, data):
        """Add a data set to the next block index"""
        index = self.n_blocks # note off by one so use as index
        self[index] = data
        self.refs.append(data)


    def get(self, index):
        """Get a block by its index or name (if the name is non-unique then
        returns the first occurence)"""
        return self[index]


    def set_block_name(self, index, name):
        """Set a block's string name at the specified index"""
        if name is None:
            return
        self.GetMetaData(index).Set(vtk.vtkCompositeDataSet.NAME(), name)
        self.Modified()


    def get_block_name(self, index):
        """Returns the string name of the block at the given index"""
        meta = self.GetMetaData(index)
        if meta is not None:
            return meta.Get(vtk.vtkCompositeDataSet.NAME())
        return None


    def keys(self):
        """Get all the block names in the dataset"""
        names = []
        for i in range(self.n_blocks):
            names.append(self.get_block_name(i))
        return names


    def __setitem__(self, index, data):
        """Sets a block with a VTK data object. To set the name simultaneously,
        pass a string name as the 2nd index.

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
        if isinstance(index, collections.Iterable) and not isinstance(index, str):
            i, name = index[0], index[1]
        elif isinstance(index, str):
            try:
                i = self.get_index_by_name(index)
            except KeyError:
                i = -1
            name = index
        else:
            i, name = index, None
        if data is not None and not is_pyvista_dataset(data):
            data = wrap(data)
        if i == -1:
            self.append(data)
            i = self.n_blocks - 1
        else:
            self.SetBlock(i, data)
        if name is None:
            name = 'Block-{0:02}'.format(i)
        self.set_block_name(i, name) # Note that this calls self.Modified()
        if data not in self.refs:
            self.refs.append(data)


    def __delitem__(self, index):
        """Removes a block at the specified index"""
        if isinstance(index, str):
            index = self.get_index_by_name(index)
        self.RemoveBlock(index)


    def __iter__(self):
        """The iterator across all blocks"""
        self._iter_n = 0
        return self

    def next(self):
        """Get the next block from the iterator"""
        if self._iter_n < self.n_blocks:
            result = self[self._iter_n]
            self._iter_n += 1
            return result
        else:
            raise StopIteration

    __next__ = next


    def pop(self, index):
        """Pops off a block at the specified index"""
        data = self[index]
        del self[index]
        return data


    def clean(self, empty=True):
        """This will remove any null blocks in place
        Parameters
        -----------
        empty : bool
            Remove any meshes that are empty as well (have zero points)
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
        return


    def _get_attrs(self):
        """An internal helper for the representation methods"""
        attrs = []
        attrs.append(("N Blocks", self.n_blocks, "{}"))
        bds = self.bounds
        attrs.append(("X Bounds", (bds[0], bds[1]), "{:.3f}, {:.3f}"))
        attrs.append(("Y Bounds", (bds[2], bds[3]), "{:.3f}, {:.3f}"))
        attrs.append(("Z Bounds", (bds[4], bds[5]), "{:.3f}, {:.3f}"))
        return attrs


    def _repr_html_(self):
        """A pretty representation for Jupyter notebooks"""
        fmt = ""
        fmt += "<table>"
        fmt += "<tr><th>Information</th><th>Blocks</th></tr>"
        fmt += "<tr><td>"
        fmt += "\n"
        fmt += "<table>\n"
        fmt += "<tr><th>{}</th><th>Values</th></tr>\n".format(type(self).__name__)
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


    def __repr__(self):
        # return a string that is Python console friendly
        fmt = "{} ({})\n".format(type(self).__name__, hex(id(self)))
        # now make a call on the object to get its attributes as a list of len 2 tuples
        row = "  {}:\t{}\n"
        for attr in self._get_attrs():
            try:
                fmt += row.format(attr[0], attr[2].format(*attr[1]))
            except:
                fmt += row.format(attr[0], attr[2].format(attr[1]))
        return fmt


    def __str__(self):
        return MultiBlock.__repr__(self)


    def __len__(self):
        return self.n_blocks


    def copy_meta_from(self, ido):
        """Copies pyvista meta data onto this object from another object"""
        # Note that `pyvista.MultiBlock` datasets currently don't have any meta.
        # This method is here for consistency witht the rest of the API and
        # incase we add meta data to this pbject down the road.
        pass


    def copy(self, deep=True):
        """
        Returns a copy of the object

        Parameters
        ----------
        deep : bool, optional
            When True makes a full copy of the object.

        Returns
        -------
        newobject : same as input
           Deep or shallow copy of the input.
        """
        thistype = type(self)
        newobject = thistype()
        if deep:
            newobject.deep_copy(self)
        else:
            newobject.shallow_copy(self)
        newobject.copy_meta_from(self)
        return newobject
