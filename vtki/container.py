"""
Containers to mimic multi block datasets
"""
import logging
from weakref import proxy
import collections

import numpy as np
import vtk
from vtk import vtkMultiBlockDataSet

log = logging.getLogger(__name__)
log.setLevel('CRITICAL')

import vtki
from vtki.utilities import wrap, is_vtki_obj, get_scalar


class MultiBlock(vtkMultiBlockDataSet):
    """
    A container class to hold many data sets which can be iterated over.
    This wraps/extends the ``vtkMultiBlockDataSet`` class in VTK so that we can
    easily plot these data sets and use the container in a Pythonic manner.
    """


    def __init__(self, *args, **kwargs):
        super(MultiBlock, self).__init__()
        deep = kwargs.pop('deep', False)

        if len(args) == 1:
            if isinstance(args[0], vtk.vtkMultiBlockDataSet):
                if deep:
                    self.DeepCopy(args[0])
                else:
                    self.ShallowCopy(args[0])


    @property
    def bounds(self):
        """Finds min/max for bounds across blocks

        Returns:
            tuple(float):
                length 6 tuple of floats containing min/max along each axis
        """
        bounds = [np.inf,-np.inf, np.inf,-np.inf, np.inf,-np.inf]

        def update_bounds(ax, nb, bounds):
            if nb[2*ax] < bounds[2*ax]:
                bounds[2*ax] = nb[2*ax]
            if nb[2*ax+1] > bounds[2*ax+1]:
                bounds[2*ax+1] = nb[2*ax+1]
            return bounds

        # get bounds for each block and update
        for i in range(self.n_blocks):
            try:
                bnds = self[i].GetBounds()
                for a in range(3):
                    bounds = update_bounds(a, bnds, bounds)
            except AttributeError:
                # Data object doesn't have bounds or is None
                pass

        return bounds


    @property
    def n_blocks(self):
        """The total number of blocks set"""
        return self.GetNumberOfBlocks()


    @n_blocks.setter
    def n_blocks(self, n):
        """The total number of blocks set"""
        self.SetNumberOfBlocks(n)


    def get_data_range(self, name):
        """Gets the min/max of a scalar given its name across all blocks"""
        mini, maxi = np.inf, -np.inf
        for i in range(self.n_blocks):
            data = self[i]
            if data is None:
                continue
            # get the scalar if availble
            arr = get_scalar(data, name)
            if arr is None:
                continue
            tmi, tma = np.nanmin(arr), np.nanmax(arr)
            if tmi < mini:
                mini = tmi
            if tma > maxi:
                maxi = tma
        return mini, maxi


    def plot(self, **kwargs):
        """
        Plots all elements in the multiblock dataset
        """
        return vtki.plot(self, **kwargs)


    def get_index_by_name(self, name):
        # find the actual index based on the name
        for i in range(self.n_blocks):
            if self.get_block_name(i) == name:
                return i
        raise RuntimeError('Block name ({}) not found'.format(name))


    def __getitem__(self, index):
        """Get a block by its index or name (if the name is non-unique then
        returns the first occurence)"""
        if isinstance(index, str):
            index = self.get_index_by_name(index)
        data = self.GetBlock(index)
        if data is None:
            return data
        if not is_vtki_obj(data):
            data = wrap(data)
        return data


    def append(self, data):
        """Add a data set to the next block index"""
        index = self.n_blocks # note off by one so ues as index
        self[index] = data


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


    def __setitem__(self, index, data):
        """Sets a block with a VTK data object. To set the name simultaneously,
        pass a string name as the 2nd index.

        Example:
            >>> multi[0] = vtki.PolyData()
            >>> multi[1, 'foo'] = vtki.UnstructuredGrid()
        """
        if isinstance(index, collections.Iterable):
            i, name = index[0], index[1]
        else:
            i, name = index, None
        self.SetBlock(i, data)
        self.set_block_name(i, name) # Note that this calls self.Modified()


    def __delitem__(self, index):
        """Removes a block at the specified index"""
        if isinstance(index, str):
            index = self.get_index_by_name(index)
        self.RemoveBlock(index)


    def __iter__(self):
        self._iter_n = 0
        return self

    def next(self):
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


    ## TODO: I can't get this to work as expected
    # def clean(self):
    #     """This will remove any null blocks"""
    #     nvalid = 0
    #     for i in range(self.n_blocks):
    #         print(i, type(self[i]), self[i], self.get_block_name(i))
    #         if self[i] is None:
    #             print('removing', i)
    #             del self[i]
    #         else:
    #             nvalid += 1
    #     #self.n_blocks = nvalid
    #     print('nvalid', nvalid)
    #     return


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
        fmt += "<tr><th>{}</th><th>Values</th></tr>\n".format(self.GetClassName())
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
            if data is None:
                fmt += row.format(i, None, None)
            else:
                fmt += row.format(i, self.get_block_name(i), data.GetClassName())

        fmt += "</table>\n"
        fmt += "\n"
        fmt += "</td></tr> </table>"
        return fmt
