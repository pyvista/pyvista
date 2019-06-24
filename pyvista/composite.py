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
from pyvista import plot
from pyvista.utilities import get_scalar, is_pyvista_obj, wrap
from pyvista.filters import DataSetFilters

log = logging.getLogger(__name__)
log.setLevel('CRITICAL')


class CompositeFilters(object):
    """An internal class to manage filtes/algorithms for composite datasets.
    """


    def extract_geometry(composite):
        """Combines the geomertry of all blocks into a single ``PolyData``
        object. Place this filter at the end of a pipeline before a polydata
        consumer such as a polydata mapper to extract geometry from all blocks
        and append them to one polydata object.
        """
        gf = vtk.vtkCompositeDataGeometryFilter()
        gf.SetInputData(composite)
        gf.Update()
        return wrap(gf.GetOutputDataObject(0))


    def combine(composite, merge_points=False):
        """Appends all blocks into a single unstructured grid.

        Parameters
        ----------
        merge_points : bool, optional
            Merge coincidental points.

        """
        alg = vtk.vtkAppendFilter()
        for block in composite:
            alg.AddInputData(block)
        alg.SetMergePoints(merge_points)
        alg.Update()
        return wrap(alg.GetOutputDataObject(0))


    def _dataset_filter_helper(composite, method, **kwargs):
        """This is an internal routine to recursively call an algorithm on every
        block of this composite and return a collected result.
        """
        output_blocks = MultiBlock()
        for i in range(composite.n_blocks):
            name = composite.get_block_name(i)
            block = composite.get(i)
            result = method(block, **kwargs)
            output_blocks[-1, name] = result
        return output_blocks

    def clip(composite, normal='x', origin=None, invert=True):
        """
        Clip a all meshes in this dataset by a plane by specifying the origin
        and normal. If no parameters are given the clip will occur in the center
        of the entire dataset.

        Parameters
        ----------
        normal : tuple(float) or str
            Length 3 tuple for the normal vector direction. Can also be
            specified as a string conventional direction such as ``'x'`` for
            ``(1,0,0)`` or ``'-x'`` for ``(-1,0,0)``, etc.

        origin : tuple(float)
            The center ``(x,y,z)`` coordinate of the plane on which the clip
            occurs

        invert : bool
            Flag on whether to flip/invert the clip

        """
        # find center of data if origin not specified
        if origin is None:
            origin = composite.center
        kwargs = locals()
        _ = kwargs.pop("composite")
        method = DataSetFilters.clip
        return composite._dataset_filter_helper(method, **kwargs)


    def clip_box(composite, bounds=None, invert=True, factor=0.35):
        """Clips a dataset by a bounding box defined by the bounds. If no bounds
        are given, a corner of the dataset bounds will be removed.

        Parameters
        ----------
        bounds : tuple(float)
            Length 6 iterable of floats: (xmin, xmax, ymin, ymax, zmin, zmax)

        invert : bool
            Flag on whether to flip/invert the clip

        factor : float, optional
            If bounds are not given this is the factor along each axis to
            extract the default box.

        """
        kwargs = locals()
        if bounds is None:
            _get_quarter = lambda dmin, dmax: dmax - ((dmax - dmin) * factor)
            xmin, xmax, ymin, ymax, zmin, zmax = composite.bounds
            xmin = _get_quarter(xmin, xmax)
            ymin = _get_quarter(ymin, ymax)
            zmin = _get_quarter(zmin, zmax)
            bounds = [xmin, xmax, ymin, ymax, zmin, zmax]
        kwargs["bounds"] = bounds
        _ = kwargs.pop("composite")
        method = DataSetFilters.clip_box
        return composite._dataset_filter_helper(method, **kwargs)


    def slice(composite, normal='x', origin=None, generate_triangles=False,
              contour=False):
        """Slice a dataset by a plane at the specified origin and normal vector
        orientation. If no origin is specified, the center of the input dataset will
        be used.

        Parameters
        ----------
        normal : tuple(float) or str
            Length 3 tuple for the normal vector direction. Can also be
            specified as a string conventional direction such as ``'x'`` for
            ``(1,0,0)`` or ``'-x'`` for ``(-1,0,0)```, etc.

        origin : tuple(float)
            The center (x,y,z) coordinate of the plane on which the slice occurs

        generate_triangles: bool, optional
            If this is enabled (``False`` by default), the output will be
            triangles otherwise, the output will be the intersection polygons.

        contour : bool, optional
            If True, apply a ``contour`` filter after slicing

        """
        if origin is None:
            origin = composite.center
        kwargs = locals()
        _ = kwargs.pop("composite")
        method = DataSetFilters.slice
        return composite._dataset_filter_helper(method, **kwargs)


    def slice_orthogonal(composite, x=None, y=None, z=None,
                         generate_triangles=False, contour=False):
        """Creates three orthogonal slices through the dataset on the three
        caresian planes. Yields a MutliBlock dataset of the three slices

        Parameters
        ----------
        x : float
            The X location of the YZ slice

        y : float
            The Y location of the XZ slice

        z : float
            The Z location of the XY slice

        generate_triangles: bool, optional
            If this is enabled (``False`` by default), the output will be
            triangles otherwise, the output will be the intersection polygons.

        contour : bool, optional
            If True, apply a ``contour`` filter after slicing

        """
        if x is None:
            x = composite.center[0]
        if y is None:
            y = composite.center[1]
        if z is None:
            z = composite.center[2]
        kwargs = locals()
        _ = kwargs.pop("composite")
        method = DataSetFilters.slice_orthogonal
        return composite._dataset_filter_helper(method, **kwargs)


    def slice_along_axis(composite, n=5, axis='x', tolerance=None,
                         generate_triangles=False, contour=False):
        """Create many slices of the input dataset along a specified axis.

        Parameters
        ----------
        n : int
            The number of slices to create

        axis : str or int
            The axis to generate the slices along. Perpendicular to the slices.
            Can be string name (``'x'``, ``'y'``, or ``'z'``) or axis index
            (``0``, ``1``, or ``2``).

        tolerance : float, optional
            The toleranceerance to the edge of the dataset bounds to create the slices

        generate_triangles: bool, optional
            If this is enabled (``False`` by default), the output will be
            triangles otherwise, the output will be the intersection polygons.

        contour : bool, optional
            If True, apply a ``contour`` filter after slicing

        """
        kwargs = locals()
        kwargs["bounds"] = composite.bounds
        kwargs["center"] = composite.center
        _ = kwargs.pop("composite")
        method = DataSetFilters.slice_along_axis
        return composite._dataset_filter_helper(method, **kwargs)


    def slice_along_line(composite, line, generate_triangles=False,
              contour=False):
        """Slices a dataset using a polyline/spline as the path. This also works
        for lines generated with :func:`pyvista.Line`

        Parameters
        ----------
        line : pyvista.PolyData
            A PolyData object containing one single PolyLine cell.

        generate_triangles: bool, optional
            If this is enabled (``False`` by default), the output will be
            triangles otherwise, the output will be the intersection polygons.

        contour : bool, optional
            If True, apply a ``contour`` filter after slicing
        """
        kwargs = locals()
        _ = kwargs.pop("composite")
        method = DataSetFilters.slice_along_line
        return composite._dataset_filter_helper(method, **kwargs)


    def outline(composite, generate_faces=False):
        """Produces an outline of the full extent for the input dataset.

        Parameters
        ----------
        generate_faces : bool, optional
            Generate solid faces for the box. This is off by default

        """
        box = pyvista.Box(bounds=composite.bounds)
        return box.outline(generate_faces=generate_faces)


    def outline_corners(composite, factor=0.2):
        """Produces an outline of the corners for the input dataset.

        Parameters
        ----------
        factor : float, optional
            controls the relative size of the corners to the length of the
            corresponding bounds

        """
        box = pyvista.Box(bounds=composite.bounds)
        return box.outline_corners(factor=factor)


    def wireframe(composite):
        """Extract all the internal/external edges of all datasets as PolyData.
        This produces a full wireframe representation of the input dataset.
        """
        method = DataSetFilters.wireframe
        return composite._dataset_filter_helper(method)


    def elevation(composite, low_point=None, high_point=None, scalar_range=None,
                  preference='point', set_active=True):
        """Generate scalar values on all datasets.  The scalar values lie within
        a user specified range, and are generated by computing a projection of
        each dataset point onto a line.
        The line can be oriented arbitrarily.
        A typical example is to generate scalars based on elevation or height
        above a plane.

        Parameters
        ----------
        low_point : tuple(float), optional
            The low point of the projection line in 3D space. Default is bottom
            center of the dataset. Otherwise pass a length 3 tuple(float).

        high_point : tuple(float), optional
            The high point of the projection line in 3D space. Default is top
            center of the dataset. Otherwise pass a length 3 tuple(float).

        scalar_range : str or tuple(float), optional
            The scalar range to project to the low and high points on the line
            that will be mapped to the dataset. If None given, the values will
            be computed from the elevation (Z component) range between the
            high and low points. Min and max of a range can be given as a length
            2 tuple(float). If ``str`` name of scalara array present in the
            dataset given, the valid range of that array will be used.

        preference : str, optional
            When a scalar name is specified for ``scalar_range``, this is the
            perfered scalar type to search for in the dataset.
            Must be either 'point' or 'cell'.

        set_active : bool, optional
            A boolean flag on whethter or not to set the new `Elevation` scalar
            as the active scalar array on the output dataset.

        Warning
        -------
        This will create a scalar array named `Elevation` on the point data of
        the input dataset and overasdf write an array named `Elevation` if present.

        """
        # Fix the projection line:
        if low_point is None:
            low_point = list(composite.center)
            low_point[2] = composite.bounds[4]
        if high_point is None:
            high_point = list(composite.center)
            high_point[2] = composite.bounds[5]
        # Fix scalar_range:
        if scalar_range is None:
            scalar_range = (low_point[2], high_point[2])
        elif isinstance(scalar_range, str):
            raise RuntimeError('String array names cannot be use with this filter on MultiBlock datasets.')
        elif isinstance(scalar_range, collections.Iterable):
            if len(scalar_range) != 2:
                raise AssertionError('scalar_range must have a length of two defining the min and max')
        else:
            raise RuntimeError('scalar_range argument ({}) not understood.'.format(type(scalar_range)))
        kwargs = locals()
        _ = kwargs.pop("composite")
        method = DataSetFilters.elevation
        return composite._dataset_filter_helper(method, **kwargs)


    def compute_cell_sizes(composite, length=True, area=True, volume=True):
        """This filter computes sizes for 1D (length), 2D (area) and 3D (volume)
        cells across all datasets.

        Parameters
        ----------
        length : bool
            Specify whether or not to compute the length of 1D cells.

        area : bool
            Specify whether or not to compute the area of 2D cells.

        volume : bool
            Specify whether or not to compute the volume of 3D cells.

        """
        kwargs = locals()
        _ = kwargs.pop("composite")
        method = DataSetFilters.compute_cell_sizes
        return composite._dataset_filter_helper(method, **kwargs)


    def cell_centers(composite, vertex=True):
        """Generate points at the center of the cells across all datasets.
        These points can be used for placing glyphs / vectors.

        Parameters
        ----------
        vertex : bool
            Enable/disable the generation of vertex cells.
        """
        kwargs = locals()
        _ = kwargs.pop("composite")
        method = DataSetFilters.cell_centers
        return composite._dataset_filter_helper(method, **kwargs)


    def cell_data_to_point_data(composite, pass_cell_data=False):
        """Transforms cell data (i.e., data specified per cell) into point data
        (i.e., data specified at cell points).
        The method of transformation is based on averaging the data values of
        all cells using a particular point. Optionally, the input cell data can
        be passed through to the output as well.

        See aslo: :func:`pyvista.DataSetFilters.point_data_to_cell_data`

        Parameters
        ----------
        pass_cell_data : bool
            If enabled, pass the input cell data through to the output
        """
        kwargs = locals()
        _ = kwargs.pop("composite")
        method = DataSetFilters.cell_data_to_point_data
        return composite._dataset_filter_helper(method, **kwargs)


    def point_data_to_cell_data(composite, pass_point_data=False):
        """Transforms point data (i.e., data specified per node) into cell data
        (i.e., data specified within cells).
        Optionally, the input point data can be passed through to the output.

        See aslo: :func:`pyvista.DataSetFilters.cell_data_to_point_data`

        Parameters
        ----------
        pass_point_data : bool
            If enabled, pass the input point data through to the output
        """
        kwargs = locals()
        _ = kwargs.pop("composite")
        method = DataSetFilters.point_data_to_cell_data
        return composite._dataset_filter_helper(method, **kwargs)


    def triangulate(composite):
        """
        Returns an all triangle mesh.  More complex polygons will be broken
        down into triangles.

        Returns
        -------
        mesh : pyvista.UnstructuredGrid
            Mesh containing only triangles.

        """
        method = DataSetFilters.triangulate
        return composite._dataset_filter_helper(method)




class MultiBlock(vtkMultiBlockDataSet, CompositeFilters):
    """
    A composite class to hold many data sets which can be iterated over.
    This wraps/extends the ``vtkMultiBlockDataSet`` class in VTK so that we can
    easily plot these data sets and use the composite in a Pythonic manner.
    """

    # Bind pyvista.plotting.plot to the object
    plot = plot

    def __init__(self, *args, **kwargs):
        super(MultiBlock, self).__init__()
        deep = kwargs.pop('deep', False)
        self.refs = []

        if len(args) == 1:
            if isinstance(args[0], vtk.vtkMultiBlockDataSet):
                if deep:
                    self.DeepCopy(args[0])
                else:
                    self.ShallowCopy(args[0])
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
        self.ShallowCopy(reader.GetOutput())


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
            try:
                bnds = self[i].GetBounds()
                for a in range(3):
                    bounds = update_bounds(a, bnds, bounds)
            except AttributeError:
                # Data object doesn't have bounds or is None
                pass

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
            arr = get_scalar(data, name)
            if arr is None:
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
        if data is not None and not is_pyvista_obj(data):
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
        if data is not None and not is_pyvista_obj(data):
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
            elif self[i] is None:
                null_blocks.append(i)
            elif empty and self[i].n_points < 1:
                null_blocks.append(i)
        # Now remove the null/empty meshes
        null_blocks = np.array(null_blocks, dtype=np.uint32)
        for i in range(len(null_blocks)):
            del self[null_blocks[i]]
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
            newobject.DeepCopy(self)
        else:
            newobject.ShallowCopy(self)
        newobject.copy_meta_from(self)
        return newobject
