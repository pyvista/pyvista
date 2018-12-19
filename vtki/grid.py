"""
Sub-classes for vtk.vtkUnstructuredGrid and vtk.vtkStructuredGrid
"""
import os
import logging

import vtk
from vtk import vtkUnstructuredGrid, vtkStructuredGrid, vtkRectilinearGrid, vtkImageData
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtkIdTypeArray
from vtk.util.numpy_support import numpy_to_vtk
from vtk import VTK_TRIANGLE
from vtk import VTK_QUAD
from vtk import VTK_QUADRATIC_TRIANGLE
from vtk import VTK_QUADRATIC_QUAD
from vtk import VTK_HEXAHEDRON
from vtk import VTK_PYRAMID
from vtk import VTK_TETRA
from vtk import VTK_WEDGE
from vtk import VTK_QUADRATIC_TETRA
from vtk import VTK_QUADRATIC_PYRAMID
from vtk import VTK_QUADRATIC_WEDGE
from vtk import VTK_QUADRATIC_HEXAHEDRON

import numpy as np

import vtki

log = logging.getLogger(__name__)
log.setLevel('CRITICAL')



class Grid(vtki.Common):
    """ Class in common with structured and unstructured grids """

    def __init__(self, *args, **kwargs):
        pass

    def plot_curvature(self, curv_type='mean', **kwargs):
        """
        Plots the curvature of the external surface of the grid

        Parameters
        ----------
        curv_type : str, optional
            One of the following strings indicating curvature types

            - mean
            - gaussian
            - maximum
            - minimum

        **kwargs : optional
            Optional keyword arguments.  See help(vtki.plot)

        Returns
        -------
        cpos : list
            Camera position, focal point, and view up.  Used for storing and
            setting camera view.

        """
        trisurf = self.extract_surface().tri_filter()
        return trisurf.plot_curvature(curv_type, **kwargs)

    @property
    def volume(self):
        """
        Computes volume by extracting the external surface and
        computing interior volume
        """
        surf = self.extract_surface().tri_filter()
        return surf.volume

    def extract_surface(self, pass_pointid=True, pass_cellid=True):
        """
        Extract surface mesh of the grid

        Parameters
        ----------
        pass_pointid : bool, optional
            Adds a point scalar "vtkOriginalPointIds" that idenfities which
            original points these surface points correspond to

        pass_cellid : bool, optional
            Adds a cell scalar "vtkOriginalPointIds" that idenfities which
            original cells these surface cells correspond to

        Returns
        -------
        extsurf : vtki.PolyData
            Surface mesh of the grid
        """
        surf_filter = vtk.vtkDataSetSurfaceFilter()
        surf_filter.SetInputData(self)
        if pass_pointid:
            surf_filter.PassThroughCellIdsOn()
        if pass_cellid:
            surf_filter.PassThroughPointIdsOn()
        surf_filter.Update()
        return vtki.PolyData(surf_filter.GetOutput())

    def surface_indices(self):
        """
        The surface indices of a grid.

        Returns
        -------
        surf_ind : np.ndarray
            Indices of the surface points.

        """
        surf = self.extract_surface(pass_cellid=True)
        return surf.point_arrays['vtkOriginalPointIds']

    def extract_edges(self, feature_angle=30, boundary_edges=True,
                      non_manifold_edges=True, feature_edges=True,
                      manifold_edges=True):
        """
        Extracts edges from the surface of the grid.  From vtk documentation:

        These edges are either
            1) boundary (used by one polygon) or a line cell;
            2) non-manifold (used by three or more polygons)
            3) feature edges (edges used by two triangles and whose
               dihedral angle > feature_angle)
            4) manifold edges (edges used by exactly two polygons).


        Parameters
        ----------
        feature_angle : float, optional
            Defaults to 30 degrees.

        boundary_edges : bool, optional
            Defaults to True

        non_manifold_edges : bool, optional
            Defaults to True

        feature_edges : bool, optional
            Defaults to True

        manifold_edges : bool, optional
            Defaults to True

        Returns
        -------
        edges : vtki.vtkPolyData
            Extracted edges

        """
        surf = self.extract_surface()
        return surf.extract_edges(feature_angle, boundary_edges,
                                  non_manifold_edges, feature_edges,
                                  manifold_edges)


class UnstructuredGrid(vtkUnstructuredGrid, Grid):
    """
    Extends the functionality of a vtk.vtkUnstructuredGrid object.

    Can be initialized by the following:

    - Creating an empty grid
    - From a vtk.vtkPolyData object
    - From cell, offset, and node arrays
    - From a file

    Examples
    --------
    >>> grid = UnstructuredGrid()
    >>> grid = UnstructuredGrid(vtkgrid)  # Initialize from a vtkUnstructuredGrid
    >>> grid = UnstructuredGrid(offset, cells, cell_type, nodes, deep=True)  # from arrays
    >>> grid = UnstructuredGrid(filename)  # from a file

    """

    def __init__(self, *args, **kwargs):
        super(UnstructuredGrid, self).__init__()
        deep = kwargs.pop('deep', False)

        if len(args) == 1:
            if isinstance(args[0], vtk.vtkUnstructuredGrid):
                if deep:
                    self.DeepCopy(args[0])
                else:
                    self.ShallowCopy(args[0])

            elif isinstance(args[0], str):
                self._load_file(args[0])

            elif isinstance(args[0], vtk.vtkStructuredGrid):
                vtkappend = vtk.vtkAppendFilter()
                vtkappend.AddInputData(args[0])
                vtkappend.Update()
                self.ShallowCopy(vtkappend.GetOutput())

            else:
                itype = type(args[0])
                raise Exception('Cannot work with input type %s' % itype)

        elif len(args) == 4:
            arg0_is_arr = isinstance(args[0], np.ndarray)
            arg1_is_arr = isinstance(args[1], np.ndarray)
            arg2_is_arr = isinstance(args[2], np.ndarray)
            arg3_is_arr = isinstance(args[3], np.ndarray)

            if all([arg0_is_arr, arg1_is_arr, arg2_is_arr, arg3_is_arr]):
                self._from_arrays(args[0], args[1], args[2], args[3], deep)
            else:
                raise Exception('All input types must be np.ndarray')

    def _from_arrays(self, offset, cells, cell_type, points, deep=True):
        """
        Create VTK unstructured grid from numpy arrays

        Parameters
        ----------
        offset : np.ndarray dtype=np.int64
            Array indicating the start location of each cell in the cells
            array.

        cells : np.ndarray dtype=np.int64
            Array of cells.  Each cell contains the number of points in the
            cell and the node numbers of the cell.

        cell_type : np.uint8
            Cell types of each cell.  Each cell type numbers can be found from
            vtk documentation.  See example below.

        points : np.ndarray
            Numpy array containing point locations.

        Examples
        --------
        >>> offset = np.array([0, 9])
        >>> cells = np.array([8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 9, 10, 11, 12, 13, 14, 15])
        >>> cell_type = np.array([vtk.VTK_HEXAHEDRON, vtk.VTK_HEXAHEDRON], np.int8)

        >>> cell1 = np.array([[0, 0, 0],
                              [1, 0, 0],
                              [1, 1, 0],
                              [0, 1, 0],
                              [0, 0, 1],
                              [1, 0, 1],
                              [1, 1, 1],
                              [0, 1, 1]])

        >>> cell2 = np.array([[0, 0, 2],
                              [1, 0, 2],
                              [1, 1, 2],
                              [0, 1, 2],
                              [0, 0, 3],
                              [1, 0, 3],
                              [1, 1, 3],
                              [0, 1, 3]])

        >>> points = np.vstack((cell1, cell2))

        >>> grid = vtki.UnstructuredGrid(offset, cells, cell_type, points)

        """

        if offset.dtype != vtki.ID_TYPE:
            offset = offset.astype(vtki.ID_TYPE)

        if cells.dtype != vtki.ID_TYPE:
            cells = cells.astype(vtki.ID_TYPE)

        if not cells.flags['C_CONTIGUOUS']:
            cells = np.ascontiguousarray(cells)

        # if cells.ndim != 1:
            # cells = cells.ravel()

        if cell_type.dtype != np.uint8:
            cell_type = cell_type.astype(np.uint8)

        # Get number of cells
        ncells = cell_type.size

        # Convert to vtk arrays
        cell_type = numpy_to_vtk(cell_type, deep=deep)
        offset = numpy_to_vtkIdTypeArray(offset, deep=deep)

        vtkcells = vtk.vtkCellArray()
        vtkcells.SetCells(ncells, numpy_to_vtkIdTypeArray(cells.ravel(), deep=deep))

        # Convert points to vtkPoints object
        points = vtki.vtk_points(points, deep=deep)

        # Create unstructured grid
        self.SetPoints(points)
        self.SetCells(cell_type, offset, vtkcells)

    def _load_file(self, filename):
        """
        Load an unstructured grid from a file.

        The file extension will select the type of reader to use.  A .vtk
        extension will use the legacy reader, while .vtu will select the VTK
        XML reader.

        Parameters
        ----------
        filename : str
            Filename of grid to be loaded.
        """
        # check file exists
        if not os.path.isfile(filename):
            raise Exception('%s does not exist' % filename)

        # Check file extention
        if '.vtu' in filename:
            reader = vtk.vtkXMLUnstructuredGridReader()
        elif '.vtk' in filename:
            reader = vtk.vtkUnstructuredGridReader()
        else:
            raise Exception('Extension should be either ".vtu" or ".vtk"')

        # load file to self
        reader.SetFileName(filename)
        reader.Update()
        grid = reader.GetOutput()
        self.ShallowCopy(grid)

    def save(self, filename, binary=True):
        """
        Writes an unstructured grid to disk.

        Parameters
        ----------
        filename : str
            Filename of grid to be written.  The file extension will select the
            type of writer to use. ".vtk" will use the legacy writer, while
            ".vtu" will select the VTK XML writer.

        binary : bool, optional
            Writes as a binary file by default.  Set to False to write ASCII.


        Notes
        -----
        Binary files write much faster than ASCII, but binary files written on
        one system may not be readable on other systems.  Binary can be used
        only ".vtk" files
        """
        # Use legacy writer if vtk is in filename
        if '.vtk' in filename:
            writer = vtk.vtkUnstructuredGridWriter()
            legacy = True
        elif '.vtu' in filename:
            writer = vtk.vtkXMLUnstructuredGridWriter()
            legacy = False
        else:
            raise Exception('Extension should be either ".vtu" or ".vtk"')

        writer.SetFileName(filename)
        writer.SetInputData(self)
        if binary and legacy:
            writer.SetFileTypeToBinary()
        writer.Write()

    @property
    def cells(self):
        """ returns a pointer to the cells as a numpy object """
        return vtk_to_numpy(self.GetCells().GetData())

    @property
    def quality(self):
        """
        Returns cell quality
        """
        try:
            import pyansys
        except:
            raise Exception('Install pyansys for this function')
        return pyansys.CellQuality(self)

    def linear_copy(self, deep=False):
        """
        Returns a copy of the input unstructured grid containing only
        linear cells.  Converts the following cell types to their linear
        equivalents.

        - VTK_QUADRATIC_TETRA      --> VTK_TETRA
        - VTK_QUADRATIC_PYRAMID    --> VTK_PYRAMID
        - VTK_QUADRATIC_WEDGE      --> VTK_WEDGE
        - VTK_QUADRATIC_HEXAHEDRON --> VTK_HEXAHEDRON

        Parameters
        ----------
        deep : bool
            When True, makes a copy of the points array.  Default False.

        Returns
        -------
        grid : vtki.UnstructuredGrid
            UnstructuredGrid containing only linear cells.
        """
        lgrid = self.copy(deep)

        # grab the vtk object
        vtk_cell_type = numpy_to_vtk(self.GetCellTypesArray(), deep=True)
        celltype = vtk_to_numpy(vtk_cell_type)
        celltype[celltype == VTK_QUADRATIC_TETRA] = VTK_TETRA
        celltype[celltype == VTK_QUADRATIC_PYRAMID] = VTK_PYRAMID
        celltype[celltype == VTK_QUADRATIC_WEDGE] = VTK_WEDGE
        celltype[celltype == VTK_QUADRATIC_HEXAHEDRON] = VTK_HEXAHEDRON

        vtk_offset = self.GetCellLocationsArray()
        lgrid.SetCells(vtk_cell_type, vtk_offset, self.GetCells())

        return lgrid

    @property
    def celltypes(self):
        return vtk_to_numpy(self.GetCellTypesArray())

    @property
    def offset(self):
        return vtk_to_numpy(self.GetCellLocationsArray())

    def extract_cells(self, ind):
        """
        Returns a subset of the grid

        Parameters
        ----------
        ind : np.ndarray
            Numpy array of cell indices to be extracted.

        Returns
        -------
        subgrid : vtki.UnstructuredGrid
            Subselected grid

        """
        if not isinstance(ind, np.ndarray):
            ind = np.array(ind, np.ndarray)

        if ind.dtype == np.bool:
            ind = ind.nonzero()[0].astype(vtki.ID_TYPE)

        if ind.dtype != vtki.ID_TYPE:
            ind = ind.astype(vtki.ID_TYPE)

        if not ind.flags.c_contiguous:
            ind = np.ascontiguousarray(ind)

        vtk_ind = numpy_to_vtkIdTypeArray(ind, deep=False)

        # Create selection objects
        selectionNode = vtk.vtkSelectionNode()
        selectionNode.SetFieldType(vtk.vtkSelectionNode.CELL)
        selectionNode.SetContentType(vtk.vtkSelectionNode.INDICES)
        selectionNode.SetSelectionList(vtk_ind)

        selection = vtk.vtkSelection()
        selection.AddNode(selectionNode)

        # extract
        extractSelection = vtk.vtkExtractSelection()
        extractSelection.SetInputData(0, self)
        extractSelection.SetInputData(1, selection)
        extractSelection.Update()
        subgrid = UnstructuredGrid(extractSelection.GetOutput())

        # extracts only in float32
        if self.points.dtype is not np.dtype('float32'):
            ind = subgrid.point_arrays['vtkOriginalPointIds']
            subgrid.points = self.points[ind]

        return subgrid

    # def extract_points(self, ind):
    #     """
    #     Returns a subset of an unstructured grid

    #     Parameters
    #     ----------
    #     ind : np.ndarray
    #         Numpy array of point indices to be extracted.

    #     Returns
    #     -------
    #     subgrid : vtki.UnstructuredGrid
    #         Subselected grid.

    #     """
    #     # Convert to vtk indices
    #     if ind.dtype != np.int64:
    #         ind = ind.astype(np.int64)
    #     vtk_ind = numpy_to_vtkIdTypeArray(ind, deep=True)

    #     # Create selection objects
    #     selectionNode = vtk.vtkSelectionNode()
    #     selectionNode.SetFieldType(vtk.vtkSelectionNode.POINT)
    #     selectionNode.SetContentType(vtk.vtkSelectionNode.INDICES)
    #     selectionNode.SetSelectionList(vtk_ind)

    #     selection = vtk.vtkSelection()
    #     selection.AddNode(selectionNode)

    #     # extract
    #     extractSelection = vtk.vtkExtractSelection()
    #     extractSelection.SetInputData(0, self)
    #     extractSelection.SetInputData(1, selection)
    #     extractSelection.Update()
    #     return UnstructuredGrid(extractSelection.GetOutput())

    def merge(self, grid=None, merge_points=True, inplace=True,
              main_has_priority=True):
        """
        Join one or many other grids to this grid.  Grid is updated
        in-place by default.

        Can be used to merge points of adjcent cells when no grids
        are input.

        Parameters
        ----------
        grid : vtk.UnstructuredGrid or list of vtk.UnstructuredGrids
            Grids to merge to this grid.

        merge_points : bool, optional
            Points in exactly the same location will be merged between
            the two meshes.

        inplace : bool, optional
            Updates grid inplace when True.

        main_has_priority : bool, optional
            When this parameter is true and merge_points is true,
            the scalar arrays of the merging grids will be overwritten
            by the original main mesh.

        Returns
        -------
        merged_grid : vtk.UnstructuredGrid
            Merged grid.  Returned when inplace is False.

        Notes
        -----
        When two or more grids are joined, the type and name of each
        scalar array must match or the arrays will be ignored and not
        included in the final merged mesh.
        """
        append_filter = vtk.vtkAppendFilter()
        append_filter.SetMergePoints(merge_points)

        if not main_has_priority:
            append_filter.AddInputData(self)

        if isinstance(grid, vtki.UnstructuredGrid):
            append_filter.AddInputData(grid)
        elif isinstance(grid, list):
            grids = grid
            for grid in grids:
                append_filter.AddInputData(grid)

        if main_has_priority:
            append_filter.AddInputData(self)

        append_filter.Update()
        merged = UnstructuredGrid(append_filter.GetOutput())
        if inplace:
            self.DeepCopy(merged)
        else:
            return merged


class StructuredGrid(vtkStructuredGrid, Grid):
    """
    Extends the functionality of a vtk.vtkStructuredGrid object
    Can be initialized in several ways:

    - Create empty grid
    - Initialize from a vtk.vtkStructuredGrid object
    - Initialize directly from the point arrays

    See _from_arrays in the documentation for more details on initializing
    from point arrays

    Examples
    --------
    >>> grid = StructuredGrid()  # Create empty grid
    >>> grid = StructuredGrid(vtkgrid)  # Initialize from a vtk.vtkStructuredGrid object
    >>> xrng = np.arange(-10, 10, 2)
    >>> yrng = np.arange(-10, 10, 2)
    >>> zrng = np.arange(-10, 10, 2)
    >>> x, y, z = np.meshgrid(xrng, yrng, zrng)
    >>> grid = vtki.StructuredGrid(x, y, z)


    """

    def __init__(self, *args, **kwargs):
        super(StructuredGrid, self).__init__()

        if len(args) == 1:
            if isinstance(args[0], vtk.vtkStructuredGrid):
                self.DeepCopy(args[0])
            elif isinstance(args[0], str):
                self._load_file(args[0])

        elif len(args) == 3:
            arg0_is_arr = isinstance(args[0], np.ndarray)
            arg1_is_arr = isinstance(args[1], np.ndarray)
            arg2_is_arr = isinstance(args[2], np.ndarray)

            if all([arg0_is_arr, arg1_is_arr, arg2_is_arr]):
                self._from_arrays(args[0], args[1], args[2])

    def _from_arrays(self, x, y, z):
        """
        Create VTK structured grid directly from numpy arrays.

        Parameters
        ----------
        x : np.ndarray
            Position of the points in x direction.

        y : np.ndarray
            Position of the points in y direction.

        z : np.ndarray
            Position of the points in z direction.
        """
        if not(x.shape == y.shape == z.shape):
            raise Exception('Input point array shapes must match exactly')

        # make the output points the same precision as the input arrays
        points = np.empty((x.size, 3), x.dtype)
        points[:, 0] = x.ravel('F')
        points[:, 1] = y.ravel('F')
        points[:, 2] = z.ravel('F')

        # ensure that the inputs are 3D
        dim = list(x.shape)
        while len(dim) < 3:
            dim.append(1)

        # Create structured grid
        self.SetDimensions(dim)
        self.SetPoints(vtki.vtk_points(points))

    def _load_file(self, filename):
        """
        Load a structured grid from a file.

        The file extension will select the type of reader to use.  A .vtk
        extension will use the legacy reader, while .vts will select the VTK
        XML reader.

        Parameters
        ----------
        filename : str
            Filename of grid to be loaded.

        """
        # check file exists
        if not os.path.isfile(filename):
            raise Exception('%s does not exist')

        # Check file extention
        if '.vts' in filename:
            legacy_writer = False
        elif '.vtk' in filename:
            legacy_writer = True
        else:
            raise Exception(
                'Extension should be either ".vts" (xml) or ".vtk" (legacy)')

        # Create reader
        if legacy_writer:
            reader = vtk.vtkStructuredGridReader()
        else:
            reader = vtk.vtkXMLStructuredGridReader()

        # load file to self
        reader.SetFileName(filename)
        reader.Update()
        grid = reader.GetOutput()
        self.ShallowCopy(grid)

    def save(self, filename, binary=True):
        """
        Writes a structured grid to disk.

        Parameters
        ----------
        filename : str
            Filename of grid to be written.  The file extension will select the
            type of writer to use.  ".vtk" will use the legacy writer, while
            ".vts" will select the VTK XML writer.

        binary : bool, optional
            Writes as a binary file by default.  Set to False to write ASCII.


        Notes
        -----
        Binary files write much faster than ASCII, but binary files written on
        one system may not be readable on other systems.  Binary can be used
        only with the legacy writer.

        """
        # Use legacy writer if vtk is in filename
        if '.vtk' in filename:
            writer = vtk.vtkStructuredGridWriter()
            legacy = True
        elif '.vts' in filename:
            writer = vtk.vtkXMLStructuredGridWriter()
            legacy = False
        else:
            raise Exception('Extension should be either ".vts" (xml) or' +
                            '".vtk" (legacy)')
        # Write
        writer.SetFileName(filename)
        writer.SetInputData(self)
        if binary and legacy:
            writer.SetFileTypeToBinary()
        writer.Write()

    @property
    def x(self):
        dim = self.GetDimensions()
        return self.points[:, 0].reshape(dim, order='F')

    @property
    def y(self):
        dim = self.GetDimensions()
        return self.points[:, 1].reshape(dim, order='F')

    @property
    def z(self):
        dim = self.GetDimensions()
        return self.points[:, 2].reshape(dim, order='F')

    @property
    def quality(self):
        """
        Computes the minimum scaled jacobian of each cell.  Cells that have
        values below 0 are invalid for a finite element analysis.

        Returns
        -------
        cellquality : np.ndarray
            Minimum scaled jacobian of each cell.  Ranges from -1 to 1.

        Notes
        -----
        Requires pyansys to be installed.

        """
        return UnstructuredGrid(self).quality


class RectilinearGrid(vtkRectilinearGrid, Grid):
    """
    Extends the functionality of a vtk.vtkRectilinearGrid object
    Can be initialized in several ways:

    - Create empty grid
    - Initialize from a vtk.vtkRectilinearGrid object
    - Initialize directly from the point arrays

    See _from_arrays in the documentation for more details on initializing
    from point arrays

    Examples
    --------
    >>> grid = RectilinearGrid()  # Create empty grid
    >>> grid = RectilinearGrid(vtkgrid)  # Initialize from a vtk.vtkRectilinearGrid object
    >>> xrng = np.arange(-10, 10, 2)
    >>> yrng = np.arange(-10, 10, 5)
    >>> zrng = np.arange(-10, 10, 1)
    >>> grid = vtki.RectilinearGrid(xrng, yrng, zrng)


    """

    def __init__(self, *args, **kwargs):
        super(RectilinearGrid, self).__init__()

        if len(args) == 1:
            if isinstance(args[0], vtk.vtkRectilinearGrid):
                self.DeepCopy(args[0])
            elif isinstance(args[0], str):
                self._load_file(args[0])

        elif len(args) == 3:
            arg0_is_arr = isinstance(args[0], np.ndarray)
            arg1_is_arr = isinstance(args[1], np.ndarray)
            arg2_is_arr = isinstance(args[2], np.ndarray)

            if all([arg0_is_arr, arg1_is_arr, arg2_is_arr]):
                self._from_arrays(args[0], args[1], args[2])


    def _from_arrays(self, x, y, z):
        """
        Create VTK rectilinear grid directly from numpy arrays. Each array
        gives the uniques coordinates of the mesh along each axial direction.
        To help ensure you are using this correctly, we take the unique values
        of each argument.

        Parameters
        ----------
        x : np.ndarray
            Coordinates of the nodes in x direction.

        y : np.ndarray
            Coordinates of the nodes in y direction.

        z : np.ndarray
            Coordinates of the nodes in z direction.
        """
        x = np.unique(x.ravel())
        y = np.unique(y.ravel())
        z = np.unique(z.ravel())
        # Set the cell spacings and dimensions of the grid
        self.SetDimensions(len(x), len(y), len(z))
        self.SetXCoordinates(numpy_to_vtk(x))
        self.SetYCoordinates(numpy_to_vtk(y))
        self.SetZCoordinates(numpy_to_vtk(z))


    @property
    def points(self):
        """ returns a pointer to the points as a numpy object """
        x = vtk_to_numpy(self.GetXCoordinates())
        y = vtk_to_numpy(self.GetYCoordinates())
        z = vtk_to_numpy(self.GetZCoordinates())
        xx, yy, zz = np.meshgrid(x,y,z, indexing='ij')
        return np.c_[xx.ravel(), yy.ravel(), zz.ravel()]

    @points.setter
    def points(self, points):
        """ set points without copying """
        if not isinstance(points, np.ndarray):
            raise TypeError('Points must be a numpy array')
        # get the unique coordinates along each axial direction
        x = np.unique(points[:,0])
        y = np.unique(points[:,1])
        z = np.unique(points[:,2])
        # Set the vtk coordinates
        self._from_arrays(x, y, z)
        #self._point_ref = points


    def _load_file(self, filename):
        """
        Load a rectilinear grid from a file.

        The file extension will select the type of reader to use.  A .vtk
        extension will use the legacy reader, while .vts will select the VTK
        XML reader.

        Parameters
        ----------
        filename : str
            Filename of grid to be loaded.

        """
        # check file exists
        if not os.path.isfile(filename):
            raise Exception('%s does not exist')

        # Check file extention
        if '.vtr' in filename:
            legacy_writer = False
        elif '.vtk' in filename:
            legacy_writer = True
        else:
            raise Exception(
                'Extension should be either ".vtr" (xml) or ".vtk" (legacy)')

        # Create reader
        if legacy_writer:
            reader = vtk.vtkRectilinearGridReader()
        else:
            reader = vtk.vtkXMLRectilinearGridReader()

        # load file to self
        reader.SetFileName(filename)
        reader.Update()
        grid = reader.GetOutput()
        self.ShallowCopy(grid)

    def save(self, filename, binary=True):
        """
        Writes a rectilinear grid to disk.

        Parameters
        ----------
        filename : str
            Filename of grid to be written.  The file extension will select the
            type of writer to use.  ".vtk" will use the legacy writer, while
            ".vts" will select the VTK XML writer.

        binary : bool, optional
            Writes as a binary file by default.  Set to False to write ASCII.


        Notes
        -----
        Binary files write much faster than ASCII, but binary files written on
        one system may not be readable on other systems.  Binary can be used
        only with the legacy writer.

        """
        # Use legacy writer if vtk is in filename
        if '.vtk' in filename:
            writer = vtk.vtkRectilinearGridWriter()
            legacy = True
        elif '.vtr' in filename:
            writer = vtk.vtkXMLRectilinearGridWriter()
            legacy = False
        else:
            raise Exception('Extension should be either ".vtr" (xml) or' +
                            '".vtk" (legacy)')
        # Write
        writer.SetFileName(filename)
        writer.SetInputData(self)
        if binary and legacy:
            writer.SetFileTypeToBinary()
        writer.Write()

    @property
    def x(self):
        return vtk_to_numpy(self.GetXCoordinates())

    @property
    def y(self):
        return vtk_to_numpy(self.GetYCoordinates())

    @property
    def z(self):
        return vtk_to_numpy(self.GetZCoordinates())

    # @property
    # def quality(self):
    #     """
    #     Computes the minimum scaled jacobian of each cell.  Cells that have
    #     values below 0 are invalid for a finite element analysis.
    #
    #     Returns
    #     -------
    #     cellquality : np.ndarray
    #         Minimum scaled jacobian of each cell.  Ranges from -1 to 1.
    #
    #     Notes
    #     -----
    #     Requires pyansys to be installed.
    #
    #     """
    #     return UnstructuredGrid(self).quality




class UniformGrid(vtkImageData, Grid):
    """
    Extends the functionality of a vtk.vtkImageData object
    Can be initialized in several ways:

    - Create empty grid
    - Initialize from a vtk.vtkImageData object
    - Initialize directly from the point arrays

    See ``_from_specs`` in the documentation for more details on initializing
    from point arrays

    Examples
    --------
    >>> grid = UniformGrid()  # Create empty grid
    >>> grid = UniformGrid(vtkgrid)  # Initialize from a vtk.vtkImageData object
    >>> dims = (10, 10, 10)
    >>> grid = vtki.UniformGrid(dims) # Using default spacing and origin
    >>> spacing = (2, 1, 5)
    >>> grid = vtki.UniformGrid(dims, spacing) # Usign default origin
    >>> origin = (10, 35, 50)
    >>> grid = vtki.UniformGrid(dims, spacing, origin) # Everything is specified

    """

    def __init__(self, *args, **kwargs):
        super(UniformGrid, self).__init__()

        if len(args) == 1:
            if isinstance(args[0], vtk.vtkImageData):
                self.DeepCopy(args[0])
            elif isinstance(args[0], str):
                self._load_file(args[0])
            else:
                arg0_is_valid = len(args[0]) == 3
                self._from_specs(args[0])

        elif len(args) > 1 and len(args) < 4:
            arg0_is_valid = len(args[0]) == 3
            arg1_is_valid = False
            if len(args) > 1:
                arg1_is_valid = len(args[1]) == 3
            arg2_is_valid = False
            if len(args) > 2:
                arg2_is_valid = len(args[2]) == 3

            if all([arg0_is_valid, arg1_is_valid, arg2_is_valid]):
                self._from_specs(args[0], args[1], args[2])
            elif all([arg0_is_valid, arg1_is_valid]):
                self._from_specs(args[0], args[1])


    def _from_specs(self, dims, spacing=(1.0,1.0,1.0), origin=(0.0, 0.0, 0.0)):
        """
        Create VTK image data directly from numpy arrays. A uniform grid is
        defined by the node spacings for each axis (uniform along each
        individual axis) and the number of nodes on each axis. These are
        relative to a specified origin (default is ``(0.0, 0.0, 0.0)``).

        Parameters
        ----------
        dims : tuple(int)
            Length 3 tuple of ints specifying how many nodes along each axis

        spacing : tuple(float)
            Length 3 tuple of floats/ints specifying the node spacings for each axis

        origin : tuple(float)
            Length 3 tuple of floats/ints specifying minimum value for each axis
        """
        xn, yn, zn = dims[0], dims[1], dims[2]
        xs, ys, zs = spacing[0], spacing[1], spacing[2]
        xo, yo, zo = origin[0], origin[1], origin[2]
        self.SetDimensions(xn, yn, zn)
        self.SetOrigin(xo, yo, zo)
        self.SetSpacing(xs, ys, zs)


    @property
    def points(self):
        """ returns a pointer to the points as a numpy object """
        # Get grid dimensions
        nx, ny, nz = self.GetDimensions()
        nx -= 1
        ny -= 1
        nz -= 1
        # get the points and convert to spacings
        dx, dy, dz = self.GetSpacing()
        # Now make the cell arrays
        ox, oy, oz = self.GetOrigin()
        x = np.cumsum(np.full(nx, dx)) + ox
        y = np.cumsum(np.full(ny, dy)) + oy
        z = np.cumsum(np.full(nz, dz)) + oz
        xx, yy, zz = np.meshgrid(x,y,z, indexing='ij')
        return np.c_[xx.ravel(), yy.ravel(), zz.ravel()]

    @points.setter
    def points(self, points):
        """ set points without copying """
        if not isinstance(points, np.ndarray):
            raise TypeError('Points must be a numpy array')
        # get the unique coordinates along each axial direction
        x = np.unique(points[:,0])
        y = np.unique(points[:,1])
        z = np.unique(points[:,2])
        nx, ny, nz = len(x), len(y), len(z)
        # TODO: this needs to be tested (unique might return a tuple)
        dx, dy, dz = np.unique(np.diff(x)), np.unique(np.diff(y)), np.unique(np.diff(z))
        ox, oy, oz = np.min(x), np.min(y), np.min(z)
        # Build the vtk object
        self._from_specs(self, (nx,ny,nz), (dx,dy,dz), (ox,oy,oz))
        #self._point_ref = points


    def _load_file(self, filename):
        """
        Load image data from a file.

        The file extension will select the type of reader to use.  A ``.vtk``
        extension will use the legacy reader, while ``.vti`` will select the VTK
        XML reader.

        Parameters
        ----------
        filename : str
            Filename of grid to be loaded.

        """
        # check file exists
        if not os.path.isfile(filename):
            raise Exception('%s does not exist')

        # Check file extention
        if '.vti' in filename:
            legacy_writer = False
        elif '.vtk' in filename:
            legacy_writer = True
        else:
            raise Exception(
                'Extension should be either ".vti" (xml) or ".vtk" (legacy)')

        # Create reader
        if legacy_writer:
            reader = vtk.vtkDataSetReader()
        else:
            reader = vtk.vtkXMLImageDataReader()

        # load file to self
        reader.SetFileName(filename)
        reader.Update()
        grid = reader.GetOutput()
        self.ShallowCopy(grid)

    def save(self, filename, binary=True):
        """
        Writes image data grid to disk.

        Parameters
        ----------
        filename : str
            Filename of grid to be written.  The file extension will select the
            type of writer to use.  ".vtk" will use the legacy writer, while
            ".vti" will select the VTK XML writer.

        binary : bool, optional
            Writes as a binary file by default.  Set to False to write ASCII.


        Notes
        -----
        Binary files write much faster than ASCII, but binary files written on
        one system may not be readable on other systems.  Binary can be used
        only with the legacy writer.

        """
        # Use legacy writer if vtk is in filename
        if '.vtk' in filename:
            writer = vtk.vtkDataSetWriter()
            legacy = True
        elif '.vti' in filename:
            writer = vtk.vtkXMLImageDataWriter()
            legacy = False
        else:
            raise Exception('Extension should be either ".vti" (xml) or' +
                            '".vtk" (legacy)')
        # Write
        writer.SetFileName(filename)
        writer.SetInputData(self)
        if binary and legacy:
            writer.SetFileTypeToBinary()
        writer.Write()

    @property
    def x(self):
        return self.points[:, 0]

    @property
    def y(self):
        return self.points[:, 1]

    @property
    def z(self):
        return self.points[:, 2]

    # @property
    # def quality(self):
    #     """
    #     Computes the minimum scaled jacobian of each cell.  Cells that have
    #     values below 0 are invalid for a finite element analysis.
    #
    #     Returns
    #     -------
    #     cellquality : np.ndarray
    #         Minimum scaled jacobian of each cell.  Ranges from -1 to 1.
    #
    #     Notes
    #     -----
    #     Requires pyansys to be installed.
    #
    #     """
    #     return UnstructuredGrid(self).quality
