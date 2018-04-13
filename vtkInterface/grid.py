"""
Sub-classes for vtk.vtkUnstructuredGrid and vtk.vtkStructuredGrid
"""
import os
import numpy as np
import vtkInterface
import logging

log = logging.getLogger(__name__)
log.setLevel('CRITICAL')

# allow readthedocs to parse objects
try:
    import vtk
    from vtk import vtkUnstructuredGrid, vtkStructuredGrid
    from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtkIdTypeArray
    from vtk.util.numpy_support import numpy_to_vtk

    # cell types
    VTK_TRIANGLE = vtk.VTK_TRIANGLE
    VTK_QUAD = vtk.VTK_QUAD
    VTK_QUADRATIC_TRIANGLE = vtk.VTK_QUADRATIC_TRIANGLE
    VTK_QUADRATIC_QUAD = vtk.VTK_QUADRATIC_QUAD
    VTK_HEXAHEDRON = vtk.VTK_HEXAHEDRON
    VTK_PYRAMID = vtk.VTK_PYRAMID
    VTK_TETRA = vtk.VTK_TETRA
    VTK_WEDGE = vtk.VTK_WEDGE
    VTK_QUADRATIC_TETRA = vtk.VTK_QUADRATIC_TETRA
    VTK_QUADRATIC_PYRAMID = vtk.VTK_QUADRATIC_PYRAMID
    VTK_QUADRATIC_WEDGE = vtk.VTK_QUADRATIC_WEDGE
    VTK_QUADRATIC_HEXAHEDRON = vtk.VTK_QUADRATIC_HEXAHEDRON

except:
    # create dummy classes
    class vtkUnstructuredGrid(object):
        def __init__(self, *args, **kwargs):
            pass

    # create dummy class
    class vtkStructuredGrid(object):
        def __init__(self, *args, **kwargs):
            pass


class Grid(vtkInterface.Common):
    """ Class in common with structured and unstructured grids """

    def __init__(self, *args, **kwargs):
        pass

    def PlotCurvature(self, curvtype='mean', rng=None):
        """
        Plots the curvature of the external surface of the grid

        Parameters
        ----------
        curvtype : str, optional
            One of the following strings indicating curvature type

            - mean
            - gaussian
            - maximum
            - minimum

        rng : list, optional
            Minimum and maximum limits on curvature plot

        """
        # extract surface and plot its curvature
        self.ExtractExteriorTri()[0].PlotCurvature(curvtype, rng)

    @property
    def volume(self):
        """
        Computes volume by extracting the external surface and
        computing interior volume
        """

        surf = self.ExtractSurface().TriFilter()
        mass = vtk.vtkMassProperties()
        mass.SetInputData(surf)
        return mass.GetVolume()

    def ExtractExteriorTri(self):
        """
        Creates an all tri surface mesh

        Returns
        -------
        trisurf : vtkInterface.PolyData
            All triangle mesh of the grid

        extsurf : vtkInterface.PolyData
            Surface mesh of the grid

        """
        # Return triangle mesh as well as original
        surf = self.ExtractSurface()
        return surf.TriFilter(), surf

    def ExtractSurface(self, pass_pointid=True, pass_cellid=True):
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
        extsurf : vtkInterface.PolyData
            Surface mesh of the grid
        """
        surf_filter = vtk.vtkDataSetSurfaceFilter()
        surf_filter.SetInputData(self)
        if pass_pointid:
            surf_filter.PassThroughCellIdsOn()
        if pass_cellid:
            surf_filter.PassThroughPointIdsOn()
        surf_filter.Update()
        return vtkInterface.PolyData(surf_filter.GetOutput())

    def ExtractSurfaceInd(self):
        """
        Output the surface indices of a grid

        Returns
        -------
        surf_ind : np.ndarray
            Indices of the surface points.

        """
        # Extract surface mesh
        surf = self.ExtractSurface(pass_cellid=False)
        return surf.GetPointScalars('vtkOriginalPointIds')

    def ExtractEdges(self, feature_angle=30, boundary_edges=True,
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
        edges : vtkInterface.vtkPolyData
            Extracted edges

        """
        surf = self.ExtractSurface()
        return surf.ExtractEdges(feature_angle, boundary_edges,
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

        if len(args) == 1:
            if isinstance(args[0], vtk.vtkUnstructuredGrid):
                self.ShallowCopy(args[0])

            elif isinstance(args[0], str):
                self.LoadFile(args[0])

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
                if 'deep' in kwargs:
                    deep = kwargs['deep']
                else:
                    deep = True
                self.MakeFromArrays(args[0], args[1], args[2], args[3], deep)
            else:
                raise Exception('All input types must be np.ndarray')

    def MakeFromArrays(self, offset, cells, cell_type, points, deep=True):
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

        >>> grid = vtkInterface.UnstructuredGrid(offset, cells, cell_type, points)
        
        """

        if offset.dtype != vtkInterface.ID_TYPE:
            offset = offset.astype(np.int64)

        if cells.dtype != vtkInterface.ID_TYPE:
            cells = cells.astype(vtkInterface.ID_TYPE)

        if not cells.flags['C_CONTIGUOUS']:
            cells = np.ascontiguousarray(cells)

        if cells.ndim != 1:
            cells = cells.ravel()

        if cell_type.dtype != np.uint8:
            cell_type = cell_type.astype(np.uint8)

        # Get number of cells
        ncells = cell_type.size

        # Convert to vtk arrays
        cell_type = numpy_to_vtk(cell_type, deep=deep)
        offset = numpy_to_vtkIdTypeArray(offset, deep=deep)

        vtkcells = vtk.vtkCellArray()
        vtkcells.SetCells(ncells, numpy_to_vtkIdTypeArray(cells, deep=deep))

        # Convert points to vtkPoints object
        points = vtkInterface.MakevtkPoints(points, deep=deep)

        # Create unstructured grid
        self.SetPoints(points)
        self.SetCells(cell_type, offset, vtkcells)

    def LoadFile(self, filename):
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
            legacy_writer = False
        elif '.vtk' in filename:
            legacy_writer = True
        else:
            raise Exception(
                'Extension should be either ".vtu" (xml) or ".vtk" (legacy)')

        # Create reader
        if legacy_writer:
            reader = vtk.vtkUnstructuredGridReader()
        else:
            reader = vtk.vtkXMLUnstructuredGridReader()

        # load file to self
        reader.SetFileName(filename)
        reader.Update()
        grid = reader.GetOutput()
        self.ShallowCopy(grid)

    def Write(self, filename, binary=True):
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
        only with the legacy writer.

        """
        # Use legacy writer if vtk is in filename
        if '.vtk' in filename:
            writer = vtk.vtkUnstructuredGridWriter()
            legacy = True
        elif '.vtu' in filename:
            writer = vtk.vtkXMLUnstructuredGridWriter()
            legacy = False
        else:
            raise Exception('Extension should be either ".vtu" (xml) or' +
                            '".vtk" (legacy)')
        # Write
        writer.SetFileName(filename)
        writer.SetInputData(self)
        if binary and legacy:
            writer.SetFileTypeToBinary
        writer.Write()

    def GetNumpyCells(self, dtype=None):
        """
        Returns a numpy array of cells.

        Parameters
        ----------
        dtype : np.dtype, optional
            When specified, the array will be converted to be the specified
            data type.

        Returns
        -------
        cells : np.ndarray
            Array identifying cell indices.

        """

        # grab cell data
        vtkarr = self.GetCells().GetData()
        if vtkarr:
            cells = vtk_to_numpy(vtkarr)
        else:
            return None

        if dtype:
            if cells.dtype != dtype:
                cells = cells.astype(dtype)

        return cells

    @property
    def cells(self):
        """ returns a pointer to the cells as a numpy object """
        return vtk_to_numpy(self.GetCells().GetData())

    @property
    def quality(self):
        """
        Calls CellQuality
        """
        return self.CellQuality()

    def CellQuality(self):
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
        try:
            import pyansys
        except:
            raise Exception('Install pyansys for this function')

        return pyansys.CellQuality(self)

    def LinearGridCopy(self, deep=False):
        """
        Returns a copy of the input unstructured grid containing only
        linear cells.  Converts the following cell types to their linear
        equivalents.

        - VTK_QUADRATIC_QUAD       --> VTK_QUADRATIC
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
        grid : vtkInterface.UnstructuredGrid
            UnstructuredGrid containing only linear cells.
        """
        lgrid = self.Copy(deep)

        vtk_cell_type = numpy_to_vtk(self.GetCellTypesArray(), deep=True)
        celltype = vtk_to_numpy(vtk_cell_type)
        celltype[celltype == VTK_QUADRATIC_TETRA] = VTK_TETRA
        celltype[celltype == VTK_QUADRATIC_PYRAMID] = VTK_PYRAMID
        celltype[celltype == VTK_QUADRATIC_WEDGE] = VTK_WEDGE
        celltype[celltype == VTK_QUADRATIC_HEXAHEDRON] = VTK_HEXAHEDRON

        offset = self.GetCellLocationsArray()
        lgrid.SetCells(vtk_cell_type, offset, self.GetCells())
        return lgrid

    @property
    def celltypes(self):
        return vtk_to_numpy(self.GetCellTypesArray())

    @property
    def offset(self):
        return vtk_to_numpy(self.GetCellLocationsArray())

    def ExtractSelectionCells(self, ind):
        """
        Returns a subset of the grid

        Parameters
        ----------
        ind : np.ndarray
            Numpy array of cell indices to be extracted.

        Returns
        -------
        subgrid : vtkInterface.UnstructuredGrid
            Subselected grid

        """
        # Convert to vtk indices
        if not isinstance(ind, np.ndarray):
            ind = np.array(ind, np.int64)

        if ind.dtype == np.bool:
            ind = ind.nonzero()[0]

        if ind.dtype != np.int64:
            ind = ind.astype(np.int64)

        if not ind.flags.c_contiguous:
            ind = np.ascontiguousarray(ind)

        vtk_ind = numpy_to_vtkIdTypeArray(ind, deep=True)

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
        if self.GetNumpyPoints().dtype is not np.dtype('float32'):
            ind = subgrid.GetPointScalars('vtkOriginalPointIds')
            subgrid.SetNumpyPoints(self.GetNumpyPoints()[ind])

        return subgrid

    def ExtractSelectionPoints(self, ind):
        """
        Returns a subset of an unstructured grid

        Parameters
        ----------
        ind : np.ndarray
            Numpy array of point indices to be extracted.

        Returns
        -------
        subgrid : vtkInterface.UnstructuredGrid
            Subselected grid.

        """
        # Convert to vtk indices
        if ind.dtype != np.int64:
            ind = ind.astype(np.int64)
        vtk_ind = numpy_to_vtkIdTypeArray(ind, deep=True)

        # Create selection objects
        selectionNode = vtk.vtkSelectionNode()
        selectionNode.SetFieldType(vtk.vtkSelectionNode.POINT)
        selectionNode.SetContentType(vtk.vtkSelectionNode.INDICES)
        selectionNode.SetSelectionList(vtk_ind)

        selection = vtk.vtkSelection()
        selection.AddNode(selectionNode)

        # extract
        extractSelection = vtk.vtkExtractSelection()
        extractSelection.SetInputData(0, self)
        extractSelection.SetInputData(1, selection)
        extractSelection.Update()
        return UnstructuredGrid(extractSelection.GetOutput())


class StructuredGrid(vtkStructuredGrid, Grid):
    """
    Extends the functionality of a vtk.vtkStructuredGrid object
    Can be initialized in several ways:

    - Create empty grid
    - Initialize from a vtk.vtkStructuredGrid object
    - Initialize directly from the point arrays

    See MakeFromArrays in the documentation for more details on initializing
    from point arrays

    Examples
    --------
    >>> grid = StructuredGrid()  # Create empty grid
    >>> grid = StructuredGrid(vtkgrid)  # Initialize from a vtk.vtkStructuredGrid object
    >>> grid = StructuredGrid(x, y, z)  # Initialize directly from the point arrays

    """

    def __init__(self, *args, **kwargs):
        super(StructuredGrid, self).__init__()

        if len(args) == 1:
            if isinstance(args[0], vtk.vtkStructuredGrid):
                self.DeepCopy(args[0])
            elif isinstance(args[0], str):
                self.LoadFile(args[0])

        elif len(args) == 3:
            arg0_is_arr = isinstance(args[0], np.ndarray)
            arg1_is_arr = isinstance(args[1], np.ndarray)
            arg2_is_arr = isinstance(args[2], np.ndarray)

            if all([arg0_is_arr, arg1_is_arr, arg2_is_arr]):
                self.MakeFromArrays(args[0], args[1], args[2])

    def MakeFromArrays(self, x, y, z):
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

        Example
        -------
        >>> x = np.arange(-10, 10, 0.25)
        >>> y = np.arange(-10, 10, 0.25)
        >>> x, y = np.meshgrid(x, y)
        >>> r = np.sqrt(x**2 + y**2)
        >>> z = np.sin(r)
        >>> grid = vtkInterface.StructuredGrid(x, y, z)

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
        self.SetPoints(vtkInterface.MakevtkPoints(points))

    def LoadFile(self, filename):
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

    def Write(self, filename, binary=True):
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
            writer.SetFileTypeToBinary
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

    def CellQuality(self):
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
        try:
            import pyansys
        except:
            raise Exception('Install pyansys for this function')

        return pyansys.CellQuality(UnstructuredGrid(self))

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
        try:
            import pyansys
        except:
            raise Exception('Install pyansys for this function')

        return pyansys.CellQuality(UnstructuredGrid(self))
