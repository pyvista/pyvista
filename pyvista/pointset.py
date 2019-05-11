"""
Sub-classes for vtk.vtkPolyData
"""
import logging
import os

import numpy as np
import vtk
from vtk import (VTK_HEXAHEDRON, VTK_PYRAMID, VTK_QUAD,
                 VTK_QUADRATIC_HEXAHEDRON, VTK_QUADRATIC_PYRAMID,
                 VTK_QUADRATIC_QUAD, VTK_QUADRATIC_TETRA,
                 VTK_QUADRATIC_TRIANGLE, VTK_QUADRATIC_WEDGE, VTK_TETRA,
                 VTK_TRIANGLE, VTK_WEDGE, vtkPolyData, vtkStructuredGrid,
                 vtkUnstructuredGrid)
from vtk.util.numpy_support import (numpy_to_vtk, numpy_to_vtkIdTypeArray,
                                    vtk_to_numpy)

import pyvista
from pyvista.filters import _get_output

log = logging.getLogger(__name__)
log.setLevel('CRITICAL')


class PolyData(vtkPolyData, pyvista.Common):
    """
    Extends the functionality of a vtk.vtkPolyData object

    Can be initialized in several ways:

    - Create an empty mesh
    - Initialize from a vtk.vtkPolyData
    - Using vertices
    - Using vertices and faces
    - From a file

    Examples
    --------
    >>> import pyvista
    >>> from pyvista import examples
    >>> import vtk
    >>> import numpy as np

    >>> surf = pyvista.PolyData()  # Create an empty mesh

    >>> # Initialize from a vtk.vtkPolyData object
    >>> vtkobj = vtk.vtkPolyData()
    >>> surf = pyvista.PolyData(vtkobj)

    >>> # initialize from just vertices
    >>> vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 0.5, 0], [0, 0.5, 0],])
    >>> surf = pyvista.PolyData(vertices)

    >>> # initialize from vertices and faces
    >>> faces = np.hstack([[3, 0, 1, 2], [3, 0, 3, 2]]).astype(np.int8)
    >>> surf = pyvista.PolyData(vertices, faces)

    >>>  # initialize from a filename
    >>> surf = pyvista.PolyData(examples.antfile)
    """

    def __init__(self, *args, **kwargs):
        super(PolyData, self).__init__()

        deep = kwargs.pop('deep', False)

        if not args:
            return
        elif len(args) == 1:
            if isinstance(args[0], vtk.vtkPolyData):
                if deep:
                    self.DeepCopy(args[0])
                else:
                    self.ShallowCopy(args[0])
            elif isinstance(args[0], str):
                self._load_file(args[0])
            elif isinstance(args[0], np.ndarray):
                points = args[0]
                if points.ndim != 2:
                    points = points.reshape((-1, 3))

                npoints = points.shape[0]
                cells = np.hstack((np.ones((npoints, 1)),
                                   np.arange(npoints).reshape(-1, 1)))
                cells = np.ascontiguousarray(cells, dtype=pyvista.ID_TYPE)
                cells = np.reshape(cells, (2*npoints))
                self._from_arrays(points, cells, deep, verts=True)
            else:
                raise TypeError('Invalid input type')

        elif len(args) == 2:
            arg0_is_array = isinstance(args[0], np.ndarray)
            arg1_is_array = isinstance(args[1], np.ndarray)
            if arg0_is_array and arg1_is_array:
                self._from_arrays(args[0], args[1], deep)
            else:
                raise TypeError('Invalid input type')
        else:
            raise TypeError('Invalid input type')

    def __repr__(self):
        return pyvista.Common.__repr__(self)

    def __str__(self):
        return pyvista.Common.__str__(self)

    def _load_file(self, filename):
        """Load a surface mesh from a mesh file.

        Mesh file may be an ASCII or binary ply, stl, or vtk mesh file.

        Parameters
        ----------
        filename : str
            Filename of mesh to be loaded.  File type is inferred from the
            extension of the filename

        Notes
        -----
        Binary files load much faster than ASCII.

        """
        filename = os.path.abspath(os.path.expanduser(filename))
        # test if file exists
        if not os.path.isfile(filename):
            raise Exception('File %s does not exist' % filename)

        # Get extension
        ext = pyvista.get_ext(filename)

        # Select reader
        if ext == '.ply':
            reader = vtk.vtkPLYReader()
        elif ext == '.stl':
            reader = vtk.vtkSTLReader()
        elif ext == '.vtk':
            reader = vtk.vtkPolyDataReader()
        elif ext == '.vtp':
            reader = vtk.vtkXMLPolyDataReader()
        elif ext == '.obj':
            reader = vtk.vtkOBJReader()
        else:
            raise TypeError('Filetype must be either "ply", "stl", "vtk", "vtp", or "obj".')

        # Load file
        reader.SetFileName(filename)
        reader.Update()
        self.ShallowCopy(reader.GetOutput())

        # sanity check
        if not np.any(self.points):
            raise AssertionError('Empty or invalid file')

    @property
    def lines(self):
        return vtk_to_numpy(self.GetLines().GetData())

    @lines.setter
    def lines(self, lines):
        if lines.dtype != pyvista.ID_TYPE:
            lines = lines.astype(pyvista.ID_TYPE)

        # get number of faces
        if lines.ndim == 1:
            div = lines.size / 3.0
            assert not div % 1, 'Invalid lines array'
            nlines = int(div)
        else:
            nlines = lines.shape[0]

        vtkcells = vtk.vtkCellArray()
        vtkcells.SetCells(nlines, numpy_to_vtkIdTypeArray(lines, deep=False))
        self.SetLines(vtkcells)

    @property
    def faces(self):
        """ returns a pointer to the points as a numpy object """
        return vtk_to_numpy(self.GetPolys().GetData())

    @faces.setter
    def faces(self, faces):
        """ set faces without copying """
        if faces.dtype != pyvista.ID_TYPE:
            faces = faces.astype(pyvista.ID_TYPE)

        # get number of faces
        if faces.ndim == 1:
            log.debug('efficiency warning')
            c = 0
            nfaces = 0
            while c < faces.size:
                c += faces[c] + 1
                nfaces += 1
        else:
            nfaces = faces.shape[0]

        vtkcells = vtk.vtkCellArray()
        vtkcells.SetCells(nfaces, numpy_to_vtkIdTypeArray(faces, deep=False))
        if faces.ndim > 1 and faces.shape[1] == 2:
            self.SetVerts(vtkcells)
        else:
            self.SetPolys(vtkcells)
        self._face_ref = faces
        self.Modified()

    # @property
    # def lines(self):
    #     """ returns a copy of the indices of the lines """
    #     lines = vtk_to_numpy(self.GetLines().GetData()).reshape((-1, 3))
    #     return np.ascontiguousarray(lines[:, 1:])

    def _from_arrays(self, vertices, faces, deep=True, verts=False):
        """
        Set polygons and points from numpy arrays

        Parameters
        ----------
        vertices : np.ndarray of dtype=np.float32 or np.float64
            Vertex array.  3D points.

        faces : np.ndarray of dtype=np.int64
            Face index array.  Faces can contain any number of points.

        Examples
        --------
        >>> import numpy as np
        >>> import pyvista
        >>> vertices = np.array([[0, 0, 0],
        ...                      [1, 0, 0],
        ...                      [1, 1, 0],
        ...                      [0, 1, 0],
        ...                      [0.5, 0.5, 1]])
        >>> faces = np.hstack([[4, 0, 1, 2, 3],
        ...                    [3, 0, 1, 4],
        ...                    [3, 1, 2, 4]])  # one square and two triangles
        >>> surf = pyvista.PolyData(vertices, faces)

        """
        if deep or verts:
            vtkpoints = vtk.vtkPoints()
            vtkpoints.SetData(numpy_to_vtk(vertices, deep=deep))
            self.SetPoints(vtkpoints)

            # Convert to a vtk array
            vtkcells = vtk.vtkCellArray()
            if faces.dtype != pyvista.ID_TYPE:
                faces = faces.astype(pyvista.ID_TYPE)

            # get number of faces
            if faces.ndim == 1:
                c = 0
                nfaces = 0
                while c < faces.size:
                    c += faces[c] + 1
                    nfaces += 1
            else:
                nfaces = faces.shape[0]

            idarr = numpy_to_vtkIdTypeArray(faces.ravel(), deep=deep)
            vtkcells.SetCells(nfaces, idarr)
            if (faces.ndim > 1 and faces.shape[1] == 2) or verts:
                self.SetVerts(vtkcells)
            else:
                self.SetPolys(vtkcells)
        else:
            self.points = vertices
            self.faces = faces

    def edge_mask(self, angle):
        """
        Returns a mask of the points of a surface mesh that have a surface
        angle greater than angle

        Parameters
        ----------
        angle : float
            Angle to consider an edge.

        """
        self.point_arrays['point_ind'] = np.arange(self.n_points)
        featureEdges = vtk.vtkFeatureEdges()
        featureEdges.SetInputData(self)
        featureEdges.FeatureEdgesOn()
        featureEdges.BoundaryEdgesOff()
        featureEdges.NonManifoldEdgesOff()
        featureEdges.ManifoldEdgesOff()
        featureEdges.SetFeatureAngle(angle)
        featureEdges.Update()
        edges = _get_output(featureEdges)
        orig_id = pyvista.point_scalar(edges, 'point_ind')

        return np.in1d(self.point_arrays['point_ind'], orig_id,
                       assume_unique=True)

    def __sub__(self, cutting_mesh):
        """ subtract two meshes """
        return self.boolean_cut(cutting_mesh)

    @property
    def n_faces(self):
        """alias for ``n_cells``"""
        return self.n_cells

    @property
    def number_of_faces(self):
        """ returns the number of cells """
        return self.n_cells

    def boolean_cut(self, cut, tolerance=1E-5, inplace=False):
        """
        Performs a Boolean cut using another mesh.

        Parameters
        ----------
        cut : pyvista.PolyData
            Mesh making the cut

        inplace : bool, optional
            Updates mesh in-place while returning nothing.

        Returns
        -------
        mesh : pyvista.PolyData
            The cut mesh when inplace=False

        """
        bfilter = vtk.vtkBooleanOperationPolyDataFilter()
        bfilter.SetOperationToIntersection()
        # bfilter.SetOperationToDifference()

        bfilter.SetInputData(1, cut)
        bfilter.SetInputData(0, self)
        bfilter.ReorientDifferenceCellsOff()
        bfilter.SetTolerance(tolerance)
        bfilter.Update()

        mesh = _get_output(bfilter)
        if inplace:
            self.overwrite(mesh)
        else:
            return mesh

    def __add__(self, mesh):
        """ adds two meshes together """
        return self.boolean_add(mesh)

    def boolean_add(self, mesh, inplace=False):
        """
        Add a mesh to the current mesh.  Does not attempt to "join"
        the meshes.

        Parameters
        ----------
        mesh : pyvista.PolyData
            The mesh to add.

        inplace : bool, optional
            Updates mesh in-place while returning nothing.

        Returns
        -------
        joinedmesh : pyvista.PolyData
            Initial mesh and the new mesh when inplace=False.

        """
        vtkappend = vtk.vtkAppendPolyData()
        vtkappend.AddInputData(self)
        vtkappend.AddInputData(mesh)
        vtkappend.Update()

        mesh = _get_output(vtkappend)
        if inplace:
            self.overwrite(mesh)
        else:
            return mesh

    def boolean_union(self, mesh, inplace=False):
        """
        Combines two meshes and attempts to create a manifold mesh.

        Parameters
        ----------
        mesh : pyvista.PolyData
            The mesh to perform a union against.

        inplace : bool, optional
            Updates mesh in-place while returning nothing.

        Returns
        -------
        union : pyvista.PolyData
            The union mesh when inplace=False.

        """
        bfilter = vtk.vtkBooleanOperationPolyDataFilter()
        bfilter.SetOperationToUnion()
        bfilter.SetInputData(1, mesh)
        bfilter.SetInputData(0, self)
        bfilter.ReorientDifferenceCellsOff()
        bfilter.Update()

        mesh = _get_output(bfilter)
        if inplace:
            self.overwrite(mesh)
        else:
            return mesh

    def boolean_difference(self, mesh, inplace=False):
        """
        Combines two meshes and retains only the volume in common
        between the meshes.

        Parameters
        ----------
        mesh : pyvista.PolyData
            The mesh to perform a union against.

        inplace : bool, optional
            Updates mesh in-place while returning nothing.

        Returns
        -------
        union : pyvista.PolyData
            The union mesh when inplace=False.

        """
        bfilter = vtk.vtkBooleanOperationPolyDataFilter()
        bfilter.SetOperationToDifference()
        bfilter.SetInputData(1, mesh)
        bfilter.SetInputData(0, self)
        bfilter.ReorientDifferenceCellsOff()
        bfilter.Update()

        mesh = _get_output(bfilter)
        if inplace:
            self.overwrite(mesh)
        else:
            return mesh

    def curvature(self, curv_type='mean'):
        """
        Returns the pointwise curvature of a mesh

        Parameters
        ----------
        mesh : vtk.polydata
            vtk polydata mesh

        curvature string, optional
            One of the following strings
            Mean
            Gaussian
            Maximum
            Minimum

        Returns
        -------
        curvature : np.ndarray
            Curvature values

        """
        curv_type = curv_type.lower()

        # Create curve filter and compute curvature
        curvefilter = vtk.vtkCurvatures()
        curvefilter.SetInputData(self)
        if curv_type == 'mean':
            curvefilter.SetCurvatureTypeToMean()
        elif curv_type == 'gaussian':
            curvefilter.SetCurvatureTypeToGaussian()
        elif curv_type == 'maximum':
            curvefilter.SetCurvatureTypeToMaximum()
        elif curv_type == 'minimum':
            curvefilter.SetCurvatureTypeToMinimum()
        else:
            raise Exception('Curv_Type must be either "Mean", ' +
                            '"Gaussian", "Maximum", or "Minimum"')
        curvefilter.Update()

        # Compute and return curvature
        curv = _get_output(curvefilter)
        return vtk_to_numpy(curv.GetPointData().GetScalars())

    def save(self, filename, binary=True):
        """
        Writes a surface mesh to disk.

        Written file may be an ASCII or binary ply, stl, or vtk mesh file.

        Parameters
        ----------
        filename : str
            Filename of mesh to be written.  File type is inferred from
            the extension of the filename unless overridden with
            ftype.  Can be one of the following types (.ply, .stl,
            .vtk)

        binary : bool, optional
            Writes the file as binary when True and ASCII when False.

        Notes
        -----
        Binary files write much faster than ASCII and have a smaller
        file size.
        """
        filename = os.path.abspath(os.path.expanduser(filename))
        file_mode = True
        # Check filetype
        ftype = filename[-3:]
        if ftype == 'ply':
            writer = vtk.vtkPLYWriter()
        elif ftype == 'vtp':
            writer = vtk.vtkXMLPolyDataWriter()
            file_mode = False
            if binary:
                writer.SetDataModeToBinary()
            else:
                writer.SetDataModeToAscii()
        elif ftype == 'stl':
            writer = vtk.vtkSTLWriter()
        elif ftype == 'vtk':
            writer = vtk.vtkPolyDataWriter()
        else:
            raise Exception('Filetype must be either "ply", "stl", or "vtk"')

        writer.SetFileName(filename)
        writer.SetInputData(self)
        if binary and file_mode:
            writer.SetFileTypeToBinary()
        elif file_mode:
            writer.SetFileTypeToASCII()
        writer.Write()

    def plot_curvature(self, curv_type='mean', **kwargs):
        """
        Plots curvature

        Parameters
        ----------
        curvtype : str, optional
            One of the following strings indicating curvature type

            - Mean
            - Gaussian
            - Maximum
            - Minimum

        **kwargs : optional
            See help(pyvista.plot)

        Returns
        -------
        cpos : list
            List of camera position, focal point, and view up
        """
        return self.plot(scalars=self.curvature(curv_type),
                         stitle='%s\nCurvature' % curv_type, **kwargs)

    def tri_filter(self, inplace=False):
        """
        Returns an all triangle mesh.  More complex polygons will be broken
        down into triangles.

        Parameters
        ----------
        inplace : bool, optional
            Updates mesh in-place while returning nothing.

        Returns
        -------
        mesh : pyvista.PolyData
            Mesh containing only triangles.  None when inplace=True

        """
        trifilter = vtk.vtkTriangleFilter()
        trifilter.SetInputData(self)
        trifilter.PassVertsOff()
        trifilter.PassLinesOff()
        trifilter.Update()

        mesh = _get_output(trifilter)
        if inplace:
            self.overwrite(mesh)
        else:
            return mesh

    def smooth(self, n_iter=20, convergence=0.0, edge_angle=15, feature_angle=45,
               boundary_smoothing=True, feature_smoothing=False, inplace=False):
        """Adjust point coordinates using Laplacian smoothing.
        The effect is to "relax" the mesh, making the cells better shaped and
        the vertices more evenly distributed.

        Parameters
        ----------
        n_iter : int
            Number of iterations for Laplacian smoothing,

        convergence : float, optional
            Convergence criterion for the iteration process. Smaller numbers
            result in more smoothing iterations. Range from (0 to 1).

        edge_angle : float, optional
            Edge angle to control smoothing along edges (either interior or boundary).

        feature_angle : float, optional
            Feature angle for sharp edge identification.

        boundary_smoothing : bool, optional
            Boolean flag to control smoothing of boundary edges.

        feature_smoothing : bool, optional
            Boolean flag to control smoothing of feature edges.

        inplace : bool, optional
            Updates mesh in-place while returning nothing.

        Returns
        -------
        mesh : pyvista.PolyData
            Decimated mesh. None when inplace=True.

        """
        alg = vtk.vtkSmoothPolyDataFilter()
        alg.SetInputData(self)
        alg.SetNumberOfIterations(n_iter)
        alg.SetConvergence(convergence)
        alg.SetFeatureEdgeSmoothing(feature_smoothing)
        alg.SetFeatureAngle(feature_angle)
        alg.SetEdgeAngle(edge_angle)
        alg.SetBoundarySmoothing(boundary_smoothing)
        alg.Update()

        mesh = _get_output(alg)
        if inplace:
            self.overwrite(mesh)
        else:
            return mesh

    def decimate_pro(self, reduction, feature_angle=45.0, split_angle=75.0, splitting=True,
                     pre_split_mesh=False, preserve_topology=False, inplace=False):
        """Reduce the number of triangles in a triangular mesh, forming a good
        approximation to the original geometry. Based on the algorithm originally
        described in "Decimation of Triangle Meshes", Proc Siggraph 92.

        Parameters
        ----------
        reduction : float
            Reduction factor. A value of 0.9 will leave 10 % of the original number
            of vertices.

        feature_angle : float, optional
            Angle used to define what an edge is (i.e., if the surface normal between
            two adjacent triangles is >= feature_angle, an edge exists).

        split_angle : float, optional
            Angle used to control the splitting of the mesh. A split line exists
            when the surface normals between two edge connected triangles are >= split_angle.

        splitting : bool, optional
            Controls the splitting of the mesh at corners, along edges, at non-manifold
            points, or anywhere else a split is required. Turning splitting off
            will better preserve the original topology of the mesh, but may not
            necessarily give the exact requested decimation.

        pre_split_mesh : bool, optional
            Separates the mesh into semi-planar patches, which are disconnected
            from each other. This can give superior results in some cases. If pre_split_mesh
            is set to True, the mesh is split with the specified split_angle. Otherwise
            mesh splitting is deferred as long as possible.

        preserve_topology : bool, optional
            Controls topology preservation. If on, mesh splitting and hole elimination
            will not occur. This may limit the maximum reduction that may be achieved.

        inplace : bool, optional
            Updates mesh in-place while returning nothing.

        Returns
        -------
        mesh : pyvista.PolyData
            Decimated mesh. None when inplace=True.

        """

        alg = vtk.vtkDecimatePro()
        alg.SetInputData(self)
        alg.SetTargetReduction(reduction)
        alg.SetPreserveTopology(preserve_topology)
        alg.SetFeatureAngle(feature_angle)
        alg.SetSplitting(splitting)
        alg.SetSplitAngle(split_angle)
        alg.SetPreSplitMesh(pre_split_mesh)
        alg.Update()

        mesh = _get_output(alg)
        if inplace:
            self.overwrite(mesh)
        else:
            return mesh

    def tube(self, radius=None, scalars=None, capping=True, n_sides=20,
             radius_factor=10, preference='point', inplace=False):
        """Generate a tube around each input line. The radius of the tube can be
        set to linearly vary with a scalar value.

        Parameters
        ----------
        radius : float
            Minimum tube radius (minimum because the tube radius may vary).

        scalars : str, optional
            Scalar array by which the radius varies

        capping : bool
            Turn on/off whether to cap the ends with polygons. Default True.

        n_sides : int
            Set the number of sides for the tube. Minimum of 3.

        radius_factor : float
            Maximum tube radius in terms of a multiple of the minimum radius.

        preference : str
            The field preference when searching for the scalar array by name

        inplace : bool, optional
            Updates mesh in-place while returning nothing.

        Returns
        -------
        mesh : pyvista.PolyData
            Tube-filtered mesh. None when inplace=True.

        """
        if n_sides < 3:
            n_sides = 3
        tube = vtk.vtkTubeFilter()
        tube.SetInputDataObject(self)
        # User Defined Parameters
        tube.SetCapping(capping)
        if radius is not None:
            tube.SetRadius(radius)
        tube.SetNumberOfSides(n_sides)
        tube.SetRadiusFactor(radius_factor)
        # Check if scalar array given
        if scalars is not None:
            if not isinstance(scalars, str):
                raise TypeError('Scalar array must be given as a string name')
            _, field = self.get_scalar(scalars, preference=preference, info=True)
            # args: (idx, port, connection, field, name)
            tube.SetInputArrayToProcess(0, 0, 0, field, scalars)
            tube.SetVaryRadiusToVaryRadiusByScalar()
        # Apply the filter
        tube.Update()

        mesh = _get_output(tube)
        if inplace:
            self.overwrite(mesh)
        else:
            return mesh

    def subdivide(self, nsub, subfilter='linear', inplace=False):
        """
        Increase the number of triangles in a single, connected triangular
        mesh.

        Uses one of the following vtk subdivision filters to subdivide a mesh.
        vtkButterflySubdivisionFilter
        vtkLoopSubdivisionFilter
        vtkLinearSubdivisionFilter

        Linear subdivision results in the fastest mesh subdivision, but it
        does not smooth mesh edges, but rather splits each triangle into 4
        smaller triangles.

        Butterfly and loop subdivision perform smoothing when dividing, and may
        introduce artifacts into the mesh when dividing.

        Subdivision filter appears to fail for multiple part meshes.  Should
        be one single mesh.

        Parameters
        ----------
        nsub : int
            Number of subdivisions.  Each subdivision creates 4 new triangles,
            so the number of resulting triangles is nface*4**nsub where nface
            is the current number of faces.

        subfilter : string, optional
            Can be one of the following: 'butterfly', 'loop', 'linear'

        inplace : bool, optional
            Updates mesh in-place while returning nothing.

        Returns
        -------
        mesh : Polydata object
            pyvista polydata object.  None when inplace=True

        Examples
        --------
        >>> from pyvista import examples
        >>> import pyvista
        >>> mesh = pyvista.PolyData(examples.planefile)
        >>> submesh = mesh.subdivide(1, 'loop') # doctest:+SKIP

        alternatively, update mesh in-place

        >>> mesh.subdivide(1, 'loop', inplace=True) # doctest:+SKIP
        """
        subfilter = subfilter.lower()
        if subfilter == 'linear':
            sfilter = vtk.vtkLinearSubdivisionFilter()
        elif subfilter == 'butterfly':
            sfilter = vtk.vtkButterflySubdivisionFilter()
        elif subfilter == 'loop':
            sfilter = vtk.vtkLoopSubdivisionFilter()
        else:
            raise Exception("Subdivision filter must be one of the following: " +
                            "'butterfly', 'loop', or 'linear'")

        # Subdivide
        sfilter.SetNumberOfSubdivisions(nsub)
        sfilter.SetInputData(self)
        sfilter.Update()

        submesh = _get_output(sfilter)
        if inplace:
            self.overwrite(submesh)
        else:
            return submesh

    def extract_edges(self, feature_angle=30, boundary_edges=True,
                     non_manifold_edges=True, feature_edges=True,
                     manifold_edges=True, inplace=False):
        """
        Extracts edges from a surface.  From vtk documentation, the edges are
        one of the following

            1) boundary (used by one polygon) or a line cell
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

        inplace : bool, optional
            Return new mesh or overwrite input.

        Returns
        -------
        edges : pyvista.vtkPolyData
            Extracted edges. None if inplace=True.

        """
        featureEdges = vtk.vtkFeatureEdges()
        featureEdges.SetInputData(self)
        featureEdges.SetFeatureAngle(feature_angle)
        featureEdges.SetManifoldEdges(manifold_edges)
        featureEdges.SetNonManifoldEdges(non_manifold_edges)
        featureEdges.SetBoundaryEdges(boundary_edges)
        featureEdges.SetFeatureEdges(feature_edges)
        featureEdges.SetColoring(False)
        featureEdges.Update()

        mesh = _get_output(featureEdges)
        if inplace:
            self.overwrite(mesh)
        else:
            return mesh

    def decimate(self, target_reduction, volume_preservation=False,
                 attribute_error=False, scalars=True, vectors=True,
                 normals=False, tcoords=True, tensors=True, scalars_weight=0.1,
                 vectors_weight=0.1, normals_weight=0.1, tcoords_weight=0.1,
                 tensors_weight=0.1, inplace=False):
        """
        Reduces the number of triangles in a triangular mesh using
        vtkQuadricDecimation.

        Parameters
        ----------
        mesh : vtk.PolyData
            Mesh to decimate

        target_reduction : float
            Fraction of the original mesh to remove.
            TargetReduction is set to 0.9, this filter will try to reduce
            the data set to 10% of its original size and will remove 90%
            of the input triangles.

        volume_preservation : bool, optional
            Decide whether to activate volume preservation which greatly reduces
            errors in triangle normal direction. If off, volume preservation is
            disabled and if AttributeErrorMetric is active, these errors can be
            large. Defaults to False.

        attribute_error : bool, optional
            Decide whether to include data attributes in the error metric. If
            off, then only geometric error is used to control the decimation.
            Defaults to False.

        scalars : bool, optional
            If attribute errors are to be included in the metric (i.e.,
            AttributeErrorMetric is on), then the following flags control which
            attributes are to be included in the error calculation. Defaults to
            True.

        vectors : bool, optional
            See scalars parameter. Defaults to True.

        normals : bool, optional
            See scalars parameter. Defaults to False.

        tcoords : bool, optional
            See scalars parameter. Defaults to True.

        tensors : bool, optional
            See scalars parameter. Defaults to True.

        scalars_weight : float, optional
            The scaling weight contribution of the scalar attribute. These
            values are used to weight the contribution of the attributes towards
            the error metric. Defaults to 0.1.

        vectors_weight : float, optional
            See scalars weight parameter. Defaults to 0.1.

        normals_weight : float, optional
            See scalars weight parameter. Defaults to 0.1.

        tcoords_weight : float, optional
            See scalars weight parameter. Defaults to 0.1.

        tensors_weight : float, optional
            See scalars weight parameter. Defaults to 0.1.

        inplace : bool, optional
            Updates mesh in-place while returning nothing.

        Returns
        -------
        outmesh : pyvista.PolyData
            Decimated mesh.  None when inplace=True.

        """
        # create decimation filter
        decimate = vtk.vtkQuadricDecimation()  # vtkDecimatePro as well

        decimate.SetVolumePreservation(volume_preservation)
        decimate.SetAttributeErrorMetric(attribute_error)
        decimate.SetScalarsAttribute(scalars)
        decimate.SetVectorsAttribute(vectors)
        decimate.SetNormalsAttribute(normals)
        decimate.SetTCoordsAttribute(tcoords)
        decimate.SetTensorsAttribute(tensors)
        decimate.SetScalarsWeight(scalars_weight)
        decimate.SetVectorsWeight(vectors_weight)
        decimate.SetNormalsWeight(normals_weight)
        decimate.SetTCoordsWeight(tcoords_weight)
        decimate.SetTensorsWeight(tensors_weight)
        decimate.SetTargetReduction(target_reduction)

        decimate.SetInputData(self)
        decimate.Update()

        mesh = _get_output(decimate)
        if inplace:
            self.overwrite(mesh)
        else:
            return mesh

    def center_of_mass(self, scalars_weight=False):
        """
        Returns the coordinates for the center of mass of the mesh.

        Parameters
        ----------
        scalars_weight : bool, optional
            Flag for using the mesh scalars as weights. Defaults to False.

        Return
        ------
        center : np.ndarray, float
            Coordinates for the center of mass.
        """
        comfilter = vtk.vtkCenterOfMass()
        comfilter.SetInputData(self)
        comfilter.SetUseScalarsAsWeights(scalars_weight)
        comfilter.Update()
        return np.array(comfilter.GetCenter())

    def compute_normals(self, cell_normals=True, point_normals=True, split_vertices=False,
                        flip_normals=False, consistent_normals=True, auto_orient_normals=False,
                        non_manifold_traversal=True, feature_angle=30.0, inplace=False):
        """
        Compute point and/or cell normals for a mesh.

        The filter can reorder polygons to insure consistent orientation across
        polygon neighbors. Sharp edges can be split and points duplicated
        with separate normals to give crisp (rendered) surface definition. It is
        also possible to globally flip the normal orientation.

        The algorithm works by determining normals for each polygon and then
        averaging them at shared points. When sharp edges are present, the edges
        are split and new points generated to prevent blurry edges (due to
        Gouraud shading).

        Parameters
        ----------
        cell_normals : bool, optional
            Calculation of cell normals. Defaults to True.

        point_normals : bool, optional
            Calculation of point normals. Defaults to True.

        split_vertices : bool, optional
            Splitting of sharp edges. Defaults to False.

        flip_normals : bool, optional
            Set global flipping of normal orientation. Flipping modifies both
            the normal direction and the order of a cell's points. Defaults to
            False.

        consistent_normals : bool, optional
            Enforcement of consistent polygon ordering. Defaults to True.

        auto_orient_normals : bool, optional
            Turn on/off the automatic determination of correct normal
            orientation. NOTE: This assumes a completely closed surface (i.e. no
            boundary edges) and no non-manifold edges. If these constraints do
            not hold, all bets are off. This option adds some computational
            complexity, and is useful if you don't want to have to inspect the
            rendered image to determine whether to turn on the FlipNormals flag.
            However, this flag can work with the FlipNormals flag, and if both
            are set, all the normals in the output will point "inward". Defaults
            to False.

        non_manifold_traversal : bool, optional
            Turn on/off traversal across non-manifold edges. Changing this may
            prevent problems where the consistency of polygonal ordering is
            corrupted due to topological loops. Defaults to True.

        feature_angle : float, optional
            The angle that defines a sharp edge. If the difference in angle
            across neighboring polygons is greater than this value, the shared
            edge is considered "sharp". Defaults to 30.0.

        inplace : bool, optional
            Updates mesh in-place while returning nothing. Defaults to False.

        Returns
        -------
        mesh : pyvista.PolyData
            Updated mesh with cell and point normals if inplace=False

        Notes
        -----
        Previous arrays named "Normals" will be overwritten.

        Normals are computed only for polygons and triangle strips. Normals are
        not computed for lines or vertices.

        Triangle strips are broken up into triangle polygons. You may want to
        restrip the triangles.

        May be easier to run mesh.point_normals or mesh.cell_normals

        """
        normal = vtk.vtkPolyDataNormals()
        normal.SetComputeCellNormals(cell_normals)
        normal.SetComputePointNormals(point_normals)
        normal.SetSplitting(split_vertices)
        normal.SetFlipNormals(flip_normals)
        normal.SetConsistency(consistent_normals)
        normal.SetAutoOrientNormals(auto_orient_normals)
        normal.SetNonManifoldTraversal(non_manifold_traversal)
        normal.SetFeatureAngle(feature_angle)
        normal.SetInputData(self)
        normal.Update()

        mesh = _get_output(normal)
        if point_normals:
            mesh.GetPointData().SetActiveNormals('Normals')
        if cell_normals:
            mesh.GetCellData().SetActiveNormals('Normals')


        if inplace:
            self.overwrite(mesh)
        else:
            return mesh

    @property
    def point_normals(self):
        """ Point normals """
        mesh = self.compute_normals(cell_normals=False, inplace=False)
        return mesh.point_arrays['Normals']

    @property
    def cell_normals(self):
        """ Cell normals  """
        mesh = self.compute_normals(point_normals=False, inplace=False)
        return mesh.cell_arrays['Normals']

    @property
    def face_normals(self):
        """ Cell normals  """
        return self.cell_normals

    def clip_with_plane(self, origin, normal, value=0, inplace=False):
        """
        Clip a pyvista.PolyData or vtk.vtkPolyData with a plane.

        Can be used to open a mesh which has been closed along a well-defined
        plane.

        Parameters
        ----------
        origin : numpy.ndarray
            3D point through which plane passes. Defines the plane together with
            normal parameter.

        normal : numpy.ndarray
            3D vector defining plane normal.

        value : float, optional
            Scalar clipping value. The default value is 0.0.

        inplace : bool, optional
            Updates mesh in-place while returning nothing.

        Returns
        -------
        mesh : pyvista.PolyData
            Updated mesh with cell and point normals if inplace=False. Otherwise None.

        Notes
        -----
        Not guaranteed to produce a manifold output.

        """

        plane = vtk.vtkPlane()
        plane.SetOrigin(origin)
        plane.SetNormal(normal)
        plane.Modified()

        clip = vtk.vtkClipPolyData()
        clip.SetValue(value)
        clip.GenerateClippedOutputOn()
        clip.SetClipFunction(plane)

        clip.SetInputData(self)
        clip.Update()

        mesh = _get_output(clip)
        if inplace:
            self.overwrite(mesh)
        else:
            return mesh

    def extract_largest(self, inplace=False):
        """
        Extract largest connected set in mesh.

        Can be used to reduce residues obtained when generating an isosurface.
        Works only if residues are not connected (share at least one point with)
        the main component of the image.

        Parameters
        ----------
        inplace : bool, optional
            Updates mesh in-place while returning nothing.

        Returns
        -------
        mesh : pyvista.PolyData
            Largest connected set in mesh

        """
        mesh =  self.connectivity(largest=True)
        if inplace:
            self.overwrite(mesh)
        else:
            return mesh



    def fill_holes(self, hole_size, inplace=False):  # pragma: no cover
        """
        Fill holes in a pyvista.PolyData or vtk.vtkPolyData object.

        Holes are identified by locating boundary edges, linking them together
        into loops, and then triangulating the resulting loops. Note that you
        can specify an approximate limit to the size of the hole that can be
        filled.

        Parameters
        ----------
        hole_size : float
            Specifies the maximum hole size to fill. This is represented as a
            radius to the bounding circumsphere containing the hole. Note that
            this is an approximate area; the actual area cannot be computed
            without first triangulating the hole.

        inplace : bool, optional
            Return new mesh or overwrite input.

        Returns
        -------
        mesh : pyvista.PolyData
            Mesh with holes filled.  None when inplace=True

        """
        logging.warning('pyvista.pointset.PolyData.fill_holes is known to segfault. ' +
                        'Use at your own risk')
        fill = vtk.vtkFillHolesFilter()
        fill.SetHoleSize(hole_size)
        fill.SetInputData(self)
        fill.Update()

        mesh = _get_output(fill)
        if inplace:
            self.overwrite(mesh)
        else:
            return mesh

    def clean(self, point_merging=True, merge_tol=None, lines_to_points=True,
              polys_to_lines=True, strips_to_polys=True, inplace=False):
        """
        Cleans mesh by merging duplicate points, remove unused
        points, and/or remove degenerate cells.

        Parameters
        ----------
        point_merging : bool, optional
            Enables point merging.  On by default.

        merge_tol : float, optional
            Set merging tolarance.  When enabled merging is set to
            absolute distance

        lines_to_points : bool, optional
            Turn on/off conversion of degenerate lines to points.  Enabled by
            default.

        polys_to_lines : bool, optional
            Turn on/off conversion of degenerate polys to lines.  Enabled by
            default.

        strips_to_polys : bool, optional
            Turn on/off conversion of degenerate strips to polys.

        inplace : bool, optional
            Updates mesh in-place while returning nothing.  Default True.

        Returns
        -------
        mesh : pyvista.PolyData
            Cleaned mesh.  None when inplace=True
        """
        clean = vtk.vtkCleanPolyData()
        clean.SetConvertLinesToPoints(lines_to_points)
        clean.SetConvertPolysToLines(polys_to_lines)
        clean.SetConvertStripsToPolys(strips_to_polys)
        if merge_tol:
            clean.ToleranceIsAbsoluteOn()
            clean.SetAbsoluteTolerance(merge_tol)
        clean.SetInputData(self)
        clean.Update()

        output = _get_output(clean)

        # Check output so no segfaults occur
        if output.n_points < 1:
            raise AssertionError('Clean tolerance is too high. Empty mesh returned.')

        if inplace:
            self.overwrite(output)
        else:
            return output

    @property
    def area(self):
        """
        Mesh surface area

        Returns
        -------
        area : float
            Total area of the mesh.

        """
        mprop = vtk.vtkMassProperties()
        mprop.SetInputData(self)
        return mprop.GetSurfaceArea()

    @property
    def volume(self):
        """
        Mesh volume - will throw a VTK error/warning if not a closed surface

        Returns
        -------
        volume : float
            Total volume of the mesh.

        """
        mprop = vtk.vtkMassProperties()
        mprop.SetInputData(self.tri_filter())
        return mprop.GetVolume()

    @property
    def obbTree(self):
        """obbTree is an object to generate oriented bounding box (OBB)
        trees. An oriented bounding box is a bounding box that does not
        necessarily line up along coordinate axes. The OBB tree is a
        hierarchical tree structure of such boxes, where deeper levels of OBB
        confine smaller regions of space.
        """
        if not hasattr(self, '_obbTree'):
            self._obbTree = vtk.vtkOBBTree()
            self._obbTree.SetDataSet(self)
            self._obbTree.BuildLocator()

        return self._obbTree


    def geodesic(self, start_vertex, end_vertex, inplace=False):
        """
        Calculates the geodesic path betweeen two vertices using Dijkstra's
        algorithm.

        Parameters
        ----------
        start_vertex : int
            Vertex index indicating the start point of the geodesic segment.

        end_vertex : int
            Vertex index indicating the end point of the geodesic segment.

        Returns
        -------
        output : pyvista.PolyData
            PolyData object consisting of the line segment between the two given
            vertices.

        """
        if start_vertex < 0 or end_vertex > self.n_points - 1:
            raise IndexError('Invalid indices.')

        dijkstra = vtk.vtkDijkstraGraphGeodesicPath()
        dijkstra.SetInputData(self)
        dijkstra.SetStartVertex(start_vertex)
        dijkstra.SetEndVertex(end_vertex)
        dijkstra.Update()

        output = _get_output(dijkstra)

        if inplace:
            self.overwrite(output)
        else:
            return output


    def geodesic_distance(self, start_vertex, end_vertex):
        """
        Calculates the geodesic distance betweeen two vertices using Dijkstra's
        algorithm.

        Parameters
        ----------
        start_vertex : int
            Vertex index indicating the start point of the geodesic segment.

        end_vertex : int
            Vertex index indicating the end point of the geodesic segment.

        Returns
        -------
        length : float
            Length of the geodesic segment.

        """
        length = self.geodesic(start_vertex, end_vertex).GetLength()
        return length

    def ray_trace(self, origin, end_point, first_point=False, plot=False,
                  off_screen=False):
        """
        Performs a single ray trace calculation given a mesh and a line segment
        defined by an origin and end_point.

        Parameters
        ----------
        origin : np.ndarray or list
            Start of the line segment.

        end_point : np.ndarray or list
            End of the line segment.

        first_point : bool, optional
            Returns intersection of first point only.

        plot : bool, optional
            Plots ray trace results

        off_screen : bool, optional
            Plots off screen.  Used for unit testing.

        Returns
        -------
        intersection_points : np.ndarray
            Location of the intersection points.  Empty array if no
            intersections.

        intersection_cells : np.ndarray
            Indices of the intersection cells.  Empty array if no
            intersections.

        """
        points = vtk.vtkPoints()
        cell_ids = vtk.vtkIdList()
        code = self.obbTree.IntersectWithLine(np.array(origin),
                                              np.array(end_point),
                                              points, cell_ids)

        intersection_points = vtk_to_numpy(points.GetData())
        if first_point and intersection_points.shape[0] >= 1:
            intersection_points = intersection_points[0]

        intersection_cells = []
        if intersection_points.any():
            if first_point:
                ncells = 1
            else:
                ncells = cell_ids.GetNumberOfIds()
            for i in range(ncells):
                intersection_cells.append(cell_ids.GetId(i))
        intersection_cells = np.array(intersection_cells)

        if plot:
            plotter = pyvista.Plotter(off_screen=off_screen)
            plotter.add_mesh(self, label='Test Mesh')
            segment = np.array([origin, end_point])
            plotter.add_lines(segment, 'b', label='Ray Segment')
            plotter.add_mesh(intersection_points, 'r', point_size=10,
                             label='Intersection Points')
            plotter.add_legend()
            plotter.add_axes()
            plotter.show()

        return intersection_points, intersection_cells

    def plot_boundaries(self, **kwargs):
        """ Plots boundaries of a mesh """
        edges = self.extract_edges()

        plotter = pyvista.Plotter(off_screen=kwargs.pop('off_screen', False),
                               notebook=kwargs.pop('notebook', None))
        plotter.add_mesh(edges, 'r', style='wireframe', legend='Edges')
        plotter.add_mesh(self, legend='Mesh', **kwargs)
        return plotter.show()

    def plot_normals(self, show_mesh=True, mag=1.0, flip=False,
                     use_every=1, **kwargs):
        """
        Plot the point normals of a mesh.
        """
        plotter = pyvista.Plotter(off_screen=kwargs.pop('off_screen', False),
                               notebook=kwargs.pop('notebook', None))
        if show_mesh:
            plotter.add_mesh(self, **kwargs)

        normals = self.point_normals
        if flip:
            normals *= -1
        plotter.add_arrows(self.points[::use_every],
                           normals[::use_every], mag=mag)
        return plotter.show()

    def remove_points(self, remove, mode='any', keep_scalars=True, inplace=False):
        """
        Rebuild a mesh by removing points.  Only valid for
        all-triangle meshes.

        Parameters
        ----------
        remove : np.ndarray
            If remove is a bool array, points that are True will be
            removed.  Otherwise, it is treated as a list of indices.

        mode : str, optional
            When 'all', only faces containing all points flagged for
            removal will be removed.  Default 'all'

        keep_scalars : bool, optional
            When True, point and cell scalars will be passed on to the
            new mesh.

        inplace : bool, optional
            Updates mesh in-place while returning nothing.

        Returns
        -------
        mesh : pyvista.PolyData
            Mesh without the points flagged for removal.  Not returned
            when inplace=False.

        ridx : np.ndarray
            Indices of new points relative to the original mesh.  Not
            returned when inplace=False.

        """
        if isinstance(remove, list):
            remove = np.asarray(remove)

        if remove.dtype == np.bool:
            if remove.size != self.n_points:
                raise AssertionError('Mask different size than n_points')
            remove_mask = remove
        else:
            remove_mask = np.zeros(self.n_points, np.bool)
            remove_mask[remove] = True

        try:
            f = self.faces.reshape(-1, 4)[:, 1:]
        except:
            raise Exception('Mesh must consist of only triangles')

        vmask = remove_mask.take(f)
        if mode == 'all':
            fmask = ~(vmask).all(1)
        else:
            fmask = ~(vmask).any(1)

        # Regenerate face and point arrays
        uni = np.unique(f.compress(fmask, 0), return_inverse=True)
        new_points = self.points.take(uni[0], 0)

        nfaces = fmask.sum()
        faces = np.empty((nfaces, 4), dtype=pyvista.ID_TYPE)
        faces[:, 0] = 3
        faces[:, 1:] = np.reshape(uni[1], (nfaces, 3))

        newmesh = PolyData(new_points, faces, deep=True)
        ridx = uni[0]

        # Add scalars back to mesh if requested
        if keep_scalars:
            for key in self.point_arrays:
                newmesh.point_arrays[key] = self.point_arrays[key][ridx]

            for key in self.cell_arrays:
                try:
                    newmesh.cell_arrays[key] = self.cell_arrays[key][fmask]
                except:
                    log.warning('Unable to pass cell key %s onto reduced mesh' %
                                key)

        # Return vtk surface and reverse indexing array
        if inplace:
            self.overwrite(newmesh)
        else:
            return newmesh, ridx

    def flip_normals(self):
        """
        Flip normals of a triangular mesh by reversing the point ordering.

        """
        if self.faces.size % 4:
            raise Exception('Can only flip normals on an all triangular mesh')

        f = self.faces.reshape((-1, 4))
        f[:, 1:] = f[:, 1:][:, ::-1]

    def delaunay_2d(self, tol=1e-05, alpha=0.0, offset=1.0, bound=False, inplace=False):
        """Apply a delaunay 2D filter along the best fitting plane"""
        alg = vtk.vtkDelaunay2D()
        alg.SetProjectionPlaneMode(vtk.VTK_BEST_FITTING_PLANE)
        alg.SetInputDataObject(self)
        alg.SetTolerance(tol)
        alg.SetAlpha(alpha)
        alg.SetOffset(offset)
        alg.SetBoundingTriangulation(bound)
        alg.Update()

        mesh = _get_output(alg)
        if inplace:
            self.overwrite(mesh)
        else:
            return mesh

    def delauney_2d(self):
        """DEPRECATED. Please see :func:`pyvista.PolyData.delaunay_2d`"""
        raise AttributeError('`delauney_2d` is deprecated because we made a '\
                             'spelling mistake. Please use `delaunay_2d`.')


class PointGrid(pyvista.Common):
    """ Class in common with structured and unstructured grids """

    def __new__(cls, *args, **kwargs):
        if cls is PointGrid:
            raise TypeError("pyvista.PointGrid is an abstract class and may not be instantiated.")
        return object.__new__(cls, *args, **kwargs)

    def __init__(self, *args, **kwargs):
        super(PointGrid, self).__init__()

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
            Optional keyword arguments.  See help(pyvista.plot)

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

    def extract_surface(self, pass_pointid=True, pass_cellid=True, inplace=False):
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

        inplace : bool, optional
            Return new mesh or overwrite input.

        Returns
        -------
        extsurf : pyvista.PolyData
            Surface mesh of the grid
        """
        surf_filter = vtk.vtkDataSetSurfaceFilter()
        surf_filter.SetInputData(self)
        if pass_pointid:
            surf_filter.PassThroughCellIdsOn()
        if pass_cellid:
            surf_filter.PassThroughPointIdsOn()
        surf_filter.Update()

        mesh = _get_output(surf_filter)
        if inplace:
            self.overwrite(mesh)
        else:
            return mesh

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
                      manifold_edges=True, inplace=False):
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

        inplace : bool, optional
            Return new mesh or overwrite input.

        Returns
        -------
        edges : pyvista.vtkPolyData
            Extracted edges

        """
        surf = self.extract_surface()
        return surf.extract_edges(feature_angle, boundary_edges,
                                  non_manifold_edges, feature_edges,
                                  manifold_edges, inplace=inplace)


class UnstructuredGrid(vtkUnstructuredGrid, PointGrid):
    """
    Extends the functionality of a vtk.vtkUnstructuredGrid object.

    Can be initialized by the following:

    - Creating an empty grid
    - From a vtk.vtkPolyData object
    - From cell, offset, and node arrays
    - From a file

    Examples
    --------
    >>> import pyvista
    >>> from pyvista import examples
    >>> import vtk

    >>> # Create an empy grid
    >>> grid = pyvista.UnstructuredGrid()

    >>> # Copy a vtkUnstructuredGrid
    >>> vtkgrid = vtk.vtkUnstructuredGrid()
    >>> grid = pyvista.UnstructuredGrid(vtkgrid)  # Initialize from a vtkUnstructuredGrid

    >>> # from arrays
    >>> #grid = pyvista.UnstructuredGrid(offset, cells, cell_type, nodes, deep=True)

    >>> # From a string filename
    >>> grid = pyvista.UnstructuredGrid(examples.hexbeamfile)

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


    def __repr__(self):
        return pyvista.Common.__repr__(self)


    def __str__(self):
        return pyvista.Common.__str__(self)


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
        >>> import numpy
        >>> import vtk
        >>> import pyvista
        >>> offset = np.array([0, 9])
        >>> cells = np.array([8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 9, 10, 11, 12, 13, 14, 15])
        >>> cell_type = np.array([vtk.VTK_HEXAHEDRON, vtk.VTK_HEXAHEDRON], np.int8)

        >>> cell1 = np.array([[0, 0, 0],
        ...                   [1, 0, 0],
        ...                   [1, 1, 0],
        ...                   [0, 1, 0],
        ...                   [0, 0, 1],
        ...                   [1, 0, 1],
        ...                   [1, 1, 1],
        ...                   [0, 1, 1]])

        >>> cell2 = np.array([[0, 0, 2],
        ...                   [1, 0, 2],
        ...                   [1, 1, 2],
        ...                   [0, 1, 2],
        ...                   [0, 0, 3],
        ...                   [1, 0, 3],
        ...                   [1, 1, 3],
        ...                   [0, 1, 3]])

        >>> points = np.vstack((cell1, cell2))

        >>> grid = pyvista.UnstructuredGrid(offset, cells, cell_type, points)

        """

        if offset.dtype != pyvista.ID_TYPE:
            offset = offset.astype(pyvista.ID_TYPE)

        if cells.dtype != pyvista.ID_TYPE:
            cells = cells.astype(pyvista.ID_TYPE)

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
        points = pyvista.vtk_points(points, deep=deep)

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
        filename = os.path.abspath(os.path.expanduser(filename))
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
        filename = os.path.abspath(os.path.expanduser(filename))
        # Use legacy writer if vtk is in filename
        if '.vtk' in filename:
            writer = vtk.vtkUnstructuredGridWriter()
            if binary:
                writer.SetFileTypeToBinary()
            else:
                writer.SetFileTypeToASCII()
        elif '.vtu' in filename:
            writer = vtk.vtkXMLUnstructuredGridWriter()
            if binary:
                writer.SetDataModeToBinary()
            else:
                writer.SetDataModeToAscii()
        else:
            raise Exception('Extension should be either ".vtu" or ".vtk"')

        writer.SetFileName(filename)
        writer.SetInputData(self)
        return writer.Write()

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
        linear cells.  Converts the following cell types to their
        linear equivalents.

        - VTK_QUADRATIC_TETRA      --> VTK_TETRA
        - VTK_QUADRATIC_PYRAMID    --> VTK_PYRAMID
        - VTK_QUADRATIC_WEDGE      --> VTK_WEDGE
        - VTK_QUADRATIC_HEXAHEDRON --> VTK_HEXAHEDRON

        Parameters
        ----------
        deep : bool
            When True, makes a copy of the points array.  Default
            False.  Cells and cell types are always copied.

        Returns
        -------
        grid : pyvista.UnstructuredGrid
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

        # track quad mask for later
        quad_quad_mask = celltype == VTK_QUADRATIC_QUAD
        celltype[quad_quad_mask] = VTK_QUAD

        quad_tri_mask = celltype == VTK_QUADRATIC_TRIANGLE
        celltype[quad_tri_mask] = VTK_TRIANGLE

        vtk_offset = self.GetCellLocationsArray()
        cells = vtk.vtkCellArray()
        cells.DeepCopy(self.GetCells())
        lgrid.SetCells(vtk_cell_type, vtk_offset, cells)

        # fixing bug with display of quad cells
        if np.any(quad_quad_mask):
            quad_offset = lgrid.offset[quad_quad_mask]
            base_point = lgrid.cells[quad_offset + 1]
            lgrid.cells[quad_offset + 5] = base_point
            lgrid.cells[quad_offset + 6] = base_point
            lgrid.cells[quad_offset + 7] = base_point
            lgrid.cells[quad_offset + 8] = base_point

        if np.any(quad_tri_mask):
            tri_offset = lgrid.offset[quad_tri_mask]
            base_point = lgrid.cells[tri_offset + 1]
            lgrid.cells[tri_offset + 4] = base_point
            lgrid.cells[tri_offset + 5] = base_point
            lgrid.cells[tri_offset + 6] = base_point

        return lgrid

    @property
    def celltypes(self):
        """Get the cell types array"""
        return vtk_to_numpy(self.GetCellTypesArray())

    @property
    def offset(self):
        """Get Cell Locations Array"""
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
        subgrid : pyvista.UnstructuredGrid
            Subselected grid

        """
        if not isinstance(ind, np.ndarray):
            ind = np.array(ind, np.ndarray)

        if ind.dtype == np.bool:
            ind = ind.nonzero()[0].astype(pyvista.ID_TYPE)

        if ind.dtype != pyvista.ID_TYPE:
            ind = ind.astype(pyvista.ID_TYPE)

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
        extract_sel = vtk.vtkExtractSelection()
        extract_sel.SetInputData(0, self)
        extract_sel.SetInputData(1, selection)
        extract_sel.Update()
        subgrid = _get_output(extract_sel)

        # extracts only in float32
        if self.points.dtype is not np.dtype('float32'):
            ind = subgrid.point_arrays['vtkOriginalPointIds']
            subgrid.points = self.points[ind]

        return subgrid

    def extract_selection_points(self, ind):
        """Returns a subset of the grid that contains the cells that
        contain any of the point indices.

        Parameters
        ----------
        ind : np.ndarray, list, or iterable
            Numpy array of point indices to be extracted.

        Returns
        -------
        subgrid : pyvista.UnstructuredGrid
            Subselected grid.
        """
        try:
            ind = np.array(ind)
        except:
            raise Exception('indices must be either a mask, array, list, or iterable')

        # Convert to vtk indices
        if ind.dtype == np.bool:
            ind = ind.nonzero()[0]

        if ind.dtype != np.int64:
            ind = ind.astype(np.int64)
        vtk_ind = numpy_to_vtkIdTypeArray(ind, deep=True)

        # Create selection objects
        selectionNode = vtk.vtkSelectionNode()
        selectionNode.SetFieldType(vtk.vtkSelectionNode.POINT)
        selectionNode.SetContentType(vtk.vtkSelectionNode.INDICES)
        selectionNode.SetSelectionList(vtk_ind)
        selectionNode.GetProperties().Set(vtk.vtkSelectionNode.CONTAINING_CELLS(), 1)

        selection = vtk.vtkSelection()
        selection.AddNode(selectionNode)

        # extract
        extract_sel = vtk.vtkExtractSelection()
        extract_sel.SetInputData(0, self)
        extract_sel.SetInputData(1, selection)
        extract_sel.Update()
        return UnstructuredGrid(extract_sel.GetOutput())

    def merge(self, grid=None, merge_points=True, inplace=False,
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

        if isinstance(grid, pyvista.UnstructuredGrid):
            append_filter.AddInputData(grid)
        elif isinstance(grid, list):
            grids = grid
            for grid in grids:
                append_filter.AddInputData(grid)

        if main_has_priority:
            append_filter.AddInputData(self)

        append_filter.Update()
        merged = _get_output(append_filter)
        if inplace:
            self.DeepCopy(merged)
        else:
            return merged

    def delaunay_2d(self, tol=1e-05, alpha=0.0, offset=1.0, bound=False):
        """Apply a delaunay 2D filter along the best fitting plane. This
        extracts the grid's points and perfoms the triangulation on those alone.
        """
        return PolyData(self.points).delaunay_2d(tol=tol, alpha=alpha,
                                                 offset=offset, bound=bound)


class StructuredGrid(vtkStructuredGrid, PointGrid):
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
    >>> import pyvista
    >>> import vtk
    >>> import numpy as np

    >>> # Create empty grid
    >>> grid = pyvista.StructuredGrid()

    >>> # Initialize from a vtk.vtkStructuredGrid object
    >>> vtkgrid = vtk.vtkStructuredGrid()
    >>> grid = pyvista.StructuredGrid(vtkgrid)

    >>> # Create from NumPy arrays
    >>> xrng = np.arange(-10, 10, 2)
    >>> yrng = np.arange(-10, 10, 2)
    >>> zrng = np.arange(-10, 10, 2)
    >>> x, y, z = np.meshgrid(xrng, yrng, zrng)
    >>> grid = pyvista.StructuredGrid(x, y, z)


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


    def __repr__(self):
        return pyvista.Common.__repr__(self)


    def __str__(self):
        return pyvista.Common.__str__(self)


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
        self.SetPoints(pyvista.vtk_points(points))

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
        filename = os.path.abspath(os.path.expanduser(filename))
        # check file exists
        if not os.path.isfile(filename):
            raise Exception('{} does not exist'.format(filename))

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
        filename = os.path.abspath(os.path.expanduser(filename))
        # Use legacy writer if vtk is in filename
        if '.vtk' in filename:
            writer = vtk.vtkStructuredGridWriter()
            if binary:
                writer.SetFileTypeToBinary()
            else:
                writer.SetFileTypeToASCII()
        elif '.vts' in filename:
            writer = vtk.vtkXMLStructuredGridWriter()
            if binary:
                writer.SetDataModeToBinary()
            else:
                writer.SetDataModeToAscii()
        else:
            raise Exception('Extension should be either ".vts" (xml) or' +
                            '".vtk" (legacy)')
        # Write
        writer.SetFileName(filename)
        writer.SetInputData(self)
        writer.Write()

    @property
    def dimensions(self):
        """Returns a length 3 tuple of the grid's dimensions"""
        return list(self.GetDimensions())

    @dimensions.setter
    def dimensions(self, dims):
        """Sets the dataset dimensions. Pass a length three tuple of integers"""
        nx, ny, nz = dims[0], dims[1], dims[2]
        self.SetDimensions(nx, ny, nz)
        self.Modified()

    @property
    def x(self):
        """The X coordinates of all points"""
        return self.points[:, 0].reshape(self.dimensions, order='F')

    @property
    def y(self):
        """The Y coordinates of all points"""
        return self.points[:, 1].reshape(self.dimensions, order='F')

    @property
    def z(self):
        """The Z coordinates of all points"""
        return self.points[:, 2].reshape(self.dimensions, order='F')

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
