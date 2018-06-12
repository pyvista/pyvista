"""
Sub-classes for vtk.vtkPolyData
"""
import os
import numpy as np
import vtkInterface
import logging

log = logging.getLogger(__name__)
log.setLevel('CRITICAL')

# allows readthedocs to autodoc
try:
    import vtk
    from vtk import vtkPolyData
    from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtkIdTypeArray
    from vtk.util.numpy_support import numpy_to_vtk

except:
    # create dummy class when readthedocs can't import vtk
    class vtkPolyData(object):
        def __init__(self, *args, **kwargs):
            pass


class PolyData(vtkPolyData, vtkInterface.Common):
    """
    Extends the functionality of a vtk.vtkPolyData object

    Can be initialized in several ways:

    - Create an empty mesh
    - Initialize from a vtk.vtkPolyData
    - Initialize using cell, offset, and node numpy arrays (see MakeFromArrays
      for more details on this method)
    - Create from a file

    Examples
    --------
    >>> surf = PolyData()  # Create an empty mesh
    >>> surf = PolyData(vtkobj)  # Initialize from a vtk.vtkPolyData object
    >>> surf = PolyData(vertices, faces)  # initialize from vertices and face
    """

    def __init__(self, *args, **kwargs):
        super(PolyData, self).__init__()

        if not args:
            return
        elif len(args) == 1:
            if isinstance(args[0], vtk.vtkPolyData):
                self.ShallowCopy(args[0])
            elif isinstance(args[0], str):
                self.LoadFile(args[0])
        elif len(args) == 2:
            arg0_is_array = isinstance(args[0], np.ndarray)
            arg1_is_array = isinstance(args[1], np.ndarray)
            if arg0_is_array and arg1_is_array:
                if 'deep' in kwargs:
                    deep = kwargs['deep']
                else:
                    deep = True
                self.MakeFromArrays(args[0], args[1], deep)
        else:
            raise TypeError('Invalid input type')

    def LoadFile(self, filename):
        """
        Load a surface mesh from a mesh file.

        Mesh file may be an ASCII or binary ply, stl, g3d, or vtk mesh file.

        Parameters
        ----------
        filename : str
            Filename of mesh to be loaded.  File type is inferred from the
            extension of the filename

        Notes
        -----
        Binary files load much faster than ASCII.

        """
        # test if file exists
        if not os.path.isfile(filename):
            raise Exception('File {:s} does not exist'.format(filename))

        # Get extension
        fext = filename[-3:].lower()

        # Select reader
        if fext == 'ply':
            reader = vtk.vtkPLYReader()
        elif fext == 'stl':
            reader = vtk.vtkSTLReader()
        elif fext == 'g3d':  # Don't use vtk reader
            v, f = vtkInterface.ReadG3D(filename)
            v /= 25.4  # convert to inches
            self.MakeFromArrays(v, f)
        elif fext == 'vtk':
            reader = vtk.vtkPolyDataReader()
        else:
            raise Exception('Filetype must be either "ply", "stl", "g3d" ' +
                            'or "vtk"')

        # Load file
        reader.SetFileName(filename)
        reader.Update()
        self.ShallowCopy(reader.GetOutput())

        # sanity check
        try:
            self.points
        except:
            raise Exception('Cannot access points.  Empty or invalid file')
        try:
            self.faces
        except:
            raise Exception('Cannot access points.  Empty or invalid file')

    @property
    def faces(self):
        """ returns a pointer to the points as a numpy object """
        return vtk_to_numpy(self.GetPolys().GetData())

    @property
    def lines(self):
        """ returns a copy of the indices of the lines """
        lines = vtk_to_numpy(self.GetLines().GetData()).reshape((-1, 3))
        return np.ascontiguousarray(lines[:, 1:])

    def MakeFromArrays(self, vertices, faces, deep=True):
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
        >>> vertices = np.array([[0, 0, 0],
                                 [1, 0, 0],
                                 [1, 1, 0],
                                 [0, 1, 0],
                                 [0.5, 0.5, 1]])
        >>> faces = np.hstack([[4, 0, 1, 2, 3],
                               [3, 0, 1, 4],
                               [3, 1, 2, 4]])  # one square and two triangles
        >>> surf = PolyData(vertices, faces)

        """
        vtkpoints = vtk.vtkPoints()
        vtkpoints.SetData(numpy_to_vtk(vertices, deep=deep))
        self.SetPoints(vtkpoints)

        # Convert to a vtk array
        vtkcells = vtk.vtkCellArray()
        if faces.dtype != vtkInterface.ID_TYPE:
            faces = faces.astype(vtkInterface.ID_TYPE)

        # get number of faces
        if faces.ndim == 1:
            c = 0
            nfaces = 0
            while c < faces.size:
                c += faces[c] + 1
                nfaces += 1
        else:
            nfaces = faces.shape[0]

        idarr = numpy_to_vtkIdTypeArray(faces, deep=deep)
        vtkcells.SetCells(nfaces, idarr)
        self.SetPolys(vtkcells)

    def GetNumpyFaces(self, force_C_CONTIGUOUS=False, nocut=False, dtype=None):
        """
        Returns the faces from a polydata object as a numpy array.  Assumes a
        triangular mesh.  Array will be sized (-1, 3) unless nocut is True.

        Parameters
        ----------
        force_C_CONTIGUOUS : bool, optional
            Force array to be c contigious.  Default False.

        nocut : bool, optional
            When true, array will be shaped (-1, 4) with the first row
            containing only 3.

        dtype : np.dtype, optional
            Output data type.

        Returns
        -------
        f : np.ndarray
            Array containing triangle point indices.
        """
        f = vtk_to_numpy(self.GetPolys().GetData()).reshape((-1, 4))

        # remove triangle size padding
        if not nocut:
            f = f[:, 1:]

        if force_C_CONTIGUOUS:
            if dtype:
                f = np.ascontiguousarray(f, dtype)
            else:
                f = np.ascontiguousarray(f)
        elif dtype:
            if f.dtype != dtype:
                f = f.astype(dtype)

        return f

    def SetNumpyFaces(self, f):
        """
        Sets mesh polygons.  Assumes a triangular mesh.

        Parameters
        ----------
        f : np.ndarray
            Face indices.  Array must be (-1, 4)

        """
        # Check shape
        if f.ndim != 2:
            raise Exception('Faces should be a 2D array')
        elif f.shape[1] != 4:
            raise Exception(
                'First column should contain the number of points per face')

        vtkcells = vtk.vtkCellArray()
        vtkcells.SetCells(f.shape[0], numpy_to_vtkIdTypeArray(f, deep=True))
        self.SetPolys(vtkcells)

    def GetEdgeMask(self, angle):
        """
        Returns a mask of the points of a surface mesh that have a surface
        angle greater than angle

        Parameters
        ----------
        angle : float
            Angle to consider an edge.

        """
        featureEdges = vtk.vtkFeatureEdges()
        featureEdges.SetInputData(self)
        featureEdges.FeatureEdgesOn()
        featureEdges.BoundaryEdgesOff()
        featureEdges.NonManifoldEdgesOff()
        featureEdges.ManifoldEdgesOff()
        featureEdges.SetFeatureAngle(angle)
        featureEdges.Update()
        edges = featureEdges.GetOutput()
        origID = vtkInterface.GetPointScalars(edges, 'vtkOriginalPointIds')

        return np.in1d(self.GetPointScalars('vtkOriginalPointIds'),
                       origID,
                       assume_unique=True)

    def BooleanCut(self, cut, tolerance=1E-5):
        """
        Performs a Boolean cut using another mesh.

        Parameters
        ----------
        cut : vtkInterface.PolyData
            Mesh making the cut

        Returns
        -------
        mesh : vtkInterface.PolyData
            The cut mesh
        """
        bfilter = vtk.vtkBooleanOperationPolyDataFilter()
        bfilter.SetOperationToIntersection()
        bfilter.SetInputData(1, cut)
        bfilter.SetInputData(0, self)
        bfilter.ReorientDifferenceCellsOff()
        bfilter.SetTolerance(tolerance)
        bfilter.Update()
        return PolyData(bfilter.GetOutput())

    def BooleanAdd(self, mesh, merge=False):
        """
        Add a mesh to the current mesh.

        Parameters
        ----------
        mesh : vtkInterface.PolyData
            The mesh to add

        Returns
        -------
        joinedmesh : vtkInterface.PolyData
            Initial mesh and the new mesh.
        """
        vtkappend = vtk.vtkAppendPolyData()
        vtkappend.AddInputData(self)
        vtkappend.AddInputData(mesh)
        vtkappend.Update()
        return PolyData(vtkappend.GetOutput())

    def BooleanUnion(self, mesh):
        """
        Returns the mesh in common between the current mesh and the input mesh.

        Parameters
        ----------
        mesh : vtkInterface.PolyData
            The mesh to perform a union against.

        Returns
        -------
        union : vtkInterface.PolyData
            The union mesh
        """
        bfilter = vtk.vtkBooleanOperationPolyDataFilter()
        bfilter.SetOperationToUnion()
        bfilter.SetInputData(1, mesh)
        bfilter.SetInputData(0, self)
        bfilter.ReorientDifferenceCellsOff()
        bfilter.Update()
        return PolyData(bfilter.GetOutput())

    def Curvature(self, curvature='mean'):
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
        curvature = curvature.lower()

        # Create curve filter and compute curvature
        curvefilter = vtk.vtkCurvatures()
        curvefilter.SetInputData(self)
        if curvature == 'mean':
            curvefilter.SetCurvatureTypeToMean()
        elif curvature == 'gaussian':
            curvefilter.SetCurvatureTypeToGaussian()
        elif curvature == 'maximum':
            curvefilter.SetCurvatureTypeToMaximum()
        elif curvature == 'minimum':
            curvefilter.SetCurvatureTypeToMinimum()
        else:
            raise Exception('Curvature must be either "Mean", ' +
                            '"Gaussian", "Maximum", or "Minimum"')
        curvefilter.Update()

        # Compute and return curvature
        curves = curvefilter.GetOutput()
        return vtk_to_numpy(curves.GetPointData().GetScalars())

    def RemovePoints(self, remove_mask, mode='all', keepscalars=True):
        """
        Rebuild a mesh by removing points that are true in "remove_mask"

        Parameters
        ----------
        remove_mask : np.ndarray
            Points that are True will be removed.

        mode : str, optional
            When 'all', only faces containing all points flagged for removal
            will be removed.  Default 'all'

        keepscalars : bool, optional
            When True, point and cell scalars will be passed on to the new
            mesh.

        Returns
        -------
        mesh : vtkInterface.PolyData
            Mesh without the points flagged for removal.

        """
        # Extract points and faces from mesh
        v = self.points
        f = self.GetNumpyFaces()

        if remove_mask.size != v.shape[0]:
            raise Exception('"remove_mask" size is not the same as the ' +
                            'number of points in the mesh')

        vmask = remove_mask.take(f)
        if mode == 'all':
            fmask = np.logical_not(vmask).all(1)
        else:
            fmask = np.logical_not(vmask).any(1)

        # Regenerate face and point arrays
        uni = np.unique(f.compress(fmask, 0), return_inverse=True)
        v = v.take(uni[0], 0)
        f = np.reshape(uni[1], (fmask.sum(), 3))

        newmesh = vtkInterface.MeshfromVF(v, f, False)
        ridx = uni[0]

        # Add scalars back to mesh if requested
        if keepscalars:
            # Point data
            narr = self.GetPointData().GetNumberOfArrays()
            for i in range(narr):
                # Extract original array
                vtkarr = self.GetPointData().GetArray(i)

                # Rearrange the indices and add this to the new polydata
                adata = vtk_to_numpy(vtkarr)[ridx]
                newmesh.AddPointScalars(adata, vtkarr.GetName())

            # Cell data
            narr = self.GetCellData().GetNumberOfArrays()
            for i in range(narr):
                # Extract original array
                vtkarr = self.GetCellData().GetArray(i)
                adata = vtk_to_numpy(vtkarr)[fmask]
                newmesh.AddCellScalars(adata, vtkarr.GetName())

        # Return vtk surface and reverse indexing array
        return newmesh, ridx

    def Write(self, filename, ftype=None, binary=True):
        """
        Writes a surface mesh to disk.

        Written file may be an ASCII or binary ply, stl, or vtk mesh file.

        Parameters
        ----------
        filename : str
            Filename of mesh to be written.  Filetype is inferred from the
            extension of the filename unless overridden with ftype.  Can be
            one of the following types (.ply, .stl, .vtk)

        ftype : str, optional
            Filetype.  Inferred from filename unless specified with a three
            character string.  Can be one of the following: 'ply',  'stl', or
            'vtk'.

        Notes
        -----
        Binary files write much faster than ASCII.
        """
        if not ftype:
            ftype = filename[-3:]

        # Check filetype
        if ftype == 'ply':
            writer = vtk.vtkPLYWriter()
        elif ftype == 'stl':
            writer = vtk.vtkSTLWriter()
        elif ftype == 'vtk':
            writer = vtk.vtkPolyDataWriter()
        else:
            raise Exception('Filetype must be either "ply", "stl" or "vtk"')

        # Write
        writer.SetFileName(filename)
        writer.SetInputData(self)
        if binary:
            writer.SetFileTypeToBinary()
        else:
            writer.SetFileTypeToASCII()
        writer.Write()

    def PlotCurvature(self, curvtype='mean', **kwargs):
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
            See help(vtkInterface.Plot)

        Returns
        -------
        cpos : list
            List of camera position, focal point, and view up
        """
        # Get curvature values and plot
        c = self.Curvature(curvtype)

        # Return camera posision
        return vtkInterface.Plot(self, scalars=c, stitle='%s\nCurvature' % curvtype, **kwargs)

    def TriFilter(self):
        """
        Returns an all triangle mesh.  More complex polygons will be broken
        down into triangles.

        Returns
        -------
        mesh : vtkInterface.PolyData
            Mesh containing only triangles.
        """
        trifilter = vtk.vtkTriangleFilter()
        trifilter.SetInputData(self)
        trifilter.PassVertsOff()
        trifilter.PassLinesOff()
        trifilter.Update()
        return PolyData(trifilter.GetOutput())

    def Subdivide(self, nsub, subfilter='linear', inplace=False):
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
            vtkInterface polydata object.  None when inplace=True

        Examples
        --------
        >>> from vtkInterface import examples
        >>> import vtkInterface
        >>> mesh = vtkInterface.LoadMesh(examples.planefile)
        >>> submesh = mesh.Subdivide(1, 'loop')
        >>> mesh.Subdivide(1, 'loop', inplace=True)  # alternatively, update mesh in-place

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
        submesh = PolyData(sfilter.GetOutput())
        if inplace:
            self.OverwriteMesh(submesh)
        else:
            return submesh

    def ExtractEdges(self, feature_angle=30, boundary_edges=True,
                     non_manifold_edges=True, feature_edges=True,
                     manifold_edges=True):
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

        Returns
        -------
        edges : vtkInterface.vtkPolyData
            Extracted edges

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
        return PolyData(featureEdges.GetOutput())

    def Decimate(self, target_reduction, volume_preservation=False,
                 attribute_error=False, scalars=True, vectors=True,
                 normals=False, tcoords=True, tensors=True, scalars_weight=0.1,
                 vectors_weight=0.1, normals_weight=0.1, tcoords_weight=0.1,
                 tensors_weight=0.1, inplace=True):
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
        outmesh : vtkInterface.PolyData
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

        if inplace:
            self.Overwrite(decimate.GetOutput())
        else:
            return PolyData(decimate.GetOutput())

    def FlipNormals(self):
        """
        Flip normals of a triangular mesh by reversing the point ordering.

        """
        if self.faces.size % 4:
            raise Exception('Can only flip normals on an all triangular mesh')

        f = self.faces.reshape((-1, 4))
        f[:, 1:] = f[:, 1:][:, ::-1]

    def OverwriteMesh(self, mesh):
        """
        Degenerated.  Use Overwrite

        """
        # copy points and point data
        self.SetPoints(mesh.GetPoints())
        self.GetPointData().DeepCopy(mesh.GetPointData())

        # copy cells and cell data
        self.SetPolys(mesh.GetPolys())
        self.GetCellData().DeepCopy(mesh.GetCellData())

        # Must rebuild or subsequent operations on this mesh will segfault
        self.BuildCells()

    def Overwrite(self, mesh):
        """
        Overwrites the old mesh data with the new mesh data

        Parameters
        ----------
        mesh : vtk.vtkPolyData or vtkInterface.PolyData
            The overwriting mesh.

        """
        # copy points and point data
        self.SetPoints(mesh.GetPoints())
        self.GetPointData().DeepCopy(mesh.GetPointData())

        # copy cells and cell data
        self.SetPolys(mesh.GetPolys())
        self.GetCellData().DeepCopy(mesh.GetCellData())

        # Must rebuild or subsequent operations on this mesh will segfault
        self.BuildCells()

    def GenerateNormals(self, cell_normals=False, point_normals=True,
                        split_vertices=True, flip_normals=False,
                        consistent_normals=True, auto_orient_normals=False,
                        non_manifold_traversal=True, feature_angle=30.0):
        """Generate point and/or cell normals for a mesh.

        The filter can reorder polygons to insure consistent orientation across
        polygon neighbors. Sharp edges can be split (TODO) and points duplicated
        with separate normals to give crisp (rendered) surface definition. It is
        also possible to globally flip the normal orientation.

        The algorithm works by determining normals for each polygon and then
        averaging them at shared points. When sharp edges are present, the edges
        are split and new points generated to prevent blurry edges (due to
        Gouraud shading).

        Parameters
        ----------
        cell_normals : bool, optional
            Calculation of cell normals. Defaults to False.

        point_normals  : bool, optional
            Calculation of point normals. Defaults to True.

        split_vertices  : bool, optional
            Splitting of sharp edges. Defaults to True.

        flip_normals : bool, optional
            Set global flipping of normal orientation. Flipping modifies both
            the normal direction and the order of a cell's points. Defaults to
            False.

            TODO: Probably need to update point ordering in PolyData when
                setting flip_normals to True.

        consistent_normals  : bool, optional
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

        Note
        ----
        Previous arrays named "Normals" will be overwritten.

        Normals are computed only for polygons and triangle strips. Normals are
        not computed for lines or vertices.

        Triangle strips are broken up into triangle polygons. You may want to
        restrip the triangles.

        Return
        ------
        no return :
            Adds point and/or cell scalar data to PolyData object.

        """

        normal = vtk.vtkPolyDataNormals()
        normal.SetComputeCellNormals(cell_normals)
        normal.SetComputePointNormals(point_normals)
        normal.SetSplitting(split_vertices) # TODO: Vertex splitting not currently working. (Issue with vtkInterface.PolyData.AddPointScalars.)
        normal.SetFlipNormals(flip_normals)
        normal.SetConsistency(consistent_normals)
        normal.SetAutoOrientNormals(auto_orient_normals)
        normal.SetNonManifoldTraversal(non_manifold_traversal)
        normal.SetFeatureAngle(feature_angle)

        normal.SetInputData(self)
        normal.Update()

        if cell_normals:
            cnorms = normal.GetOutput().GetCellData().GetArray('Normals')
            self.AddCellScalars(vtk_to_numpy(cnorms), 'Normals', setactive=True,
                                deep=True)
            self.GetCellData().SetNormals(cnorms)

        if point_normals:
            pnorms = normal.GetOutput().GetPointData().GetArray('Normals')
            self.AddPointScalars(vtk_to_numpy(pnorms), 'Normals',
                                 setactive=True, deep=True)
            self.GetPointData().SetNormals(pnorms)


    def ClipPlane(self, origin, normal, value=0):
      """Clip a vtkInterface.PolyData or vtk.vtkPolyData with a plane.

      Can be used to open a mesh which has been closed along a well-defined
      plane.

      Parameters
      ----------
      origin : ndarray
          3D point through which plane passes. Defines the plane together with
          normal parameter.
      normal : ndarray
          3D vector defining plane normal.
      value : float, optional
          Scalar clipping value. The default value is 0.0.

      Note
      ----
      Not guaranteed to produce a manifold output.

      Return
      ------
      no return :
          Overwrites mesh.
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

      self.OverwriteMesh(clip.GetOutput())

    def ExtractLargest(self):
        """Extract largest connected set in mesh.

        Can be used to reduce residues obtained when generating an isosurface.
        Works only if residues are not connected (share at least one point with)
        the main component of the image.

        Return
        ------
        no return :
            Overwrites mesh.
        """
        connect = vtk.vtkConnectivityFilter()
        connect.SetExtractionModeToLargestRegion()

        connect.SetInputData(self)
        connect.Update()

        geofilter = vtk.vtkGeometryFilter()

        geofilter.SetInputData(connect.GetOutput())
        geofilter.Update()

        self.OverwriteMesh(geofilter.GetOutput())

    def FillHoles(self, size):
        """Fill holes in a vtkInterface.PolyData or vtk.vtkPolyData object.

        Holes are identified by locating boundary edges, linking them together
        into loops, and then triangulating the resulting loops. Note that you
        can specify an approximate limit to the size of the hole that can be
        filled.

        Parameters
        ----------
        size : float
            Specifies the maximum hole size to fill. This is represented as a
            radius to the bounding circumsphere containing the hole. Note that
            this is an approximate area; the actual area cannot be computed
            without first triangulating the hole.

        Return
        ------
        no return :
            Overwrites mesh.
        """

        fill = vtk.vtkFillHolesFilter()
        fill.SetHoleSize(size)

        fill.SetInputData(self)
        fill.Update()

        self.OverwriteMesh(fill.GetOutput())

    def Clean(self, point_merging=True, mergtol=None, lines_to_points=True,
              polys_to_lines=True, strips_to_polys=True):
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

        """
        clean = vtk.vtkCleanPolyData()
        if not lines_to_points:
            clean.ConvertLinesToPointsOff()
        if not polys_to_lines:
            clean.ConvertPolysToLinesOff()
        if not strips_to_polys:
            clean.ConvertStripsToPolysOff()
        if mergtol:
            clean.ToleranceIsAbsoluteOn()
            clean.SetAbsoluteTolerance(mergtol)
        clean.SetInputData(self)
        clean.Update()

        self.OverwriteMesh(clean.GetOutput())

    # def __del__(self):
    #     log.debug('Object collected')
