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
        Email author to suggest support for another filetype.


        Parameters
        ----------
        filename : str
            Filename of mesh to be written.  Filetype is inferred from the
            extension of the filename unless overridden with ftype.  Can be
            one of the following types (.ply, .stl, .vtk)

        ftype : str, optional
            Filetype.  Inferred from filename unless specified with a three
            character string.  Can be one of the following: 'ply',  'stl', or
            'vtk'

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

    def Subdivide(self, nsub, subfilter='linear'):
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

        Returns
        -------
        mesh : Polydata object
            vtkInterface polydata object.

        Examples
        --------
        >>> from vtkInterface import examples
        >>> import vtkInterface
        >>> mesh = vtkInterface.LoadMesh(examples.planefile)
        >>> submesh = mesh.Subdivide(1, 'loop')

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
        return PolyData(sfilter.GetOutput())

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

    def Decimate(self, target_reduction):
        """
        Reduces the number of triangles in a triangular mesh using
        vtkQuadricDecimation.

        Parameters
        ----------
        mesh : vtk.PolyData
            Mesh to decimate

        target_reduction : float Fraction of the original mesh to remove.
            TargetReduction is set to 0.9, this filter will try to reduce
            the data set to 10% of its original size and will remove 90%
            of the input triangles.

        Returns
        -------
        outmesh : vtkInterface.PolyData
            Decimated mesh

        """
        # create decimation filter
        decimate = vtk.vtkQuadricDecimation()  # vtkDecimatePro as well
        decimate.SetInputData(self)
        decimate.SetTargetReduction(target_reduction)
        decimate.Update()
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
        Overwrites the old mesh data with the new mesh data

        Parameters
        ----------
        mesh : vtk.vtkPolyData or vtkInterface.PolyData
            The overwriting mesh.

        """
        self.SetPoints(mesh.GetPoints())
        self.SetPolys(mesh.GetPolys())

        # Must rebuild or subsequent operations on this mesh will segfault
        self.BuildCells()

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
