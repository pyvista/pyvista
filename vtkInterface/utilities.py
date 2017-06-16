"""
VTK based supporting functions

"""
import os
import warnings
import types

import numpy as np
import vtk
from vtk.util import numpy_support as VN

# Determine if using vtk > 5
new_vtk = vtk.vtkVersion().GetVTKMajorVersion() > 5
if not new_vtk:
    warnings.warn('Using VTK version 5 or less. May encounter errors')

# mesh morph import
from vtkInterface import Plot
from vtkInterface import PlotCurvature

try:
    from pyansys import CellQuality
    hasqualfunc = True
except:
    hasqualfunc = False
    

#==============================================================================
# Functions
#==============================================================================
def RotateX(mesh, angle):
    """ Rotates mesh about the X axis"""
    trans = vtk.vtkTransform()
    trans.RotateX(angle)
    ApplyTransformationInPlace(mesh, trans)
        

def RotateY(mesh, angle):
    """ Rotates mesh or refined mesh inplace """
    trans = vtk.vtkTransform()
    trans.RotateY(angle)
    ApplyTransformationInPlace(mesh, trans)
        
        
def RotateZ(mesh, angle):
    """ Rotates mesh or refined mesh inplace """
    trans = vtk.vtkTransform()
    trans.RotateZ(angle)
    ApplyTransformationInPlace(mesh, trans)
        
        
def Translate(mesh, xyz):
    """ Translates a mesh inplace """
                    
    trans = vtk.vtkTransform()
    trans.Translate(xyz[0], xyz[1], xyz[2])
    ApplyTransformationInPlace(mesh, trans)


def Subdivide(mesh, nsub, subfilter='linear'):
    """
    Increase the number of triangles in a single, connected triangular mesh.
    
    Uses one of the following vtk subdivision filters to subdivide a mesh.
    vtkButterflySubdivisionFilter
    vtkLoopSubdivisionFilter
    vtkLinearSubdivisionFilter

    Linear subdivision results in the fastest mesh subdivision, but it does not
    smooth mesh edges, but rather splits each triangle into 4 smaller 
    triangles.
    
    Butterfly and loop subdivision perform smoothing when dividing, and may
    introduce artifacts into the mesh when dividing.
    
    Subdivision filter appears to fail for multiple part meshes.  Should be one
    single mesh.
    
    Parameters
    ----------
    mesh : vtk.vtkPolyData
        Mesh to be subdivided.  Must be a triangular mesh.
    nsub : int
        Number of subdivisions.  Each subdivision creates 4 new triangles, so
        the number of resulting triangles is nface*4**nsub where nface is the
        current number of faces.
    subfilter : string, optional
        Can be one of the following: 'butterfly', 'loop', 'linear'
        
    Returns
    -------
    mesh : vtkPolydata object
        VTK surface mesh object.
        
    Examples
    --------
    >>> from vtkInterface import examples
    >>> import vtkInterface
    
    >>> mesh = vtkInterface.LoadMesh(examples.planefile)
    >>> submesh = mesh.Subdivide(1, 'loop')
    
    """
    
    # select filter
    if subfilter is 'linear':
        sfilter = vtk.vtkLinearSubdivisionFilter() 
    elif subfilter is 'butterfly':
        sfilter = vtk.vtkButterflySubdivisionFilter()
    elif subfilter is 'loop':
        sfilter = vtk.vtkLoopSubdivisionFilter()
    else:
        raise Exception("Subdivision filter must be one of the following: " +\
                        "'butterfly', 'loop', or 'linear'")
        
    # Subdivide
    sfilter.SetNumberOfSubdivisions(nsub)
    sfilter.SetInputData(mesh)
    sfilter.Update()
    submesh = sfilter.GetOutput()
    AddFunctions(submesh)

    return submesh


def AddFunctions(grid):
    
    # check if it already has the convenience functions
    if hasattr(grid, 'Plot'):
        return
    
    # otherwise, add the convenience functions
    if isinstance(grid, vtk.vtkUnstructuredGrid):
        GridAddExtraFunctions(grid)
    elif isinstance(grid, vtk.vtkStructuredGrid):
        GridAddExtraFunctions(grid)
    elif isinstance(grid, vtk.vtkPolyData):
        PolyAddExtraFunctions(grid)
    else:
        PolyAddExtraFunctions(grid)
        
        
def PolyAddExtraFunctions(poly):
    """ Adds convenience functions to vtk.vtkpoly objects """
    poly.GetNumpyPoints  = types.MethodType(GetPoints, poly)
    poly.SetNumpyPoints  = types.MethodType(SetPoints, poly)
    poly.GetNumpyFaces   = types.MethodType(GetFaces, poly)
    poly.GetPointScalars = types.MethodType(GetPointScalars, poly)
    poly.AddPointScalars = types.MethodType(AddPointScalars, poly)
    poly.GetCellScalars  = types.MethodType(GetCellScalars, poly)
    poly.AddCellScalars  = types.MethodType(AddCellScalars, poly)      
    poly.ApplyTransformation = types.MethodType(ApplyTransformation, poly)      
    poly.ApplyTransformationInPlace = types.MethodType(ApplyTransformationInPlace, poly)
    poly.Plot            = types.MethodType(Plot, poly)
    poly.RotateX         = types.MethodType(RotateX, poly)
    poly.RotateY         = types.MethodType(RotateY, poly)
    poly.RotateZ         = types.MethodType(RotateZ, poly)
    poly.Translate       = types.MethodType(Translate, poly)
    poly.Copy            = types.MethodType(CopyVtkObject, poly)
    poly.GetEdgeMask     = types.MethodType(GetEdgeMask, poly)
    poly.BooleanCut      = types.MethodType(BooleanCut, poly)
    poly.BooleanAdd      = types.MethodType(BooleanAdd, poly)
    poly.GetCurvature    = types.MethodType(GetCurvature, poly)
    poly.SetNumpyPolys   = types.MethodType(SetNumpyPolys, poly)
    poly.RemovePoints    = types.MethodType(RemovePoints, poly)
    poly.WriteMesh       = types.MethodType(WriteMesh, poly)
    poly.CheckArrayExists = types.MethodType(CheckArrayExists, poly)
    poly.PlotCurvature = types.MethodType(PlotCurvature, poly)
    poly.TriFilter      = types.MethodType(TriFilter, poly)
    poly.Subdivide       = types.MethodType(Subdivide, poly)
    
    
def GridAddExtraFunctions(grid):
    """
    Adds convenience functions to vtkUnstructuredGrid or vtkStructuredGrid
    objects
    """

    # Check if object is a unstructred or structured grid    
#    if isinstance(grid, vtk.vtkUnstructuredGrid):
#        gridobj = vtk.vtkUnstructuredGrid
#    elif isinstance(grid, vtk.vtkStructuredGrid):
#        gridobj = vtk.vtkStructuredGrid
#    else:
#        raise Exception('Cannot add grid functions to a non-grid object')
        
    # Add unbound function added
    grid.GetNumpyPoints  = types.MethodType(GetPoints, grid)
    grid.SetNumpyPoints  = types.MethodType(SetPoints, grid)
    grid.GetNumpyCells   = types.MethodType(ReturnCells, grid)
    grid.GetPointScalars = types.MethodType(GetPointScalars, grid)
    grid.AddPointScalars = types.MethodType(AddPointScalars, grid)    
    grid.GetCellScalars  = types.MethodType(GetCellScalars, grid)
    grid.AddCellScalars  = types.MethodType(AddCellScalars, grid)      
    grid.ApplyTransformation = types.MethodType(ApplyTransformation, grid)      
    grid.ApplyTransformationInPlace = types.MethodType(ApplyTransformationInPlace, grid)      
    grid.ExtractExteriorTri = types.MethodType(ExtractExteriorTri, grid)      
    grid.ExtractSurface = types.MethodType(ExtractSurface, grid)      
    grid.Plot = types.MethodType(Plot, grid)
    grid.Copy = types.MethodType(CopyVtkObject, grid)
    grid.CheckArrayExists = types.MethodType(CheckArrayExists, grid)
    grid.ExtractSurfaceInd = types.MethodType(ExtractSurfaceInd, grid)
    grid.TriFilter      = types.MethodType(TriFilter, grid)
    grid.WriteGrid  = types.MethodType(WriteGrid, grid)

    # Optional pyansys cell quality calculator
    if hasqualfunc:
        grid.CellQuality = types.MethodType(CellQuality, grid)
        


def MakeuGrid(offset, cells, cell_type, nodes):
    """ Create VTK unstructured grid """
    
    # Check inputs (necessary since offset and cells can int32)    
    if offset.dtype != 'int64':
        offset = offset.astype(np.int64)
    
    if cells.dtype != 'int64':
        cells = cells.astype(np.int64)
    
    if not cells.flags['C_CONTIGUOUS']:
        cells = np.ascontiguousarray(cells)
        
    if cells.ndim != 1:
        cells = cells.ravel()
        
    if cell_type != np.uint8:
        cell_type = cell_type.astype(np.uint8)
    
    # Get number of cells
    ncells = cell_type.size
    
    # Convert to vtk arrays
    cell_type = VN.numpy_to_vtk(cell_type, deep=True)
    offset = VN.numpy_to_vtkIdTypeArray(offset, deep=True)
    
    vtkcells = vtk.vtkCellArray()
    vtkcells.SetCells(ncells, VN.numpy_to_vtkIdTypeArray(cells, deep=True))
    
    # Convert points to vtkfloat object
    vtkArray = VN.numpy_to_vtk(np.ascontiguousarray(nodes), deep=True)
    points = vtk.vtkPoints()
    points.SetData(vtkArray)
    
    # Create unstructured grid
    uGrid = vtk.vtkUnstructuredGrid()
    uGrid.SetPoints(points)
    uGrid.SetCells(cell_type, offset, vtkcells)
    GridAddExtraFunctions(uGrid)
    
    return uGrid

def BooleanCut(mesh, cut):
    """ Performs a boolean cut on 'mesh' using 'cut' """
    
    bfilter = vtk.vtkBooleanOperationPolyDataFilter()
    bfilter.SetOperationToIntersection()
    bfilter.SetInputData(1, cut)
    bfilter.SetInputData(0, mesh)
    bfilter.ReorientDifferenceCellsOff()
    bfilter.Update()
    cut = bfilter.GetOutput()
    PolyAddExtraFunctions(cut)
    return cut


def RemovePoints(mesh, remove_mask, mode='all', keepscalars=True):
    """ 
    Rebuild a mesh by removing points that are true in "remove_mask"
    """

    # Extract points and faces from mesh
    v = mesh.GetNumpyPoints()
    f = mesh.GetNumpyFaces()
        
    vmask = remove_mask.take(f)
    if mode == 'all':
        fmask = np.logical_not(vmask).all(1)
    elif mode == '_any':
        fmask = np.logical_not(vmask).any(1)
    else:
        fmask = np.logical_not(vmask).any(1)
    
    # Regenerate face and point arrays
    uni = np.unique(f.compress(fmask, 0), return_inverse=True)
    v = v.take(uni[0], 0)
    f = np.reshape(uni[1], (fmask.sum(), 3))
    
    newmesh = MeshfromVF(v, f, False)
    ridx = uni[0]

    # Add scalars back to mesh if requested
    if keepscalars:
        # Point data
        narr = mesh.GetPointData().GetNumberOfArrays()
        for i in range(narr):
            # Extract original array
            vtkarr = mesh.GetPointData().GetArray(i)
            
            # Rearrange the indices and add this to the new polydata
            adata = VN.vtk_to_numpy(vtkarr)[ridx]
            newmesh.AddPointScalars(adata, vtkarr.GetName())
        
        # Cell data
        narr = mesh.GetCellData().GetNumberOfArrays()
        for i in range(narr):
            # Extract original array
            vtkarr = mesh.GetCellData().GetArray(i)
            adata = VN.vtk_to_numpy(vtkarr)[fmask]
            newmesh.AddCellScalars(adata, vtkarr.GetName())
            
    # Return vtk surface and reverse indexing array
    return newmesh, ridx


def BooleanAdd(meshA, meshB):
    """ Creates a new mesh that is the boolean add """
    vtkappend = vtk.vtkAppendPolyData()
    vtkappend.AddInputData(meshA)
    vtkappend.AddInputData(meshB)
    vtkappend.Update()
    mesh = vtkappend.GetOutput()
    PolyAddExtraFunctions(mesh)
    return mesh
    
    

def OverwriteMesh(oldmesh, newmesh):
    """ Overwrites the old mesh data with the new mesh data """
    oldmesh.SetPoints(newmesh.GetPoints())
    oldmesh.SetPolys(newmesh.GetPolys())

    # Must rebuild or subsequent operations on this mesh will segfault
    oldmesh.BuildCells()


def CheckArrayExists(vobj, name):
    """ Returns true if array exists in a vtk object """
    if vobj.GetPointData().GetArray(name):
        return True
    else:
        return False
        

def ReturnCells(grid, dtype=None):
    """
    Returns the raw numpy array of cells from a vtk grid object
    """
    
    # grab cell data
    vtkarr = grid.GetCells().GetData()
    if vtkarr:
        cells = VN.vtk_to_numpy(vtkarr)
    else:
        return None
    
    if dtype:
        if cells.dtype != dtype:
            cells = cells.astype(dtype)
            
    return cells


def GetPointScalars(vobj, name):
    """ Returns point scalars of a vtk object """
    vtkarr = vobj.GetPointData().GetArray(name)

    if vtkarr:
        array = VN.vtk_to_numpy(vtkarr)
        
        # Convert int8 to bool
        if array.dtype == np.int8:
            array = array.astype(np.bool)
            
        return array
    
    else:
        return None
        
        
def GetCellScalars(vobj, name):
    """ Returns point scalars of a vtk object """
    vtkarr = vobj.GetCellData().GetArray(name)
    if vtkarr:
        return VN.vtk_to_numpy(vtkarr)
    else:
        return None


def CopyVtkObject(vtkobject):
    """ Copies a vtk structured, unstructured, or polydata object """
    
    # Grab vtk type from string (seems to be vtk version dependent)
    if isinstance(vtkobject, vtk.vtkPolyData):
        vtkobject_copy = vtk.vtkPolyData()

    elif isinstance(vtkobject, vtk.vtkUnstructuredGrid):
        vtkobject_copy = vtk.vtkUnstructuredGrid()
        
    elif isinstance(vtkobject, vtk.vtkStructuredGrid):
        vtkobject_copy = vtk.vtkStructuredGrid()
        
    else:
        raise Exception('Unsupported object for VTK copy')
        
    # copy, add extra functions
    vtkobject_copy.DeepCopy(vtkobject)
    AddFunctions(vtkobject_copy)
    
    return vtkobject_copy
    

def PerformLandmarkTrans(sland, tland):
    """ Performs a landmark transformation between two sets of points """
    
    slandvtk = MakevtkPoints(sland)
    tlandvtk = MakevtkPoints(tland)

    landtrans = vtk.vtkLandmarkTransform()
    landtrans.SetSourceLandmarks(slandvtk)
    landtrans.SetTargetLandmarks(tlandvtk)
    landtrans.SetModeToRigidBody()
    landtrans.Update()

    return landtrans


def MakevtkPoints(points, deep=True):
    """ Convert numpy points to vtkPoints """

    # Data checking
    if not points.flags['C_CONTIGUOUS']:
        points = np.ascontiguousarray(points)

    vtkpts = vtk.vtkPoints()
    vtkpts.SetData(VN.numpy_to_vtk(points, deep=deep))
    return vtkpts
    

def SetVTKInput(obj, inp):
    """ Accounts for version discrepancy between VTK versions in input method """
    if new_vtk:
        obj.SetInputData(inp)
    else:
        obj.SetInput(inp)
        
        
def CyclicReplication(surf, ncopy):        
    """ Replicates a surface cyclically "ncopy" times about the z-axis """
    
    # Create VTK append object
    vtkappend = vtk.vtkAppendPolyData()
        
    # Copy and translate mesh
    rang = 360.0/ncopy
    for i in range(ncopy):
            
        # Transform mesh
        trans = vtk.vtkTransform()
        trans.RotateZ(rang*i)
        tsurf = ApplyTransformation(surf, trans)
        
        # Append
        if new_vtk:
            vtkappend.AddInputData(tsurf)
        else:
            vtkappend.AddInput(tsurf)
            
    # Combine meshes and return
    vtkappend.Update()

    surfrep = vtkappend.GetOutput()
    GridAddExtraFunctions(surfrep)
    return surfrep
    
    
def CyclicReplication2(surf, ncopy, cyclicpair):        
    """ Replicates a surface cyclically "ncopy" times about the z-axis """
    
    # Create VTK append object
    vtkappend = vtk.vtkAppendPolyData()

    # Get rotation angle
    rang = 360.0/ncopy

    lowpair = cyclicpair[:, 0]
    higpair = cyclicpair[:, 1]

    # Create an averaged interface surface
    gridpts = GetPoints(surf)[lowpair]
        
    # Copy and translate mesh
    for i in range(ncopy):
            
        # Transform mesh
        trans = vtk.vtkTransform()
        trans.RotateZ(rang*i)
        tsurf = ApplyTransformation(surf, trans)
        
        # Set low and high side points
        tsurfpts = GetPoints(tsurf)
        tsurfpts[lowpair] = XY_rot(gridpts, rang*i)
        tsurfpts[higpair] = XY_rot(gridpts, rang*(i + 1))
        SetPoints(tsurf, tsurfpts)        
        
        # Append
        if new_vtk:
            vtkappend.AddInputData(tsurf)
        else:
            vtkappend.AddInput(tsurf)
            
    # Combine meshes and return
    vtkappend.Update()
    return vtkappend.GetOutput()
    
    
    
def CyclicReplicationSolid(grid, ncopy, cyclicpair):
    """ 
    Replicates a structured/unstructured grid cyclically "ncopy" times about the
    z-axis """
    # Get rotation angle
    rang = 360.0/ncopy

    lowpair = cyclicpair[0]
    higpair = cyclicpair[1]

    # Create an averaged interface surface
    gridpts = GetPoints(grid)[lowpair]

    # Copy and translate mesh
    vtkappend = vtk.vtkAppendFilter()
    for i in range(ncopy):
            
        # Transform mesh
        trans = vtk.vtkTransform()
        trans.RotateZ(rang*i)
        tgrid = ApplyTransformationSolid(grid, trans)
        
        # Set low and high side points
        tsurfpts = GetPoints(tgrid)
        tsurfpts[lowpair] = XY_rot(gridpts, rang*i)
        if i == ncopy - 1:
            tsurfpts[higpair] = XY_rot(gridpts, 0.0)
        else:
            tsurfpts[higpair] = XY_rot(gridpts, rang*(i + 1))
        SetPoints(tgrid, tsurfpts)
        
        # Append
        if new_vtk:
            vtkappend.AddInputData(tgrid)
        else:
            vtkappend.AddInput(tgrid)
            
    # Combine meshes and add VTK_Utilities functions
    vtkappend.MergePointsOn()
    vtkappend.Update()
    gridout = vtkappend.GetOutput()
    GridAddExtraFunctions(gridout)
    
    return gridout


def AddCellScalars(grid, scalars, name, setactive=True):
    """ Adds cell scalars to uGrid """
    vtkarr = VN.numpy_to_vtk(np.ascontiguousarray(scalars), deep=True)
    vtkarr.SetName(name)
    grid.GetCellData().AddArray(vtkarr)
    if setactive:
        grid.GetCellData().SetActiveScalars(name)
    

def UpdateCellScalars(mesh, scalars, name):
    """ Updates the underlying data of a mesh """
    vtkarrnew = VN.numpy_to_vtk(np.ascontiguousarray(scalars), deep=True)
    vtkarrnew.SetName(name)
    
    # Get a pointer to the accuracy array
    vtkarr = mesh.GetCellData().GetArray(name)
    vtkarr.DeepCopy(vtkarrnew)    
    
    
def AddPointScalars(mesh, scalars, name, setactive=False):
    """
    Adds point scalars to a VTK object or structured/unstructured grid """
    
    if scalars.dtype == np.bool:
        scalars = scalars.astype(np.int8)
    
    vtkarr = VN.numpy_to_vtk(np.ascontiguousarray(scalars), deep=True)
    vtkarr.SetName(name)
    mesh.GetPointData().AddArray(vtkarr)
    if setactive:
        mesh.GetPointData().SetActiveScalars(name)
    
    
def UpdatePointScalars(mesh, scalars, name):
    """ Updates the underlying data of a mesh """
    vtkarrnew = VN.numpy_to_vtk(np.ascontiguousarray(scalars), deep=True)
    vtkarrnew.SetName(name)
    
    # Get a pointer to the accuracy array
    vtkarr = mesh.GetPointData().GetArray(name)
    vtkarr.DeepCopy(vtkarrnew)


def GetBoundaryPoints(mesh):
    """
    Extracts boundary points from a vtk mesh and return the indices of those
    points """    
    
    # Create feature object
    featureEdges = vtk.vtkFeatureEdges()
    SetVTKInput(featureEdges, mesh) 
    featureEdges.FeatureEdgesOff()
    featureEdges.BoundaryEdgesOn()
    featureEdges.NonManifoldEdgesOff()
    featureEdges.ManifoldEdgesOff()
    featureEdges.Update()
    edges = featureEdges.GetOutput()

    # Get the location of the boundary points
    return GetPoints(edges)#.astype(np.float64)

    # Return indices of the boundary points
#    bidx = Meshutil.SingleNeighbor(GetPoints(mesh), bpts) 

#    return bidx


#def GetEdgePoints(mesh, angle, return_ind=True):
#    """
#    Extracts edge points from a vtk mesh for a given angle and return the
#    indices of those points
#    
#    """
#    
#    featureEdges = vtk.vtkFeatureEdges()
#    SetVTKInput(featureEdges, mesh)
#    featureEdges.FeatureEdgesOn()
#    featureEdges.BoundaryEdgesOff()
#    featureEdges.NonManifoldEdgesOff()
#    featureEdges.ManifoldEdgesOff()
#    featureEdges.SetFeatureAngle(angle)
#    featureEdges.ColoringOff()
#    featureEdges.Update()
#    edges = featureEdges.GetOutput()
#    
#    # Extract Points
#    epts = VN.vtk_to_numpy(edges.GetPoints().GetData())
#    
#    if return_ind:
#        ind = Meshutil.SingleNeighbor(GetPoints(mesh), epts)
#        return edges, ind
#    else:
#        return edges
    
    
def GetEdgeMask(mesh, angle):
    """
    DESCRIPTION
    Returns a mask of the points of a surface mesh that have a surface angle
    greater than angle
    
    """
    
    featureEdges = vtk.vtkFeatureEdges()
    SetVTKInput(featureEdges, mesh)
    featureEdges.FeatureEdgesOn()
    featureEdges.BoundaryEdgesOff()
    featureEdges.NonManifoldEdgesOff()
    featureEdges.ManifoldEdgesOff()
    featureEdges.SetFeatureAngle(angle)
    featureEdges.Update()
    edges = featureEdges.GetOutput()
    
    return np.in1d(mesh.GetPointScalars('vtkOriginalPointIds'),
                   GetPointScalars(edges, 'vtkOriginalPointIds'), 
                                  assume_unique=True)


def GetMeshAreaVol(mesh):
    """ Returns volume and area of a triangular mesh """
    mprop = vtk.vtkMassProperties()
    SetVTKInput(mprop, mesh) 
    mprop.Update() 
    return mprop.GetSurfaceArea(), mprop.GetVolume()


def GetPoints(mesh, datatype=None, deep=False):
    """ returns points from a mesh as numpy array """
    points = VN.vtk_to_numpy(mesh.GetPoints().GetData())
    
    if datatype:
        if points.dtype != datatype:
            return points.astype(datatype)
    
    # Copy if requested
    if deep:
        return points.copy()
    else:
        return points


def GetFaces(mesh, force_C_CONTIGUOUS=False, nocut=False, dtype=None,
             raw=False):
    """ 
    Returns the faces from a polydata object as a numpy int array
    """
    if raw:
        return VN.vtk_to_numpy(mesh.GetPolys().GetData())

    if nocut:
        return VN.vtk_to_numpy(mesh.GetPolys().GetData()).reshape((-1, 4))

    # otherwise remove triangle size padding    
    f = VN.vtk_to_numpy(mesh.GetPolys().GetData()).reshape(-1, 4)[:, 1:]

    if force_C_CONTIGUOUS:
        f = np.ascontiguousarray(f)

    if dtype:
        if f.dtype != dtype:
            f = f.astype(dtype)
            
    return f


def GenStructuredGrid(x, y=None, z=None):
    """ Generates structured grid """
    
    # check if user has input an array of points
    if not np.any(y) and not np.any(z):
        points = x
        
    # Assemble points array
    else:
#        points = np.hstack((x.ravel(order='F'),
#                            y.ravel(order='F'),
#                            z.ravel(order='F'))).reshape(-1, 3, order='F')
        
        points = np.empty((x.size, 3))
        points[:, 0] = x.ravel('F')
        points[:, 1] = y.ravel('F')
        points[:, 2] = z.ravel('F')
                            
    # Convert to VTK points array
    vtkpoints = MakevtkPoints(points)

    dim = list(x.shape)
    while len(dim) < 3:
        dim.append(1)
        
    # Create structured grid
    sgrid = vtk.vtkStructuredGrid()
    sgrid.SetDimensions(dim)
    sgrid.SetPoints(vtkpoints)
    GridAddExtraFunctions(sgrid)

    return sgrid


def ExtractSurfaceInd(uGrid):
    """ Output the surface indices of a grid """
    # Extract surface mesh
    surf = vtk.vtkDataSetSurfaceFilter()
    SetVTKInput(surf, uGrid)        
    surf.PassThroughPointIdsOn()
    surf.Update()
    surf = surf.GetOutput()
    return GetPointScalars(surf, 'vtkOriginalPointIds')


def ExtractSurface(grid):
    """ Extracts surface mesh of a vtk grid """
    # Extract surface mesh
    surf = vtk.vtkDataSetSurfaceFilter()
    SetVTKInput(surf, grid)        
    surf.PassThroughPointIdsOn()
    surf.PassThroughCellIdsOn()
    surf.Update()
    
    # Return output
    surf = surf.GetOutput()
    PolyAddExtraFunctions(surf)
    return surf


def ExtractExteriorTri(uGrid, extract_extern=False):
    """
    Creates an all tri surface mesh from an unstructured grid
    
    Input:
        uGrid: Unstructured grid
        
    Output:
        extsurf: vtkPolyData surface containing array 'vtkOriginalPointIds' relating the points of
                 extsurf and uGrid
    
    """    

    # Extract surface mesh
    surf = vtk.vtkDataSetSurfaceFilter()
    SetVTKInput(surf, uGrid)        
    surf.PassThroughPointIdsOn()
    surf.PassThroughCellIdsOn()
    surf.Update()
    surf = surf.GetOutput()
    PolyAddExtraFunctions(surf)
    
    # Return triangle mesh as well as original
    return TriFilter(surf), surf

    
def TriFilter(mesh):
    """ Returns an all triangle mesh of a polydata """
    trifilter = vtk.vtkTriangleFilter()
    SetVTKInput(trifilter, mesh)
    trifilter.PassVertsOff()
    trifilter.PassLinesOff()
    trifilter.Update()
    
    # Return triangular exterior surface mesh and nontriangular
    trimesh = trifilter.GetOutput()
    PolyAddExtraFunctions(trimesh)
    return trimesh
        

    
def CyclicRotate(mesh, angle):
    """ Rotates a mesh about the z axis for a given angle """
    
    # Extract and rotate points from mesh
    pts = XY_rot(GetPoints(mesh), angle)
    
    # Reinsert these points into the mesh
    SetPoints(mesh, pts)
    
    
def XY_rot(p, ang, inplace=False, deg=True):
    """ Rotates points p angle ang (in deg) about the Z axis """
    
    # Copy original array to if not inplace
    if not inplace:
        p = p.copy()
    
    # Convert angle to radians
    if deg:
        ang *= np.pi/180
    
    x = p[:, 0]*np.cos(ang) - p[:, 1]*np.sin(ang)
    y = p[:, 0]*np.sin(ang) + p[:, 1]*np.cos(ang)
    p[:, 0] = x
    p[:, 1] = y

    if not inplace:
        return p
    
    
def XY_rot_inplace(p, ang):
    """ Rotates points p angle ang (in deg) about the Z axis """
    warnings.warn('XY_rot_inplace is depreciated.  Use XY_rot with inplace=True')
                                   
    # Copy original array to avoid modifying the points inplace
    pcpy = p.copy()
    # Convert angle to radians
    ang *= np.pi/180
    
    x = pcpy[:, 0]*np.cos(ang) - pcpy[:, 1]*np.sin(ang)
    y = pcpy[:, 0]*np.sin(ang) + pcpy[:, 1]*np.cos(ang)
    pcpy[:, 0] = x
    pcpy[:, 1] = y

    return pcpy
    
    
def XY_rot_rad(p, ang):
    """ Rotates points p angle ang (in deg) about the Z axis """
    warnings.warn('XY_rot_rad is depreciated.  Use XY_rot with deg=False')
    
    x = p[:, 0]*np.cos(ang) - p[:, 1]*np.sin(ang)
    y = p[:, 0]*np.sin(ang) + p[:, 1]*np.cos(ang)
    p[:, 0] = x
    p[:, 1] = y

    return p
    

def GridtoTri(uGrid):
    """
    Creates an all tri surface mesh from an unstructured grid
    
    Input:
        uGrid: Unstructured grid
        
    Output:
        surf: vtkPolyData surface
        IDs: Map between uGrid points and surf points
    
    """

    # Extract surface mesh
    surf = vtk.vtkDataSetSurfaceFilter()
    SetVTKInput(surf, uGrid)
    surf.PassThroughPointIdsOn()
    surf.PassThroughCellIdsOn()
    surf.Update()
    # relationship between the surface nodes and the original unstructured mesh
    origID = GetPointScalars(surf, 'vtkOriginalPointIds')    

    trianglefilter = vtk.vtkTriangleFilter()
    SetVTKInput(trianglefilter, surf.GetOutput())
    trianglefilter.PassVertsOn()
    trianglefilter.PassLinesOff()
    trianglefilter.Update()
    
    # Try returning a non-triangle filter surface
    return surf.GetOutput(), origID
    

def FEMtoTri(points, tri, quad):
    """ Creates an all tri surface mesh from FEM data  """
    

    # Assemble faces and split quad faces into triangles
    if np.any(quad):
        tri = np.vstack((tri, quad[:, np.array([0, 2, 3])], quad[:, :3]))

    # Reindex triangle array 0
    uni = np.unique(tri, return_inverse=True)
    tri = np.reshape(uni[1], tri.shape)
    
    # Convert points to polydata object
    vtkfloat = VN.numpy_to_vtk(np.ascontiguousarray(points[uni[0], :]), deep=True)
    vtkpointarray = vtk.vtkPoints()
    vtkpointarray.SetData(vtkfloat)
    
    # Convert triangle face to cell polydata object
    threes = np.ones((tri.shape[0], 1))*3
    IDX = np.ascontiguousarray(np.hstack((threes, tri)).astype('int64').ravel())
    vtkcellarray = vtk.vtkCellArray()
    vtkcellarray.SetCells(tri.shape[0], VN.numpy_to_vtkIdTypeArray(IDX, deep = True))

    # Create polydata object
    surf = vtk.vtkPolyData()
    surf.SetPoints(vtkpointarray)
    surf.SetPolys(vtkcellarray)

    # ensure consistent ordering
    vtknormals = vtk.vtkPolyDataNormals()
    SetVTKInput(vtknormals, surf)
    vtknormals.SplittingOff()
    vtknormals.AutoOrientNormalsOn()
    vtknormals.Update()

    surf.SetPolys(vtknormals.GetOutput().GetPolys())
    
    # return polydata surface as well as relationship between surface and original numbering scheme
    return surf, uni[0]
        
    
    
def SetPoints(VTKobject, points, deep=True):
    """ Sets points on a VTK object """
    VTKobject.SetPoints(MakevtkPoints(points))
        
    
def GetCurvature(mesh, curvature='Mean'):
    """
    DESCRIPTION
    Returns the pointwise curvature of a mesh
    
    INPUTS
    mesh (vtk polydata)
        vtk polydata mesh
        
    curvature (string, optional default 'Gaussian')
        Mean
        Gaussian
        Maximum
        Minimum
        
    OUTPUTS
    curvature (numpy array)
        Curvature values
        
    """
    
    # Create curve filter and compute curvature
    curvefilter = vtk.vtkCurvatures()
    SetVTKInput(curvefilter, mesh)
    if curvature == 'Mean':
        curvefilter.SetCurvatureTypeToMean()
    elif curvature == 'Gaussian': 
        curvefilter.SetCurvatureTypeToGaussian()
    elif curvature == 'Maximum':
        curvefilter.SetCurvatureTypeToMaximum()
    elif curvature == 'Minimum':
        curvefilter.SetCurvatureTypeToMinimum()
    else:
        raise Exception('Curvature must be either "Mean", "Gaussian", "Maximum", or "Minimum"')
    curvefilter.Update()
    
    # Compute and return curvature
    curves = curvefilter.GetOutput()
    return VN.vtk_to_numpy(curves.GetPointData().GetScalars())
    

def CleanMesh(mesh, return_indices=False, mergtol=None):
    """ Cleans mesh and returns original indices """
    
    if return_indices:
        npoints = mesh.GetNumberOfPoints()
        AddPointScalars(mesh, np.arange(npoints), 'cleanIDX', False)

    clean = vtk.vtkCleanPolyData()
    clean.ConvertPolysToLinesOff()
    clean.ConvertLinesToPointsOff()
    clean.ConvertStripsToPolysOff()
    if mergtol:
        clean.ToleranceIsAbsoluteOn()
        clean.SetAbsoluteTolerance(mergtol)
    SetVTKInput(clean, mesh)
    clean.Update()
    
    # Create new polydata and add extra functions
    cleanmesh = clean.GetOutput()
    PolyAddExtraFunctions(cleanmesh)
    
    if return_indices:
        origID = VN.vtk_to_numpy(cleanmesh.GetPointData().GetArray('cleanIDX'))
        
        # remove last array
        narr = cleanmesh.GetPointData().GetNumberOfArrays()
        cleanmesh.GetPointData().RemoveArray(narr - 1)
        
        return cleanmesh, origID
        
    else:
        return cleanmesh
        
        
#def GenTriangularStructSurf(points, shp):
#    """ Generate a triangular surface from a set of structured points """
#    
#    # Convert points to a vtk array
#    vtkArray = VN.numpy_to_vtk(np.ascontiguousarray(points), deep=True)
#    points = vtk.vtkPoints()
#    points.SetData(vtkArray)
#    
#    # Create structured grid
#    sgrid = vtk.vtkStructuredGrid()
#    sgrid.SetDimensions(shp)
#    sgrid.SetPoints(points)    
#    
#    # Return a triangular surface
#    return ExtractExteriorTri(sgrid)[0]


def GenStructSurf(x, y, z, trionly=False):
    """ Generate a triangular surface from a set of structured points """
    
    points = np.empty((x.size, 3))
    points[:, 0] = x.ravel('F')
    points[:, 1] = y.ravel('F')
    points[:, 2] = z.ravel('F')
    
    # Convert points to a vtk array
    vtkpoints = MakevtkPoints(points)

    dim = list(x.shape)
    while len(dim) < 3:
        dim.append(1)
    
    # Create structured grid
    sgrid = vtk.vtkStructuredGrid()
    sgrid.SetDimensions(dim)
    sgrid.SetPoints(vtkpoints) 
    AddFunctions(sgrid)
    
    # Return a triangular surface
    if trionly:
        return ExtractExteriorTri(sgrid)[0]
    else:
        return ExtractExteriorTri(sgrid)[1]
        
        
def MakeLine(points):
    """ Generates line from points.  Assumes points are ordered """
    
    # Assuming ordered points, create array defining line order
    npoints = points.shape[0] - 1
    lines = np.vstack((2*np.ones(npoints, np.int),
                       np.arange(npoints),
                       np.arange(1, npoints + 1))).T.ravel()

    # Create polydata object
    line = vtk.vtkPolyData()
    PolyAddExtraFunctions(line)
    line.SetNumpyPoints(points)
    line.SetNumpyPolys(lines)
    return line


def SetNumpyPolys(mesh, array, deep=True):
    """ Convenience function to set polys using a numpy array """

    id_array = VN.numpy_to_vtkIdTypeArray(array, deep=deep)
    vtkcells = vtk.vtkCellArray()
    vtkcells.SetCells(array.shape[0], id_array)
    mesh.SetPolys(vtkcells)
    

def MeshfromVF(points, triangles_in, clean=True):
    """ Generates mesh from points and triangles """
    
    # Add face padding
    if triangles_in.shape[1] == 3:
        triangles = np.empty((triangles_in.shape[0], 4), dtype=np.int64)
        triangles[:, -3:] = triangles_in
        triangles[:, 0] = 3
        
    else:
        triangles = triangles_in
    
    # Data checking
    if not points.flags['C_CONTIGUOUS']:
        points = np.ascontiguousarray(points)
        
    if not triangles.flags['C_CONTIGUOUS'] or triangles.dtype != 'int64':
        triangles = np.ascontiguousarray(triangles, 'int64')

    # Convert to vtk objects
    vtkpoints = vtk.vtkPoints()
    vtkpoints.SetData(VN.numpy_to_vtk(points, deep=True))
    
    # Convert to a vtk array
    vtkcells = vtk.vtkCellArray()
    vtkcells.SetCells(triangles.shape[0], VN.numpy_to_vtkIdTypeArray(triangles,
                                                                     deep=True))
    
    # Create polydata object
    mesh = vtk.vtkPolyData()
    PolyAddExtraFunctions(mesh)
    mesh.SetPoints(vtkpoints)
    mesh.SetPolys(vtkcells)
    
    # return cleaned mesh
    if clean:
        return CleanMesh(mesh)    
    else:
        return mesh
    
    
def CreateVectorPolyData(orig, vec):
    """ Creates a vtkPolyData object composed of vectors """
        
    
    # Create vtk points and cells objects
    vpts = vtk.vtkPoints()
    vpts.SetData(VN.numpy_to_vtk(np.ascontiguousarray(orig), deep=True))
    
    npts = orig.shape[0]
    cells = np.hstack((np.ones((npts, 1), 'int'),
                       np.arange(npts).reshape((-1, 1))))
                       
    cells = np.ascontiguousarray(cells)
    vcells = vtk.vtkCellArray()
    vcells.SetCells(npts, VN.numpy_to_vtkIdTypeArray(cells, deep=True))
    
    # Create vtkPolyData object
    pdata = vtk.vtkPolyData()
    pdata.SetPoints(vpts);
    pdata.SetVerts(vcells)
    
    # Add vectors to polydata
    name='vectors'
    vtkfloat = VN.numpy_to_vtk(np.ascontiguousarray(vec), deep=True)
    vtkfloat.SetName(name)
    pdata.GetPointData().AddArray(vtkfloat)
    pdata.GetPointData().SetActiveVectors(name)
    
    # Add magnitude of vectors to polydata
    name='mag'
    scalars = (vec*vec).sum(1)**0.5
    vtkfloat = VN.numpy_to_vtk(np.ascontiguousarray(scalars), deep=True)
    vtkfloat.SetName(name)
    pdata.GetPointData().AddArray(vtkfloat)
    pdata.GetPointData().SetActiveScalars(name)
    
    return pdata    
    
    
def ApplyTransformation(mesh, trans):
    """
    Apply vtk transformation to vtk mesh.  Returns a copied transformed mesh
    """

    # Convert 4x4 matrix to a transformation object if applicable
    if trans.IsA('vtkMatrix4x4'):
        transform = vtk.vtkTransform()
        transform.SetMatrix(trans)
        trans = transform
    
    if mesh.IsA('vtkUnstructuredGrid'):
        transformFilter = vtk.vtkTransformFilter()
    else:
        transformFilter = vtk.vtkTransformPolyDataFilter()
        
    transformFilter.SetOutputPointsPrecision(vtk.vtkAlgorithm.DOUBLE_PRECISION)
    SetVTKInput(transformFilter, mesh)
    transformFilter.SetTransform(trans)    
    transformFilter.Update()
    tout = transformFilter.GetOutput()
    
    # add functions and return
    AddFunctions(tout)
    return tout
    
    
def ApplyTransformationInPlace(mesh, trans):
    """
    Apply a transformation in place to PolyData or an unstructred grid

    """

    # Convert 4x4 matrix to a transformation object if applicable
    if trans.IsA('vtkMatrix4x4'):
        transform = vtk.vtkTransform()
        transform.SetMatrix(trans)
        trans = transform
    
    if mesh.IsA('vtkUnstructuredGrid'):
        transformFilter = vtk.vtkTransformFilter()
    else:
        transformFilter = vtk.vtkTransformPolyDataFilter()
        
    transformFilter.SetOutputPointsPrecision(vtk.vtkAlgorithm.DOUBLE_PRECISION)
    SetVTKInput(transformFilter, mesh)
    transformFilter.SetTransform(trans)    
    transformFilter.Update()
    tmesh = transformFilter.GetOutput()    
    
    # Apply to mesh
    SetPoints(mesh, GetPoints(tmesh))
    
    
def ApplyTransformationSolid(grid, trans):
    """ Apply vtk transformation to vtk mesh """

    # Convert 4x4 matrix to a transformation object if applicable
    if trans.IsA('vtkMatrix4x4'):
        transform = vtk.vtkTransform()
        transform.SetMatrix(trans)
        trans = transform
    
    transformFilter = vtk.vtkTransformFilter()
    SetVTKInput(transformFilter, grid)
    transformFilter.SetTransform(trans)    
    transformFilter.Update()
    return transformFilter.GetOutput()
    
    
def AlignMesh(fixed, moving, nlan=[], niter=[], mmdist=[]):
    """ Aligns the moving mesh to the fixed mesh """
    
    # Perform ICP
    vtkICP = vtk.vtkIterativeClosestPointTransform()
    vtkICP.SetSource(moving)
    vtkICP.SetTarget(fixed)
    vtkICP.GetLandmarkTransform().SetModeToRigidBody()
    
    # Set settings if applicable
    if nlan:
        vtkICP.SetMaximumNumberOfLandmarks(nlan)
    if niter:
        vtkICP.SetMaximumNumberOfIterations(niter)
    if mmdist:
        vtkICP.SetMaximumMeanDistance(mmdist)
    vtkICP.StartByMatchingCentroidsOn()
    vtkICP.Modified()
    vtkICP.Update()
    
    return vtkICP

    
###############################################################################
# File I/O
###############################################################################
def WriteGrid(grid, filename, binary=True):
    """
    Writes a structured or unstructured grid to disk.
        
    The file extension will select the type of writer to use.  *.vtk will use
    the legacy writer, while *.vtu will select the VTK XML writer.

    Object types other than vtk structured or unstructured grids will raise 
    an error.
    
    Parameters
    ----------
    grid : vtkUnstructuredGrid or vtkStructuredGrid
        Structured or unstructured vtk grid.
    filename : str
        Filename of grid to be written.  The file extension will select the 
        type of writer to use.  *.vtk will use the legacy writer, while *.vtu 
        will select the VTK XML writer
    binary : bool, optional
        Writes as a binary file by default.  Set to False to write ASCII
        
    Returns
    -------
    None
        
    Notes
    -----
    Binary files write much faster than ASCII, but binary files written on one 
    system may not be readable on other systems.  Binary can only be selected
    for the legacy writer.
    
    
    """
    
    # Check file extention
    if '.vtu' in filename:
        legacy_writer = False
    elif '.vtk' in filename:
        legacy_writer = True
    else:
        raise Exception('Extension should be either ".vtu" (xml) or ".vtk" (legacy)')
    
    
    # Create writer
    if isinstance(grid, vtk.vtkUnstructuredGrid):
        if legacy_writer:
            writer = vtk.vtkUnstructuredGridWriter()
        else:
            writer = vtk.vtkXMLUnstructuredGridWriter()

    elif isinstance(grid, vtk.vtkStructuredGrid):
        if legacy_writer:
            writer = vtk.vtkStructuredGridWriter()
        else:
            writer = vtk.vtkXMLStructuredGridWriter()

    elif isinstance(grid, vtk.vtkPolyData):
        raise Exception('Use WriteMesh for vtkPolydata objects')
    
    else:
        raise Exception('Cannot write this object to file using WriteGrid')
    
    # Write
    writer.SetFileName(filename)
    writer.SetInputData(grid)
    if binary and legacy_writer:
        writer.SetFileTypeToBinary
    writer.Write()


def LoadGrid(filename, structured=False):
    """
    LoadGrid(filename)
    Load a vtkUnstructuredGrid or vtkStructuredGrid from a file.
    
    The file extension will select the type of reader to use.  *.vtk will use
    the legacy reader, while *.vtu will select the VTK XML reader.
    
    Parameters
    ----------
    filename : str
        Filename of grid to be loaded.  The file extension will select the type
        of reader to use.  *.vtk will use the legacy reader, while *.vtu will 
        select the VTK XML reader.
    structured : bool, optional
        By detault, this function reads an unstructured grid.  Setting this to
        true allows a structured grid to be read
        
    Returns
    -------
    grid : vtkUnstructuredGrid or vtkStructuredGrid object
        Grid from vtk reader.
        
    """
    # Check file extention
    if '.vtu' in filename:
        legacy_writer = False
    elif '.vtk' in filename:
        legacy_writer = True
    else:
        raise Exception('Extension should be either ".vtu" (xml) or ".vtk" (legacy)')
        
    # Create reader
    if structured:
        if legacy_writer:
            reader = vtk.vtkStructuredGridReader()
        else:
            reader = vtk.vtkXMLStructuredGridReader()
    else:
        if legacy_writer:
            reader = vtk.vtkUnstructuredGridReader()
        else:
            reader = vtk.vtkXMLUnstructuredGridReader()

    # load file
    reader.SetFileName(filename)
    reader.Update()
    grid = reader.GetOutput()
    AddFunctions(grid)

    return grid


def WriteMesh(mesh, filename, ftype=None, binary=True):
    """
    WriteMesh(filename)
    Writes a surface mesh to disk.
    
    Written file may be an ASCII or binary ply, stl, or vtk mesh file.  Email
    author to suggest support for another filetype.
    
    Parameters
    ----------
    filename : str
        Filename of mesh to be written.  Filetype is inferred from the
        extension of the filename unless overridden with ftype.  Can be one of
        the following types (.ply, .stl, .vtk)
        
    ftype : str, optional
        Filetype.  Inferred from filename unless specified with a three
        character string.  Can be one of the following: 'ply',  'stl', or 'vtk'
        
    Returns
    -------
    None
        
    Notes
    -----
    Binary files write much faster than ASCII.
    
    
    """
    
    if not ftype:
        ftype = filename[-3:]
    
    # Get filetype
    if ftype == 'ply':
        writer = vtk.vtkPLYWriter()
    elif ftype == 'stl':
        writer = vtk.vtkSTLWriter()
    elif ftype == 'vtk':
        writer = vtk.vtkPolyDataWriter()
    else:
        raise Exception('Unknown file type')
        
    # Write
    writer.SetFileName(filename)
    SetVTKInput(writer, mesh)
    if binary:
        writer.SetFileTypeToBinary()
    else:
        writer.SetFileTypeToASCII()
    writer.Write()
    
    
def LoadMesh(filename):
    """
    LoadMesh(filename)
    Load a surface mesh from a mesh file.
    
    Mesh file may be an ASCII or binary ply, stl, g3d, or vtk mesh file.  Email
    author to suggest support for another filetype.
    
    Parameters
    ----------
    filename : str
        Filename of mesh to be loaded.  Filetype is inferred from the extension
        of the filename
        
    Returns
    -------
    mesh : vtkPolydata object
        VTK surface mesh object.
        
    Notes
    -----
    Binary files load much faster than ASCII.
    
    Examples
    --------
    >>> from vtkInterface import examples
    >>> import vtkInterface
    
    >>> mesh = vtkInterface.LoadMesh(examples.planefile)
    >>> mesh.Plot()
    
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
        
    elif fext == 'g3d': # Don't use vtk reader
        v, f = ReadG3D(filename)
        v /= 25.4 # convert to inches
        return MeshfromVF(v, f, False)
        
    elif fext == 'vtk':
        reader = vtk.vtkPolyDataReader()
        
    else:
        raise Exception('Unknown file type')
    
    # Load file
    reader.SetFileName(filename) 
    reader.Update()
    mesh = reader.GetOutput()
    PolyAddExtraFunctions(mesh)
    return mesh
    
    
def ReadG3D(filename):
    """ Reads data from a *.g3d file and outputs points and triangle arrays
    """
    
    # open file and skip header and part of triangle header
    f = open(filename)
    f.seek(96 + 144)
    
    # Number of points, poitner to first point, and size of point
    pt_dat = np.fromstring(f.read(4*3), dtype=np.uint32)
    
    # Number of triangles, pointer to first triangle, and size of triangle
    tri_dat = np.fromstring(f.read(4*3), dtype=np.uint32)
    
    # Read in points
    f.seek(pt_dat[1]) # Seek to start of point data
    points = np.zeros((pt_dat[0], 3))
    for i in range(pt_dat[0]):
        points[i] = np.fromstring(f.read(24), dtype=np.float64)
        f.seek(f.tell() + 4)
    
    # Read in triangles
    tri = np.fromstring(f.read(12*tri_dat[0]), dtype=np.uint32)
    triangles = np.zeros((tri_dat[0], 4), dtype=np.int64)
    triangles[:, 0] = 3
    triangles[:, 1:] = tri.reshape((-1, 3))
    
    # Close file
    f.close()
    
    return points, triangles


#==============================================================================
# 
#==============================================================================
#def Sphere(radius, center=[0, 0, 0]):
#    source = vtk.vtkSphereSource()
#    source.SetCenter(0,0,0)
#    source.SetRadius(5.0)


