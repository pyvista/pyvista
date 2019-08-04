from __future__ import absolute_import
from . import dataset_adapter as dsa
import numpy
from vtk.util import numpy_support
import vtk

def _cell_derivatives (narray, dataset, attribute_type, filter):
    if not dataset :
       raise RuntimeError('Need a dataset to compute _cell_derivatives.')

    # Reshape n dimensional vector to n by 1 matrix
    if len(narray.shape) == 1 :
       narray = narray.reshape((narray.shape[0], 1))

    ncomp = narray.shape[1]
    if attribute_type == 'scalars' and ncomp != 1 :
       raise RuntimeError('This function expects scalars. ' +
                           'Input shape ' + str(narray.shape))
    if attribute_type == 'vectors' and ncomp != 3 :
       raise RuntimeError('This function expects vectors. ' +
                           'Input shape ' + str(narray.shape))

    # numpy_to_vtk converts only contiguous arrays
    if not narray.flags.contiguous : narray = narray.copy()
    varray = numpy_support.numpy_to_vtk(narray)

    if attribute_type == 'scalars': varray.SetName('scalars')
    else : varray.SetName('vectors')

    # create a dataset with only our array but the same geometry/topology
    ds = dataset.NewInstance()
    ds.UnRegister(None)
    ds.CopyStructure(dataset.VTKObject)

    if dsa.ArrayAssociation.FIELD == narray.Association :
       raise RuntimeError('Unknown data association. Data should be associated with points or cells.')

    if dsa.ArrayAssociation.POINT == narray.Association :
       # Work on point data
       if narray.shape[0] != dataset.GetNumberOfPoints() :
          raise RuntimeError('The number of points does not match the number of tuples in the array')
       if attribute_type == 'scalars': ds.GetPointData().SetScalars(varray)
       else : ds.GetPointData().SetVectors(varray)
    elif dsa.ArrayAssociation.CELL == narray.Association :
       # Work on cell data
       if narray.shape[0] != dataset.GetNumberOfCells() :
          raise RuntimeError('The number of does not match the number of tuples in the array')

       # Since vtkCellDerivatives only works with point data, we need to convert
       # the cell data to point data first.

       ds2 = dataset.NewInstance()
       ds2.UnRegister(None)
       ds2.CopyStructure(dataset.VTKObject)

       if attribute_type == 'scalars' : ds2.GetCellData().SetScalars(varray)
       else : ds2.GetCellData().SetVectors(varray)

       c2p = vtk.vtkCellDataToPointData()
       c2p.SetInputData(ds2)
       c2p.Update()

       # Set the output to the ds dataset
       if attribute_type == 'scalars':
          ds.GetPointData().SetScalars(c2p.GetOutput().GetPointData().GetScalars())
       else:
          ds.GetPointData().SetVectors(c2p.GetOutput().GetPointData().GetVectors())

    filter.SetInputData(ds)

    if dsa.ArrayAssociation.POINT == narray.Association :
       # Since the data is associated with cell and the query is on points
       # we have to convert to point data before returning
       c2p = vtk.vtkCellDataToPointData()
       c2p.SetInputConnection(filter.GetOutputPort())
       c2p.Update()
       return c2p.GetOutput().GetPointData()
    elif dsa.ArrayAssociation.CELL == narray.Association :
       filter.Update()
       return filter.GetOutput().GetCellData()
    else :
       # We shall never reach here
       raise RuntimeError('Unknown data association. Data should be associated with points or cells.')

def _cell_quality (dataset, quality) :
    if not dataset : raise RuntimeError('Need a dataset to compute _cell_quality')

    # create a dataset with only our array but the same geometry/topology
    ds = dataset.NewInstance()
    ds.UnRegister(None)
    ds.CopyStructure(dataset.VTKObject)

    filter = vtk.vtkCellQuality()
    filter.SetInputData(ds)

    if   "area"         == quality : filter.SetQualityMeasureToArea()
    elif "aspect"       == quality : filter.SetQualityMeasureToAspectRatio()
    elif "aspect_gamma" == quality : filter.SetQualityMeasureToAspectGamma()
    elif "condition"    == quality : filter.SetQualityMeasureToCondition()
    elif "diagonal"     == quality : filter.SetQualityMeasureToDiagonal()
    elif "jacobian"     == quality : filter.SetQualityMeasureToJacobian()
    elif "max_angle"    == quality : filter.SetQualityMeasureToMaxAngle()
    elif "shear"        == quality : filter.SetQualityMeasureToShear()
    elif "skew"         == quality : filter.SetQualityMeasureToSkew()
    elif "min_angle"    == quality : filter.SetQualityMeasureToMinAngle()
    elif "volume"       == quality : filter.SetQualityMeasureToVolume()
    else : raise RuntimeError('Unknown cell quality ['+quality+'].')

    filter.Update()

    varray = filter.GetOutput().GetCellData().GetArray("CellQuality")
    ans = dsa.vtkDataArrayToVTKArray(varray, dataset)

    # The association information has been lost over the vtk filter
    # we must reconstruct it otherwise lower pipeline will be broken.
    ans.Association = dsa.ArrayAssociation.CELL

    return ans

def _matrix_math_filter (narray, operation) :
    if operation not in ['Determinant', 'Inverse', 'Eigenvalue', 'Eigenvector'] :
       raise RuntimeError('Unknown quality measure ['+operation+']'+
                          ' Supported are [Determinant, Inverse, Eigenvalue, Eigenvector]')

    if narray.ndim != 3 :
       raise RuntimeError(operation+' only works for an array of matrices(3D array).'+
                           ' Input shape ' + str(narray.shape))
    elif narray.shape[1] != narray.shape[2] :
       raise RuntimeError(operation+' requires an array of 2D square matrices.' +
                           ' Input shape ' + str(narray.shape))

    # numpy_to_vtk converts only contiguous arrays
    if not narray.flags.contiguous : narray = narray.copy()

    # Reshape is necessary because numpy_support.numpy_to_vtk only works with 2D or
    # less arrays.
    nrows = narray.shape[0]
    ncols = narray.shape[1] * narray.shape[2]
    narray = narray.reshape(nrows, ncols)

    ds = vtk.vtkImageData()
    ds.SetDimensions(nrows, 1, 1)

    varray = numpy_support.numpy_to_vtk(narray)
    varray.SetName('tensors')
    ds.GetPointData().SetTensors(varray)

    filter = vtk.vtkMatrixMathFilter()

    if   operation == 'Determinant'  : filter.SetOperationToDeterminant()
    elif operation == 'Inverse'      : filter.SetOperationToInverse()
    elif operation == 'Eigenvalue'   : filter.SetOperationToEigenvalue()
    elif operation == 'Eigenvector'  : filter.SetOperationToEigenvector()

    filter.SetInputData(ds)
    filter.Update()

    varray = filter.GetOutput().GetPointData().GetArray(operation)

    ans = dsa.vtkDataArrayToVTKArray(varray)

    # The association information has been lost over the vtk filter
    # we must reconstruct it otherwise lower pipeline will be broken.
    ans.Association = narray.Association
    ans.DataSet = narray.DataSet

    return ans

# Python interfaces
def abs (narray) :
    "Returns the absolute values of an array of scalars/vectors/tensors."
    return numpy.abs(narray)

def all (narray, axis=None):
    "Returns the min value of an array of scalars/vectors/tensors."
    if narray is dsa.NoneArray:
      return dsa.NoneArray
    ans = numpy.all(numpy.array(narray), axis)
    return ans

def area (dataset) :
    "Returns the surface area of each cell in a mesh."
    return _cell_quality(dataset, "area")

def aspect (dataset) :
    "Returns the aspect ratio of each cell in a mesh."
    return _cell_quality(dataset, "aspect")

def aspect_gamma (dataset) :
    "Returns the aspect ratio gamma of each cell in a mesh."
    return _cell_quality(dataset, "aspect_gamma")

def condition (dataset) :
    "Returns the condition number of each cell in a mesh."
    return _cell_quality(dataset, "condition")

def cross (x, y) :
    "Return the cross product for two 3D vectors from two arrays of 3D vectors."
    if x is dsa.NoneArray or y is dsa.NoneArray:
      return dsa.NoneArray

    if x.ndim != y.ndim or x.shape != y.shape:
       raise RuntimeError('Both operands must have same dimension and shape.' +
                          ' Input shapes ' + str(x.shape) + ' and ' + str(y.shape))

    if x.ndim != 1 and x.ndim != 2 :
       raise RuntimeError('Cross only works for 3D vectors or an array of 3D vectors.' +
                          ' Input shapes ' + str(x.shape) + ' and ' + str(y.shape))

    if x.ndim == 1 and x.shape[0] != 3 :
       raise RuntimeError('Cross only works for 3D vectors.' +
                          ' Input shapes ' + str(x.shape) + ' and ' + str(y.shape))

    if x.ndim == 2 and x.shape[1] != 3 :
       raise RuntimeError('Cross only works for an array of 3D vectors.' +
                          'Input shapes ' + str(x.shape) + ' and ' + str(y.shape))

    return numpy.cross(x, y)

def curl (narray, dataset=None):
    "Returns the curl of an array of 3D vectors."
    if not dataset : dataset = narray.DataSet
    if not dataset : raise RuntimeError('Need a dataset to compute curl.')

    if narray.ndim != 2 or narray.shape[1] != 3 :
       raise RuntimeError('Curl only works with an array of 3D vectors.' +
                          'Input shape ' + str(narray.shape))

    cd = vtk.vtkCellDerivatives()
    cd.SetVectorModeToComputeVorticity()

    res = _cell_derivatives(narray, dataset, 'vectors', cd)

    retVal = res.GetVectors()
    retVal.SetName("vorticity")

    ans = dsa.vtkDataArrayToVTKArray(retVal, dataset)

    # The association information has been lost over the vtk filter
    # we must reconstruct it otherwise lower pipeline will be broken.
    ans.Association = narray.Association

    return ans

def divergence (narray, dataset=None):
    "Returns the divergence of an array of 3D vectors."
    if not dataset : dataset = narray.DataSet
    if not dataset : raise RuntimeError('Need a dataset to compute divergence')

    if narray.ndim != 2 or narray.shape[1] != 3 :
       raise RuntimeError('Divergence only works with an array of 3D vectors.' +
                          ' Input shape ' + str(narray.shape))

    g = gradient(narray, dataset)
    g = g.reshape(g.shape[0], 3, 3)

    a = dsa.VTKArray\
           (numpy.add.reduce(g.diagonal(axis1=1, axis2=2), 1), dataset=g.DataSet)
    try:
        a.Association = g.Association
    except AttributeError: pass
    return a


def det (narray) :
    "Returns the determinant of an array of 2D square matrices."
    return _matrix_math_filter(narray, "Determinant")

def determinant (narray) :
    "Returns the determinant of an array of 2D square matrices."
    return det(narray)

def diagonal (dataset) :
    "Returns the diagonal length of each cell in a dataset."
    return _cell_quality(dataset, "diagonal")

def dot (a1, a2):
    "Returns the dot product of two scalars/vectors of two array of scalars/vectors."
    if a1 is dsa.NoneArray or a2 is dsa.NoneArray:
      return dsa.NoneArray

    if a1.shape[1] != a2.shape[1] :
     raise RuntimeError('Dot product only works with vectors of same dimension.' +
                         ' Input shapes ' + str(a1.shape) + ' and ' + str(a2.shape))
    m = a1*a2
    va = dsa.VTKArray(numpy.add.reduce(m, 1))
    if hasattr(m, "Association"):
      va.Association = m.Association
    if a1.DataSet == a2.DataSet : va.DataSet = a1.DataSet
    return va

def eigenvalue (narray) :
    "Returns the eigenvalue of an array of 2D square matrices."
    return _matrix_math_filter(narray, "Eigenvalue")

def eigenvector (narray) :
    "Returns the eigenvector of an array of 2D square matrices."
    return _matrix_math_filter(narray, "Eigenvector")

def gradient(narray, dataset=None):
    "Returns the gradient of an array of scalars/vectors."
    if not dataset: dataset = narray.DataSet
    if not dataset: raise RuntimeError('Need a dataset to compute gradient')

    try:
      ncomp = narray.shape[1]
    except IndexError:
      ncomp = 1
    if ncomp != 1 and ncomp != 3:
       raise RuntimeError('Gradient only works with scalars (1 component) and vectors (3 component)' +
                          ' Input shape ' + str(narray.shape))

    cd = vtk.vtkCellDerivatives()
    if ncomp == 1 : attribute_type = 'scalars'
    else : attribute_type = 'vectors'

    res = _cell_derivatives(narray, dataset, attribute_type, cd)

    if ncomp == 1 : retVal = res.GetVectors()
    else : retVal = res.GetTensors()

    try:
        if narray.GetName() : retVal.SetName("gradient of " + narray.GetName())
        else : retVal.SetName("gradient")
    except AttributeError : retVal.SetName("gradient")

    ans = dsa.vtkDataArrayToVTKArray(retVal, dataset)

    # The association information has been lost over the vtk filter
    # we must reconstruct it otherwise lower pipeline will be broken.
    ans.Association = narray.Association

    return ans

def inv (narray) :
    "Returns the inverse an array of 2D square matrices."
    return _matrix_math_filter(narray, "Inverse")

def inverse (narray) :
    "Returns the inverse of an array of 2D square matrices."
    return inv(narray)

def jacobian (dataset) :
    "Returns the jacobian of an array of 2D square matrices."
    return _cell_quality(dataset, "jacobian")

def laplacian (narray, dataset=None) :
    "Returns the jacobian of an array of scalars."
    if not dataset : dataset = narray.DataSet
    if not dataset : raise RuntimeError('Need a dataset to compute laplacian')
    ans = gradient(narray, dataset)
    return divergence(ans)

def ln (narray) :
    "Returns the natural logarithm of an array of scalars/vectors/tensors."
    return numpy.log(narray)

def log (narray) :
    "Returns the natural logarithm of an array of scalars/vectors/tensors."
    return ln(narray)

def log10 (narray) :
    "Returns the base 10 logarithm of an array of scalars/vectors/tensors."
    return numpy.log10(narray)

def max (narray, axis=None):
    "Returns the maximum value of an array of scalars/vectors/tensors."
    if narray is dsa.NoneArray:
      return dsa.NoneArray
    ans = numpy.max(narray, axis)
#    if len(ans.shape) == 2 and ans.shape[0] == 3 and ans.shape[1] == 3: ans.reshape(9)
    return ans

def max_angle (dataset) :
    "Returns the maximum angle of each cell in a dataset."
    return _cell_quality(dataset, "max_angle")

def mag (a) :
    "Returns the magnigude of an array of scalars/vectors."
    return numpy.sqrt(dot(a, a))

def mean (narray, axis=None) :
    "Returns the mean value of an array of scalars/vectors/tensors."
    if narray is dsa.NoneArray:
      return dsa.NoneArray
    ans = numpy.mean(numpy.array(narray), axis)
#    if len(ans.shape) == 2 and ans.shape[0] == 3 and ans.shape[1] == 3: ans.reshape(9)
    return ans

def min (narray, axis=None):
    "Returns the min value of an array of scalars/vectors/tensors."
    if narray is dsa.NoneArray:
      return dsa.NoneArray
    ans = numpy.min(narray, axis)
#    if len(ans.shape) == 2 and ans.shape[0] == 3 and ans.shape[1] == 3: ans.reshape(9)
    return ans

def min_angle (dataset) :
    "Returns the minimum angle of each cell in a dataset."
    return _cell_quality(dataset, "min_angle")

def norm (a) :
    "Returns the normalized values of an array of scalars/vectors."
    return a/mag(a).reshape((a.shape[0], 1))

def shear (dataset) :
    "Returns the shear of each cell in a dataset."
    return _cell_quality(dataset, "shear")

def skew (dataset) :
    "Returns the skew of each cell in a dataset."
    return _cell_quality(dataset, "skew")

def strain (narray, dataset=None) :
    "Returns the strain of an array of 3D vectors."
    if not dataset : dataset = narray.DataSet
    if not dataset : raise RuntimeError('Need a dataset to compute strain')

    if 2 != narray.ndim or 3 != narray.shape[1] :
       raise RuntimeError('strain only works with an array of 3D vectors' +
                          'Input shape ' + str(narray.shape))

    cd = vtk.vtkCellDerivatives()
    cd.SetTensorModeToComputeStrain()

    res = _cell_derivatives(narray, dataset, 'vectors', cd)

    retVal = res.GetTensors()
    retVal.SetName("strain")

    ans = dsa.vtkDataArrayToVTKArray(retVal, dataset)

    # The association information has been lost over the vtk filter
    # we must reconstruct it otherwise lower pipeline will be broken.
    ans.Association = narray.Association

    return ans

def sum (narray, axis=None):
    "Returns the min value of an array of scalars/vectors/tensors."
    if narray is dsa.NoneArray:
      return dsa.NoneArray
    return numpy.sum(narray, axis)

def surface_normal (dataset) :
    "Returns the surface normal of each cell in a dataset."
    if not dataset : raise RuntimeError('Need a dataset to compute surface_normal')

    ds = dataset.NewInstance()
    ds.UnRegister(None)
    ds.CopyStructure(dataset.VTKObject)

    filter = vtk.vtkPolyDataNormals()
    filter.SetInputData(ds)
    filter.ComputeCellNormalsOn()
    filter.ComputePointNormalsOff()

    filter.SetFeatureAngle(180)
    filter.SplittingOff()
    filter.ConsistencyOff()
    filter.AutoOrientNormalsOff()
    filter.FlipNormalsOff()
    filter.NonManifoldTraversalOff()
    filter.Update()

    varray = filter.GetOutput().GetCellData().GetNormals()
    ans = dsa.vtkDataArrayToVTKArray(varray, dataset)

    # The association information has been lost over the vtk filter
    # we must reconstruct it otherwise lower pipeline will be broken.
    ans.Association = dsa.ArrayAssociation.CELL

    return ans

def trace (narray) :
    "Returns the trace of an array of 2D square matrices."
    ax1 = 0
    ax2 = 1
    if narray.ndim > 2 :
       ax1 = 1
       ax2 = 2
    return numpy.trace(narray, axis1=ax1, axis2=ax2)

def var (narray, axis=None) :
    "Returns the mean value of an array of scalars/vectors/tensors."
    if narray is dsa.NoneArray:
      return dsa.NoneArray
    return numpy.var(narray, axis)

def volume (dataset) :
    "Returns the volume of each cell in a dataset."
    #def _cell_quality (dataset, quality) :
    if not dataset : raise RuntimeError('Need a dataset to compute volume')

    # create a dataset with only our array but the same geometry/topology
    ds = dataset.NewInstance()
    ds.UnRegister(None)
    ds.CopyStructure(dataset.VTKObject)

    filter = vtk.vtkCellSizeFilter()
    filter.SetInputData(ds)
    filter.ComputeVertexCountOff()
    filter.ComputeLengthOff()
    filter.ComputeAreaOff()
    filter.Update()

    varray = filter.GetOutput().GetCellData().GetArray("Volume")
    varray.SetName("CellQuality")
    ans = dsa.vtkDataArrayToVTKArray(varray, dataset)

    # The association information has been lost over the vtk filter
    # we must reconstruct it otherwise lower pipeline will be broken.
    ans.Association = dsa.ArrayAssociation.CELL

    return ans

def vorticity(narray, dataset=None):
    "Returns the vorticity/curl of an array of 3D vectors."
    return curl(narray, dataset)

def vertex_normal (dataset) :
    "Returns the vertex normal of each point in a dataset."
    if not dataset : raise RuntimeError('Need a dataset to compute vertex_normal')

    ds = dataset.NewInstance()
    ds.UnRegister(None)
    ds.CopyStructure(dataset.VTKObject)

    filter = vtk.vtkPolyDataNormals()
    filter.SetInputData(ds)
    filter.ComputeCellNormalsOff()
    filter.ComputePointNormalsOn()

    filter.SetFeatureAngle(180)
    filter.SplittingOff()
    filter.ConsistencyOff()
    filter.AutoOrientNormalsOff()
    filter.FlipNormalsOff()
    filter.NonManifoldTraversalOff()
    filter.Update()

    varray = filter.GetOutput().GetPointData().GetNormals()
    ans = dsa.vtkDataArrayToVTKArray(varray, dataset)

    # The association information has been lost over the vtk filter
    # we must reconstruct it otherwise lower pipeline will be broken.
    ans.Association = dsa.ArrayAssociation.POINT

    return ans

def make_vector(ax, ay, az=None):
    if ax is dsa.NoneArray or ay is dsa.NoneArray or ay is dsa.NoneArray:
      return dsa.NoneArray

    if len(ax.shape) != 1 or len(ay.shape) != 1 or (az is not None and len(az.shape) != 1):
        raise ValueError("Can only merge 1D arrays")

    if az is None:
        az = numpy.zeros(ax.shape)
    v = numpy.vstack([ax, ay, az]).transpose().view(dsa.VTKArray)
    # Copy defaults from first array. The user can always
    # overwrite this
    try:
        v.DataSet = ax.DataSet
    except AttributeError: pass
    try:
        v.Association = ax.Association
    except AttributeError: pass
    return v
