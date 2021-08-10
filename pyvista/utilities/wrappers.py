"""Wrapper mapping."""
import pyvista

_wrappers = {
    'vtkExplicitStructuredGrid': pyvista.ExplicitStructuredGrid,
    'vtkUnstructuredGrid': pyvista.UnstructuredGrid,
    'vtkRectilinearGrid': pyvista.RectilinearGrid,
    'vtkStructuredGrid': pyvista.StructuredGrid,
    'vtkPolyData': pyvista.PolyData,
    'vtkImageData': pyvista.UniformGrid,
    'vtkStructuredPoints': pyvista.UniformGrid,
    'vtkMultiBlockDataSet': pyvista.MultiBlock,
    'vtkTable': pyvista.Table,
    # 'vtkParametricSpline': pyvista.Spline,
}
