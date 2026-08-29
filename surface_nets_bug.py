"""Pure-VTK reproducer for the frog_tissues vtkSurfaceNets3D 9.6 vs 9.7 regression.
No pyvista involved except that the input .vti was originally saved via pyvista's
examples.load_frog_tissues() -- reading/writing it here is plain VTK.
"""

from __future__ import annotations

import sys

from vtkmodules.util.vtkConstants import VTK_UNSIGNED_CHAR
from vtkmodules.vtkCommonCore import vtkVersion
from vtkmodules.vtkCommonDataModel import vtkDataObject
from vtkmodules.vtkCommonDataModel import vtkImageData
from vtkmodules.vtkFiltersCore import vtkFeatureEdges
from vtkmodules.vtkFiltersCore import vtkSurfaceNets3D
from vtkmodules.vtkFiltersCore import vtkThreshold
from vtkmodules.vtkFiltersGeometry import vtkGeometryFilter
from vtkmodules.vtkImagingStencil import vtkImageStencil
from vtkmodules.vtkImagingStencil import vtkPolyDataToImageStencil
import vtkmodules.vtkInteractionStyle
from vtkmodules.vtkIOXML import vtkXMLImageDataReader
from vtkmodules.vtkRenderingCore import vtkActor
from vtkmodules.vtkRenderingCore import vtkPolyDataMapper
from vtkmodules.vtkRenderingCore import vtkRenderer
from vtkmodules.vtkRenderingCore import vtkRenderWindow
from vtkmodules.vtkRenderingCore import vtkRenderWindowInteractor
from vtkmodules.vtkRenderingCore import vtkTextActor
import vtkmodules.vtkRenderingOpenGL2


def read(filename):
    reader = vtkXMLImageDataReader()
    reader.SetFileName(filename)
    reader.Update()
    return reader.GetOutput()


def surface_nets(image, smoothing=False):
    alg = vtkSurfaceNets3D()
    alg.SetInputData(image)
    if smoothing:
        alg.SmoothingOn()
    else:
        alg.SmoothingOff()
    alg.Update()
    return alg.GetOutput()


def edge_count(poly, boundary, non_manifold):
    fe = vtkFeatureEdges()
    fe.SetInputData(poly)
    fe.SetBoundaryEdges(boundary)
    fe.SetNonManifoldEdges(non_manifold)
    fe.SetFeatureEdges(False)
    fe.SetManifoldEdges(False)
    fe.Update()
    return fe.GetOutput().GetNumberOfCells()


def print_surface_info(name, poly):
    print(f'{name} info:')
    print('  n_points:', poly.GetNumberOfPoints())
    print('  n_cells:', poly.GetNumberOfCells())
    print('  open boundary edges:', edge_count(poly, True, False))
    print('  non-manifold edges:', edge_count(poly, False, True))


# --- Voxelize: vtkPolyDataToImageStencil -> vtkImageStencil -> points-to-cells -> threshold ---


def threshold_cells(dataset, array_name, lower, upper):
    threshold = vtkThreshold()
    threshold.SetInputData(dataset)
    threshold.SetInputArrayToProcess(0, 0, 0, vtkDataObject.FIELD_ASSOCIATION_CELLS, array_name)
    threshold.SetThresholdFunction(vtkThreshold.THRESHOLD_BETWEEN)
    threshold.SetLowerThreshold(lower)
    threshold.SetUpperThreshold(upper)
    threshold.Update()
    return threshold.GetOutput()


def poly_data_to_image_data(poly, reference_image, foreground_value=1, background_value=0):
    poly_to_stencil = vtkPolyDataToImageStencil()
    poly_to_stencil.SetInputData(poly)
    poly_to_stencil.SetOutputSpacing(*reference_image.GetSpacing())
    poly_to_stencil.SetOutputOrigin(*reference_image.GetOrigin())
    poly_to_stencil.SetOutputWholeExtent(reference_image.GetExtent())
    poly_to_stencil.Update()

    binary_mask = vtkImageData()
    binary_mask.SetExtent(reference_image.GetExtent())
    binary_mask.SetSpacing(reference_image.GetSpacing())
    binary_mask.SetOrigin(reference_image.GetOrigin())
    binary_mask.AllocateScalars(VTK_UNSIGNED_CHAR, 1)
    binary_mask.GetPointData().GetScalars().SetName('mask')
    binary_mask.GetPointData().GetScalars().Fill(background_value)

    stencil = vtkImageStencil()
    stencil.SetInputData(binary_mask)
    stencil.SetStencilConnection(poly_to_stencil.GetOutputPort())
    stencil.ReverseStencilOn()
    stencil.SetBackgroundValue(foreground_value)
    stencil.Update()
    return stencil.GetOutput()


def points_to_cells(image_data):
    """Re-mesh point-sampled image data as cell-sampled image data, i.e. pyvista's
    points_to_cells(): the container grows by one point per axis and the original
    point data becomes cell data on the new grid, so voxels are represented as cells
    rather than points.
    """
    dims = image_data.GetDimensions()
    spacing = image_data.GetSpacing()
    origin = image_data.GetOrigin()

    cells_image = vtkImageData()
    cells_image.SetDimensions(dims[0] + 1, dims[1] + 1, dims[2] + 1)
    cells_image.SetSpacing(spacing)
    cells_image.SetOrigin(
        origin[0] - spacing[0] / 2,
        origin[1] - spacing[1] / 2,
        origin[2] - spacing[2] / 2,
    )
    cells_image.GetCellData().SetScalars(image_data.GetPointData().GetScalars())
    return cells_image


def voxelize(poly, reference_image):
    mask = poly_data_to_image_data(poly, reference_image)
    voxel_cells = points_to_cells(mask)
    cells = threshold_cells(voxel_cells, 'mask', 0.5, 255)

    geom = vtkGeometryFilter()
    geom.SetInputData(cells)
    geom.Update()
    return geom.GetOutput()


def plot_comparison(poly_left, poly_right):
    """Side-by-side view: raw SurfaceNets output (colored by BoundaryLabels) on the
    left, the voxelized (stencil -> points_to_cells -> threshold -> surface) result
    on the right. Both should look the same if the stencil is robust to the input
    mesh's topology.
    """
    bl_arr = poly_left.GetCellData().GetArray('BoundaryLabels')

    left_mapper = vtkPolyDataMapper()
    left_mapper.SetInputData(poly_left)
    left_mapper.SetScalarModeToUseCellData()
    left_mapper.SelectColorArray('BoundaryLabels')
    left_mapper.SetArrayComponent(0)
    left_mapper.SetColorModeToMapScalars()
    left_mapper.ScalarVisibilityOn()
    left_mapper.SetScalarRange(bl_arr.GetRange(0))
    left_actor = vtkActor()
    left_actor.SetMapper(left_mapper)

    right_mapper = vtkPolyDataMapper()
    right_mapper.SetInputData(poly_right)
    right_mapper.ScalarVisibilityOff()
    right_actor = vtkActor()
    right_actor.SetMapper(right_mapper)
    right_actor.GetProperty().SetColor(0.2, 0.4, 1.0)  # blue, like the voxel cells

    render_window = vtkRenderWindow()
    render_window.SetSize(1600, 600)

    bounds = poly_left.GetBounds()
    center = poly_left.GetCenter()

    for viewport, actor, label in (
        ((0.0, 0.0, 0.5, 1.0), left_actor, 'SurfaceNets (raw)'),
        ((0.5, 0.0, 1.0, 1.0), right_actor, 'Voxelized (stencil + threshold)'),
    ):
        renderer = vtkRenderer()
        renderer.SetViewport(*viewport)
        renderer.AddActor(actor)
        renderer.SetBackground(0.2, 0.3, 0.4)
        render_window.AddRenderer(renderer)

        text = vtkTextActor()
        text.SetInput(label)
        text.GetTextProperty().SetFontSize(18)
        text.SetPosition(10, 10)
        renderer.AddViewProp(text)

        camera = renderer.GetActiveCamera()
        camera.SetPosition(
            center[0],
            center[1],
            bounds[5] + (bounds[5] - bounds[4]) * 4,
        )
        camera.SetFocalPoint(center)
        camera.SetViewUp(0, 1, 0)
        renderer.ResetCamera()
        camera.Zoom(1.3)
        renderer.ResetCameraClippingRange()

    interactor = vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)

    render_window.Render()
    interactor.Start()


def main():
    print('VTK version:', vtkVersion.GetVTKVersion())

    smoothing = '--smooth' in sys.argv

    image = read('frog_tissues.vti')
    poly = surface_nets(image, smoothing=smoothing)

    print_surface_info('surface_net', poly)

    voxel_surface = voxelize(poly, image)
    print_surface_info('voxel_surface', voxel_surface)

    plot_comparison(poly, voxel_surface)


if __name__ == '__main__':
    main()
