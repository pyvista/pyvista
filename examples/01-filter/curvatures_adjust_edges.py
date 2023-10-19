import math

import numpy as np
from vtk.util import numpy_support
from vtkmodules.numpy_interface import dataset_adapter as dsa
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkCommonCore import VTK_DOUBLE
from vtkmodules.vtkFiltersCore import vtkFeatureEdges, vtkIdFilter
from vtkmodules.vtkFiltersGeneral import vtkCurvatures
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera
from vtkmodules.vtkInteractionWidgets import vtkCameraOrientationWidget
from vtkmodules.vtkRenderingAnnotation import vtkScalarBarActor
from vtkmodules.vtkRenderingCore import (
    vtkActor2D,
    vtkPolyDataMapper,
    vtkRenderWindow,
    vtkRenderWindowInteractor,
    vtkTextMapper,
)

import pyvista as pv


def main(argv):
    desired_surface = 'RandomHills'
    source = get_source(desired_surface)
    if not source:
        print('The surface is not available.')
        return

    gc = vtkCurvatures()
    gc.SetInputData(source)
    gc.SetCurvatureTypeToGaussian()
    gc.Update()
    if desired_surface in ['RandomHills']:
        adjust_edge_curvatures(gc.GetOutput(), 'Gauss_Curvature')
    source.GetPointData().AddArray(
        gc.GetOutput().GetPointData().GetAbstractArray('Gauss_Curvature')
    )

    mc = vtkCurvatures()
    mc.SetInputData(source)
    mc.SetCurvatureTypeToMean()
    mc.Update()
    if desired_surface in ['RandomHills']:
        adjust_edge_curvatures(mc.GetOutput(), 'Mean_Curvature')
    source.GetPointData().AddArray(mc.GetOutput().GetPointData().GetAbstractArray('Mean_Curvature'))

    # Let's visualise what we have done.

    colors = vtkNamedColors()
    colors.SetColor("ParaViewBkg", [82, 87, 110, 255])

    window_width = 1024
    window_height = 512

    ren_win = vtkRenderWindow()
    ren_win.SetSize(window_width, window_height)
    iren = vtkRenderWindowInteractor()
    iren.SetRenderWindow(ren_win)
    style = vtkInteractorStyleTrackballCamera()
    iren.SetInteractorStyle(style)

    # Create a common text property.
    text_property = pv.TextProperty()
    text_property.font_size = 24
    text_property.justification_horizontal = "center"

    lut = pv.LookupTable('coolwarm', n_values=256)

    # Define viewport ranges
    xmins = [0, 0.5]
    xmaxs = [0.5, 1]
    ymins = [0, 0]
    ymaxs = [1.0, 1.0]

    camera = None

    has_cow = False
    if pv.vtk_version_info >= (9, 0, 20210718):
        cam_orient_manipulator = vtkCameraOrientationWidget()
        has_cow = True

    curvature_types = ['Gauss_Curvature', 'Mean_Curvature']
    for idx, curvature_name in enumerate(curvature_types):
        plotter = pv.Plotter()
        curvature_title = curvature_name.replace('_', '\n')

        source.GetPointData().SetActiveScalars(curvature_name)
        scalar_range = source.GetPointData().GetScalars(curvature_name).GetRange()

        bands = get_bands(scalar_range, 10)
        freq = get_frequencies(bands, source)
        bands, freq = adjust_ranges(bands, freq)
        print(curvature_name)
        print_bands_frequencies(bands, freq)

        mapper = vtkPolyDataMapper()
        mapper.SetInputData(source)
        mapper.SetScalarModeToUsePointFieldData()
        mapper.SelectColorArray(curvature_name)
        mapper.SetScalarRange(scalar_range)
        mapper.SetLookupTable(lut)

        actor = pv.Actor(mapper=mapper)

        # Create a scalar bar
        scalar_bar = vtkScalarBarActor()
        scalar_bar.SetLookupTable(mapper.GetLookupTable())
        scalar_bar.SetTitle(curvature_title)
        scalar_bar.UnconstrainedFontSizeOn()
        scalar_bar.SetNumberOfLabels(min(5, len(freq)))
        scalar_bar.SetMaximumWidthInPixels(window_width // 8)
        scalar_bar.SetMaximumHeightInPixels(window_height // 3)
        scalar_bar.SetBarRatio(scalar_bar.GetBarRatio() * 0.5)
        scalar_bar.SetPosition(0.85, 0.1)

        text_mapper = vtkTextMapper()
        text_mapper.SetInput(curvature_title)
        text_mapper.SetTextProperty(text_property)

        text_actor = vtkActor2D()
        text_actor.SetMapper(text_mapper)
        text_actor.SetPosition(250, 16)

        renderer = plotter.renderers[0]
        renderer.SetBackground(colors.GetColor3d('ParaViewBkg'))

        renderer.add_actor(actor)
        renderer.add_actor(text_actor)
        renderer.add_actor(scalar_bar)

        ren_win.AddRenderer(renderer)

        if idx == 0:
            if has_cow:
                cam_orient_manipulator.SetParentRenderer(renderer)
            camera = renderer.camera
            camera.elevation = 60
        else:
            renderer.camera = camera
        renderer.SetViewport(xmins[idx], ymins[idx], xmaxs[idx], ymaxs[idx])
        renderer.reset_camera()

    if has_cow:
        # Enable the widget.
        cam_orient_manipulator.On()

    ren_win.Render()
    ren_win.SetWindowName('CurvaturesAdjustEdges')
    iren.Start()


def adjust_edge_curvatures(source, curvature_name, epsilon=1.0e-08):
    """
    This function adjusts curvatures along the edges of the surface by replacing
     the value with the average value of the curvatures of points in the neighborhood.

    Remember to update the vtkCurvatures object before calling this.

    :param source: A vtkPolyData object corresponding to the vtkCurvatures object.
    :param curvature_name: The name of the curvature, 'Gauss_Curvature' or 'Mean_Curvature'.
    :param epsilon: Absolute curvature values less than this will be set to zero.
    :return:
    """

    source = pv.wrap(source)

    def compute_distance(pt_id_a, pt_id_b):
        """
        Compute the distance between two points given their ids.

        :param pt_id_a:
        :param pt_id_b:
        :return:
        """
        pt_a = np.array(source.GetPoint(pt_id_a))
        pt_b = np.array(source.GetPoint(pt_id_b))
        return np.linalg.norm(pt_a - pt_b)

    # Get the active scalars
    source.GetPointData().SetActiveScalars(curvature_name)
    np_source = dsa.WrapDataObject(source)
    curvatures = np_source.PointData[curvature_name]

    #  Get the boundary point IDs.
    array_name = 'ids'
    id_filter = vtkIdFilter()
    id_filter.SetInputData(source)
    id_filter.SetPointIds(True)
    id_filter.SetCellIds(False)
    id_filter.SetPointIdsArrayName(array_name)
    id_filter.SetCellIdsArrayName(array_name)
    id_filter.Update()

    edges = vtkFeatureEdges()
    edges.SetInputConnection(id_filter.GetOutputPort())
    edges.BoundaryEdgesOn()
    edges.ManifoldEdgesOff()
    edges.NonManifoldEdgesOff()
    edges.FeatureEdgesOff()
    edges.Update()

    edge_array = edges.GetOutput().GetPointData().GetArray(array_name)
    boundary_ids = []
    for i in range(edges.GetOutput().GetNumberOfPoints()):
        boundary_ids.append(edge_array.GetValue(i))
    # Remove duplicate Ids.
    p_ids_set = set(boundary_ids)

    # Iterate over the edge points and compute the curvature as the weighted
    # average of the neighbours.
    count_invalid = 0
    for p_id in boundary_ids:
        p_ids_neighbors = set(source.point_neighbors(p_id))
        # Keep only interior points.
        p_ids_neighbors -= p_ids_set
        # Compute distances and extract curvature values.
        curvs = [curvatures[p_id_n] for p_id_n in p_ids_neighbors]
        dists = [compute_distance(p_id_n, p_id) for p_id_n in p_ids_neighbors]
        curvs = np.array(curvs)
        dists = np.array(dists)
        curvs = curvs[dists > 0]
        dists = dists[dists > 0]
        if len(curvs) > 0:
            weights = 1 / np.array(dists)
            weights /= weights.sum()
            new_curv = np.dot(curvs, weights)
        else:
            # Corner case.
            count_invalid += 1
            # Assuming the curvature of the point is planar.
            new_curv = 0.0
        # Set the new curvature value.
        curvatures[p_id] = new_curv

    #  Set small values to zero.
    if epsilon != 0.0:
        curvatures = np.where(abs(curvatures) < epsilon, 0, curvatures)
        # Curvatures is now an ndarray
        curv = numpy_support.numpy_to_vtk(
            num_array=curvatures.ravel(), deep=True, array_type=VTK_DOUBLE
        )
        curv.SetName(curvature_name)
        source.GetPointData().RemoveArray(curvature_name)
        source.GetPointData().AddArray(curv)
        source.GetPointData().SetActiveScalars(curvature_name)


def get_source(source):
    surface = source.lower()
    available_surfaces = [
        'randomhills',
    ]
    if surface not in available_surfaces:
        return None
    if surface == 'randomhills':
        source = pv.ParametricRandomHills(
            random_seed=1, number_of_hills=30, u_res=51, v_res=51, texture_coordinates=True
        )
        return source.translate((0.0, 5.0, 15.0)).rotate_x(-90.0)
    return None


def get_frequencies(bands, src):
    """
    Count the number of scalars in each band.
    The scalars used are the active scalars in the polydata.

    :param: bands - The bands.
    :param: src - The vtkPolyData source.
    :return: The frequencies of the scalars in each band.
    """
    freq = dict()
    for i in range(len(bands)):
        freq[i] = 0
    tuples = src.GetPointData().GetScalars().GetNumberOfTuples()
    for i in range(tuples):
        x = src.GetPointData().GetScalars().GetTuple1(i)
        for j in range(len(bands)):
            if x <= bands[j][2]:
                freq[j] += 1
                break
    return freq


def adjust_ranges(bands, freq):
    """
    The bands and frequencies are adjusted so that the first and last
     frequencies in the range are non-zero.
    :param bands: The bands dictionary.
    :param freq: The frequency dictionary.
    :return: Adjusted bands and frequencies.
    """
    # Get the indices of the first and last non-zero elements.
    first = 0
    for k, v in freq.items():
        if v != 0:
            first = k
            break
    rev_keys = list(freq.keys())[::-1]
    last = rev_keys[0]
    for idx in list(freq.keys())[::-1]:
        if freq[idx] != 0:
            last = idx
            break
    # Now adjust the ranges.
    min_key = min(freq.keys())
    max_key = max(freq.keys())
    for idx in range(min_key, first):
        freq.pop(idx)
        bands.pop(idx)
    for idx in range(last + 1, max_key + 1):
        freq.popitem()
        bands.popitem()
    old_keys = freq.keys()
    adj_freq = dict()
    adj_bands = dict()

    for idx, k in enumerate(old_keys):
        adj_freq[idx] = freq[k]
        adj_bands[idx] = bands[k]

    return adj_bands, adj_freq


def get_bands(d_r, number_of_bands, precision=2, nearest_integer=False):
    """
    Divide a range into bands
    :param: d_r - [min, max] the range that is to be covered by the bands.
    :param: number_of_bands - The number of bands, a positive integer.
    :param: precision - The decimal precision of the bounds.
    :param: nearest_integer - If True then [floor(min), ceil(max)] is used.
    :return: A dictionary consisting of the band number and [min, midpoint, max] for each band.
    """
    prec = abs(precision)
    if prec > 14:
        prec = 14

    bands = dict()
    if (d_r[1] < d_r[0]) or (number_of_bands <= 0):
        return bands
    x = list(d_r)
    if nearest_integer:
        x[0] = math.floor(x[0])
        x[1] = math.ceil(x[1])
    dx = (x[1] - x[0]) / float(number_of_bands)
    b = [x[0], x[0] + dx / 2.0, x[0] + dx]
    i = 0
    while i < number_of_bands:
        b = list(map(lambda ele_b: round(ele_b, prec), b))
        if i == 0:
            b[0] = x[0]
        bands[i] = b
        b = [b[0] + dx, b[1] + dx, b[2] + dx]
        i += 1
    return bands


def print_bands_frequencies(bands, freq, precision=2):
    prec = abs(precision)
    if prec > 14:
        prec = 14

    if len(bands) != len(freq):
        print('Bands and Frequencies must be the same size.')
        return
    s = 'Bands & Frequencies:\n'
    total = 0
    width = prec + 6
    for k, v in bands.items():
        total += freq[k]
        for j, q in enumerate(v):
            if j == 0:
                s += f'{k:4d} ['
            if j == len(v) - 1:
                s += f'{q:{width}.{prec}f}]: {freq[k]:8d}\n'
            else:
                s += f'{q:{width}.{prec}f}, '
    width = 3 * width + 13
    s += f'{"Total":{width}s}{total:8d}\n'
    print(s)


if __name__ == '__main__':
    import sys

    main(sys.argv)
