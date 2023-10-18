#!/usr/bin/env python

import math

import numpy as np
from vtk.util import numpy_support
from vtkmodules.numpy_interface import dataset_adapter as dsa
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkCommonComputationalGeometry import (
    vtkParametricBour,
    vtkParametricEnneper,
    vtkParametricMobius,
    vtkParametricTorus,
)
from vtkmodules.vtkCommonCore import (
    VTK_DOUBLE,
    vtkDoubleArray,
    vtkFloatArray,
    vtkLookupTable,
)
from vtkmodules.vtkCommonTransforms import vtkTransform
from vtkmodules.vtkFiltersCore import (
    vtkDelaunay2D,
    vtkFeatureEdges,
    vtkIdFilter,
    vtkPolyDataNormals,
    vtkPolyDataTangents,
    vtkTriangleFilter,
)
from vtkmodules.vtkFiltersGeneral import vtkCurvatures, vtkTransformPolyDataFilter
from vtkmodules.vtkFiltersModeling import vtkLinearSubdivisionFilter
from vtkmodules.vtkFiltersSources import (
    vtkCubeSource,
    vtkParametricFunctionSource,
    vtkTexturedSphereSource,
)

# noinspection PyUnresolvedReferences
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera
from vtkmodules.vtkInteractionWidgets import vtkCameraOrientationWidget
from vtkmodules.vtkRenderingAnnotation import vtkScalarBarActor
from vtkmodules.vtkRenderingCore import (
    vtkActor2D,
    vtkColorTransferFunction,
    vtkPolyDataMapper,
    vtkRenderWindow,
    vtkRenderWindowInteractor,
    vtkTextMapper,
)

import pyvista as pv


def main(argv):
    # desired_surface = 'Bour'
    # desired_surface = 'Cube'
    # desired_surface = 'Hills'
    # desired_surface = 'Enneper'
    # desired_surface = 'Mobius'
    desired_surface = 'RandomHills'
    # desired_surface = 'Sphere'
    # desired_surface = 'Torus'
    source = get_source(desired_surface)
    if not source:
        print('The surface is not available.')
        return

    gc = vtkCurvatures()
    gc.SetInputData(source)
    gc.SetCurvatureTypeToGaussian()
    gc.Update()
    if desired_surface in ['Bour', 'Enneper', 'Hills', 'RandomHills', 'Torus']:
        adjust_edge_curvatures(gc.GetOutput(), 'Gauss_Curvature')
    if desired_surface == 'Bour':
        # Gaussian curvature is -1/(r(r+1)^4))
        constrain_curvatures(gc.GetOutput(), 'Gauss_Curvature', -0.0625, -0.0625)
    if desired_surface == 'Enneper':
        # Gaussian curvature is -4/(1 + r^2)^4
        constrain_curvatures(gc.GetOutput(), 'Gauss_Curvature', -0.25, -0.25)
    if desired_surface == 'Cube':
        constrain_curvatures(gc.GetOutput(), 'Gauss_Curvature', 0.0, 0.0)
    if desired_surface == 'Mobius':
        constrain_curvatures(gc.GetOutput(), 'Gauss_Curvature', 0.0, 0.0)
    if desired_surface == 'Sphere':
        # Gaussian curvature is 1/r^2
        constrain_curvatures(gc.GetOutput(), 'Gauss_Curvature', 4.0, 4.0)
    source.GetPointData().AddArray(
        gc.GetOutput().GetPointData().GetAbstractArray('Gauss_Curvature')
    )

    mc = vtkCurvatures()
    mc.SetInputData(source)
    mc.SetCurvatureTypeToMean()
    mc.Update()
    if desired_surface in ['Bour', 'Enneper', 'Hills', 'RandomHills', 'Torus']:
        adjust_edge_curvatures(mc.GetOutput(), 'Mean_Curvature')
    if desired_surface == 'Bour':
        # Mean curvature is 0
        constrain_curvatures(mc.GetOutput(), 'Mean_Curvature', 0.0, 0.0)
    if desired_surface == 'Enneper':
        # Mean curvature is 0
        constrain_curvatures(mc.GetOutput(), 'Mean_Curvature', 0.0, 0.0)
    if desired_surface == 'Sphere':
        # Mean curvature is 1/r
        constrain_curvatures(mc.GetOutput(), 'Mean_Curvature', 2.0, 2.0)
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

    lut = get_diverging_lut()
    # lut = get_diverging_lut1()

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

        actor = pv.Actor(mapper = mapper)

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


def constrain_curvatures(source, curvature_name, lower_bound=0.0, upper_bound=0.0):
    """
    This function constrains curvatures to the range [lower_bound ... upper_bound].

    Remember to update the vtkCurvatures object before calling this.

    :param source: A vtkPolyData object corresponding to the vtkCurvatures object.
    :param curvature_name: The name of the curvature, 'Gauss_Curvature' or 'Mean_Curvature'.
    :param lower_bound: The lower bound.
    :param upper_bound: The upper bound.
    :return:
    """

    bounds = list()
    if lower_bound < upper_bound:
        bounds.append(lower_bound)
        bounds.append(upper_bound)
    else:
        bounds.append(upper_bound)
        bounds.append(lower_bound)

    # Get the active scalars
    source.GetPointData().SetActiveScalars(curvature_name)
    np_source = dsa.WrapDataObject(source)
    curvatures = np_source.PointData[curvature_name]

    # Set upper and lower bounds.
    curvatures = np.where(curvatures < bounds[0], bounds[0], curvatures)
    curvatures = np.where(curvatures > bounds[1], bounds[1], curvatures)
    # Curvatures is now an ndarray
    curv = numpy_support.numpy_to_vtk(
        num_array=curvatures.ravel(), deep=True, array_type=VTK_DOUBLE
    )
    curv.SetName(curvature_name)
    source.GetPointData().RemoveArray(curvature_name)
    source.GetPointData().AddArray(curv)
    source.GetPointData().SetActiveScalars(curvature_name)


def get_diverging_lut():
    """
    See: [Diverging Color Maps for Scientific Visualization](https://www.kennethmoreland.com/color-maps/)
                       start point         midPoint            end point
     cool to warm:     0.230, 0.299, 0.754 0.865, 0.865, 0.865 0.706, 0.016, 0.150
     purple to orange: 0.436, 0.308, 0.631 0.865, 0.865, 0.865 0.759, 0.334, 0.046
     green to purple:  0.085, 0.532, 0.201 0.865, 0.865, 0.865 0.436, 0.308, 0.631
     blue to brown:    0.217, 0.525, 0.910 0.865, 0.865, 0.865 0.677, 0.492, 0.093
     green to red:     0.085, 0.532, 0.201 0.865, 0.865, 0.865 0.758, 0.214, 0.233

    :return:
    """
    ctf = vtkColorTransferFunction()
    ctf.SetColorSpaceToDiverging()
    # Cool to warm.
    ctf.AddRGBPoint(0.0, 0.230, 0.299, 0.754)
    ctf.AddRGBPoint(0.5, 0.865, 0.865, 0.865)
    ctf.AddRGBPoint(1.0, 0.706, 0.016, 0.150)

    table_size = 256
    lut = vtkLookupTable()
    lut.SetNumberOfTableValues(table_size)
    lut.Build()

    for i in range(0, table_size):
        rgba = list(ctf.GetColor(float(i) / table_size))
        rgba.append(1)
        lut.SetTableValue(i, rgba)

    return lut


def get_diverging_lut1():
    colors = vtkNamedColors()
    # Colour transfer function.
    ctf = vtkColorTransferFunction()
    ctf.SetColorSpaceToDiverging()
    p1 = [0.0] + list(colors.GetColor3d('MidnightBlue'))
    p2 = [0.5] + list(colors.GetColor3d('Gainsboro'))
    p3 = [1.0] + list(colors.GetColor3d('DarkOrange'))
    ctf.AddRGBPoint(*p1)
    ctf.AddRGBPoint(*p2)
    ctf.AddRGBPoint(*p3)

    table_size = 256
    lut = vtkLookupTable()
    lut.SetNumberOfTableValues(table_size)
    lut.Build()

    for i in range(0, table_size):
        rgba = list(ctf.GetColor(float(i) / table_size))
        rgba.append(1)
        lut.SetTableValue(i, rgba)

    return lut


def get_bour():
    u_resolution = 51
    v_resolution = 51
    surface = vtkParametricBour()

    source = vtkParametricFunctionSource()
    source.SetUResolution(u_resolution)
    source.SetVResolution(v_resolution)
    source.GenerateTextureCoordinatesOn()
    source.SetParametricFunction(surface)
    source.Update()

    # Build the tangents.
    tangents = vtkPolyDataTangents()
    tangents.SetInputConnection(source.GetOutputPort())
    tangents.Update()

    transform = vtkTransform()
    transform.RotateX(0.0)
    transform_filter = vtkTransformPolyDataFilter()
    transform_filter.SetInputConnection(tangents.GetOutputPort())
    transform_filter.SetTransform(transform)
    transform_filter.Update()

    return transform_filter.GetOutput()


def get_cube():
    surface = vtkCubeSource()

    # Triangulate.
    triangulation = vtkTriangleFilter()
    triangulation.SetInputConnection(surface.GetOutputPort())
    # Subdivide the triangles
    subdivide = vtkLinearSubdivisionFilter()
    subdivide.SetInputConnection(triangulation.GetOutputPort())
    subdivide.SetNumberOfSubdivisions(3)
    # Now the tangents.
    tangents = vtkPolyDataTangents()
    tangents.SetInputConnection(subdivide.GetOutputPort())
    tangents.Update()
    return tangents.GetOutput()


def get_hills():
    # Create four hills on a plane.
    # This will have regions of negative, zero and positive Gsaussian curvatures.

    x_res = 50
    y_res = 50
    x_min = -5.0
    x_max = 5.0
    dx = (x_max - x_min) / (x_res - 1)
    y_min = -5.0
    y_max = 5.0
    dy = (y_max - y_min) / (x_res - 1)

    # Make a grid.
    points = pv.Points()
    for i in range(0, x_res):
        x = x_min + i * dx
        for j in range(0, y_res):
            y = y_min + j * dy
            points.InsertNextPoint(x, y, 0)

    # Add the grid points to a polydata object.
    plane = pv.PolyData()
    plane.SetPoints(points)

    # Triangulate the grid.
    delaunay = vtkDelaunay2D()
    delaunay.SetInputData(plane)
    delaunay.Update()

    polydata = delaunay.GetOutput()

    elevation = vtkDoubleArray()
    elevation.SetNumberOfTuples(points.GetNumberOfPoints())

    #  We define the parameters for the hills here.
    # [[0: x0, 1: y0, 2: x variance, 3: y variance, 4: amplitude]...]
    hd = [
        [-2.5, -2.5, 2.5, 6.5, 3.5],
        [2.5, 2.5, 2.5, 2.5, 2],
        [5.0, -2.5, 1.5, 1.5, 2.5],
        [-5.0, 5, 2.5, 3.0, 3],
    ]
    xx = [0.0] * 2
    for i in range(0, points.GetNumberOfPoints()):
        x = list(polydata.GetPoint(i))
        for j in range(0, len(hd)):
            xx[0] = (x[0] - hd[j][0] / hd[j][2]) ** 2.0
            xx[1] = (x[1] - hd[j][1] / hd[j][3]) ** 2.0
            x[2] += hd[j][4] * math.exp(-(xx[0] + xx[1]) / 2.0)
            polydata.GetPoints().SetPoint(i, x)
            elevation.SetValue(i, x[2])

    textures = vtkFloatArray()
    textures.SetNumberOfComponents(2)
    textures.SetNumberOfTuples(2 * polydata.GetNumberOfPoints())
    textures.SetName("Textures")

    for i in range(0, x_res):
        tc = [i / (x_res - 1.0), 0.0]
        for j in range(0, y_res):
            # tc[1] = 1.0 - j / (y_res - 1.0)
            tc[1] = j / (y_res - 1.0)
            textures.SetTuple(i * y_res + j, tc)

    polydata.GetPointData().SetScalars(elevation)
    polydata.GetPointData().GetScalars().SetName("Elevation")
    polydata.GetPointData().SetTCoords(textures)

    normals = vtkPolyDataNormals()
    normals.SetInputData(polydata)
    normals.SetInputData(polydata)
    normals.SetFeatureAngle(30)
    normals.SplittingOff()

    tr1 = vtkTransform()
    tr1.RotateX(-90)

    tf1 = vtkTransformPolyDataFilter()
    tf1.SetInputConnection(normals.GetOutputPort())
    tf1.SetTransform(tr1)
    tf1.Update()

    return tf1.GetOutput()


def get_enneper():
    u_resolution = 51
    v_resolution = 51
    surface = vtkParametricEnneper()

    source = vtkParametricFunctionSource()
    source.SetUResolution(u_resolution)
    source.SetVResolution(v_resolution)
    source.GenerateTextureCoordinatesOn()
    source.SetParametricFunction(surface)
    source.Update()

    # Build the tangents.
    tangents = vtkPolyDataTangents()
    tangents.SetInputConnection(source.GetOutputPort())
    tangents.Update()

    transform = vtkTransform()
    transform.RotateX(0.0)
    transform_filter = vtkTransformPolyDataFilter()
    transform_filter.SetInputConnection(tangents.GetOutputPort())
    transform_filter.SetTransform(transform)
    transform_filter.Update()

    return transform_filter.GetOutput()


def get_mobius():
    u_resolution = 51
    v_resolution = 51
    surface = vtkParametricMobius()
    surface.SetMinimumV(-0.25)
    surface.SetMaximumV(0.25)

    source = vtkParametricFunctionSource()
    source.SetUResolution(u_resolution)
    source.SetVResolution(v_resolution)
    source.GenerateTextureCoordinatesOn()
    source.SetParametricFunction(surface)
    source.Update()

    # Build the tangents.
    tangents = vtkPolyDataTangents()
    tangents.SetInputConnection(source.GetOutputPort())
    tangents.Update()

    transform = vtkTransform()
    transform.RotateX(-90.0)
    transform_filter = vtkTransformPolyDataFilter()
    transform_filter.SetInputConnection(tangents.GetOutputPort())
    transform_filter.SetTransform(transform)
    transform_filter.Update()

    return transform_filter.GetOutput()


def get_sphere():
    theta_resolution = 32
    phi_resolution = 32
    surface = vtkTexturedSphereSource()
    surface.SetThetaResolution(theta_resolution)
    surface.SetPhiResolution(phi_resolution)

    # Now the tangents.
    tangents = vtkPolyDataTangents()
    tangents.SetInputConnection(surface.GetOutputPort())
    tangents.Update()

    return tangents.GetOutput()


def get_torus():
    u_resolution = 51
    v_resolution = 51
    surface = vtkParametricTorus()

    source = vtkParametricFunctionSource()
    source.SetUResolution(u_resolution)
    source.SetVResolution(v_resolution)
    source.GenerateTextureCoordinatesOn()
    source.SetParametricFunction(surface)
    source.Update()

    # Build the tangents.
    tangents = vtkPolyDataTangents()
    tangents.SetInputConnection(source.GetOutputPort())
    tangents.Update()

    transform = vtkTransform()
    transform.RotateX(-90.0)
    transform_filter = vtkTransformPolyDataFilter()
    transform_filter.SetInputConnection(tangents.GetOutputPort())
    transform_filter.SetTransform(transform)
    transform_filter.Update()

    return transform_filter.GetOutput()


def get_source(source):
    surface = source.lower()
    available_surfaces = [
        'bour',
        'cube',
        'enneper',
        'hills',
        'mobius',
        'randomhills',
        'sphere',
        'torus',
    ]
    if surface not in available_surfaces:
        return None
    elif surface == 'bour':
        return get_bour()
    elif surface == 'cube':
        return get_cube()
    elif surface == 'enneper':
        return get_enneper()
    elif surface == 'hills':
        return get_hills()
    elif surface == 'mobius':
        return get_mobius()
    elif surface == 'randomhills':
        source = pv.ParametricRandomHills(
            random_seed=1, number_of_hills=30, u_res=51, v_res=51, texture_coordinates=True
        )
        return source.translate((0.0, 5.0, 15.0)).rotate_x(-90.0)
    elif surface == 'sphere':
        return get_sphere()
    elif surface == 'torus':
        return get_torus()
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
