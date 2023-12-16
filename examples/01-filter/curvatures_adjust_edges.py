"""
.. _curvatures_adjust_edges:

curvatures Adjust Edges
~~~~~~~~~~~~~~~~~~~~~~~
"""
import math

import pyvista as pv


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


source = (
    pv.ParametricRandomHills(
        random_seed=1, number_of_hills=30, u_res=51, v_res=51, texture_coordinates=True
    )
    .translate((0.0, 5.0, 15.0))
    .rotate_x(-90.0)
)

source['Gauss_Curvature'] = source.curvature("gaussian", adjust_edges=True)
source['Mean_Curvature'] = source.curvature("mean", adjust_edges=True)

# Let's visualise what we have done.

window_width = 1024
window_height = 512

plotter = pv.Plotter(shape=(1, 2), window_size=(window_width, window_height))

# Create a common text property.
text_property = pv.TextProperty()
text_property.font_size = 24
text_property.justification_horizontal = "center"
text_property.color = "white"

lut = pv.LookupTable('coolwarm', n_values=256)

# Define viewport ranges
xmins = [0, 0.5]
xmaxs = [0.5, 1]
ymins = [0, 0]
ymaxs = [1.0, 1.0]

curvature_name = 'Gauss_Curvature'
plotter.subplot(0, 0)
curvature_title = curvature_name.replace('_', '\n')

source.GetPointData().SetActiveScalars(curvature_name)
scalar_range = source.GetPointData().GetScalars(curvature_name).GetRange()

bands = get_bands(scalar_range, 10)
freq = get_frequencies(bands, source)
bands, freq = adjust_ranges(bands, freq)
print(curvature_name)
print_bands_frequencies(bands, freq)

mapper = pv.DataSetMapper()
mapper.SetInputData(source)
mapper.SetScalarModeToUsePointFieldData()
mapper.SelectColorArray(curvature_name)
mapper.SetScalarRange(scalar_range)
mapper.SetLookupTable(lut)

actor = pv.Actor(mapper=mapper)

text_actor = pv.Text(text=curvature_title)
text_actor.prop = text_property
text_actor.position = (250, 16)

plotter.add_actor(actor)
plotter.set_background([82, 87, 110])
plotter.add_actor(text_actor)
plotter.add_scalar_bar(
    title=curvature_title,
    unconstrained_font_size=True,
    mapper=mapper,
    n_labels=min(5, len(freq)),
    position_x=0.85,
    position_y=0.1,
    vertical=True,
    color='white',
)
renderer = plotter.renderers[0]

camera = renderer.camera
camera.elevation = 60
renderer.SetViewport(xmins[0], ymins[0], xmaxs[0], ymaxs[0])
renderer.reset_camera()

curvature_name = 'Mean_Curvature'
plotter.subplot(0, 1)
curvature_title = curvature_name.replace('_', '\n')

source.GetPointData().SetActiveScalars(curvature_name)
scalar_range = source.GetPointData().GetScalars(curvature_name).GetRange()

bands = get_bands(scalar_range, 10)
freq = get_frequencies(bands, source)
bands, freq = adjust_ranges(bands, freq)
print(curvature_name)
print_bands_frequencies(bands, freq)

mapper = pv.DataSetMapper()
mapper.SetInputData(source)
mapper.SetScalarModeToUsePointFieldData()
mapper.SelectColorArray(curvature_name)
mapper.SetScalarRange(scalar_range)
mapper.SetLookupTable(lut)

actor = pv.Actor(mapper=mapper)

text_actor = pv.Text(text=curvature_title)
text_actor.prop = text_property
text_actor.position = (250, 16)

plotter.add_actor(actor)
plotter.set_background([82, 87, 110])
plotter.add_actor(text_actor)
plotter.add_scalar_bar(
    title=curvature_title,
    unconstrained_font_size=True,
    mapper=mapper,
    n_labels=min(5, len(freq)),
    position_x=0.85,
    position_y=0.1,
    vertical=True,
    color='white',
)
renderer = plotter.renderers[1]


renderer.camera = camera
renderer.SetViewport(xmins[1], ymins[1], xmaxs[1], ymaxs[1])
renderer.reset_camera()

plotter.add_camera_orientation_widget()
plotter.show()
