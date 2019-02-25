---
title: 'vtki: A Python interface for the Visualization Toolkit'
tags:
  - Python
  - visualization
  - 3D
authors:
  - name: C. Bane Sullivan
    orcid: 0000-0001-8628-4566
    affiliation: 1
  - name: Alex Kaszynski
    orcid: 0000-0001-6293-5352
    affiliation: 2
affiliations:
 - name: Department of Geophysics, Colorado School of Mines, Golden, CO, USA
   index: 1
 - name: TODO
   index: 2

date: 10 February 2019
bibliography: paper.bib
---

# Summary

The Visualization Toolkit (VTK) is an excellent visualization software library,
moreover, with Python bindings, it combines the speed of C++ with the rapid
prototyping of Python [@vtkbook].
However, VTK code programmed in Python generally looks the same as its C++
counterpart. This Python package seeks to simplify common mesh creation and
plotting routines without compromising on the speed of the C++ VTK backend.
At its core, `vtki` is a pure Python helper module for VTK that interfaces with
back to VTK data objects through NumPy [@numpy] and direct array access.
This package expands upon VTK's data objects by creating classes
that extend their VTK counterpart.
VTK data objects passed to `vtki` have an added layer of functionality on top of
that object providing a wrapping layer that creates an accessible and intuitive
interface back to the VTK library to foster rapid prototyping and analysis of
VTK datasets.

## Data Types

Datasets are any spatially referenced information and usually consist of
geometrical representations of a surface or volume in 3D space.
In VTK, the abstract class `vtk.vtkDataSet` represents a set of common
functionality for spatially referenced datasets [@vtkbook].
In `vtki`, the common functionality shared across spatially referenced datasets
is shared in the `vtki.Common` class which holds methods and attributes for
quickly accessing scalar arrays associated with the dataset or easily inspecting
attributes of the dataset such as all the scalar names or number of points
present.

In VTK, datasets consist of geometry, topology, and attributes to which `vtki`
provides direct access through NumPy arrays [@vtkbook]:

- The geometry of the dataset is the collection of points and cells in 2D or 3D
space.
- Topology defines the structure of the dataset, or how the points are connected
to each other to form a cells constructing a surface or volume.
- Attributes are any data values that are associated with either the points or
cells of the dataset.

All of the following data types are subclasses of their corresponding VTK class
and share a set of common functionality which `vtki` implements into the base
class  `vtki.Common`.

| VTK Class                  | `vtki` Implementation   |
|----------------------------|-------------------------|
| `vtk.vtkDataSet`           | `vtki.Common`           |
| `vtk.vtkPolyData`          | `vtki.PolyData`         |
| `vtk.vtkUnstructuredGrid`  | `vtki.UnstructuredGrid` |
| `vtk.vtkStructuredGrid`    | `vtki.StructuredGrid`   |
| `vtk.vtkRectilinearGrid`   | `vtki.RectilinearGrid`  |
| `vtk.vtkImageData`         | `vtki.UniformGrid`      |
| `vtk.vtkMultiBlockDataSet` | `vtki.MultiBlock`       |


Creation of VTK data objects over the `vtki` interface can be completed in a few
lines of code. Loading files supported by the VTK library is:

```python
import vtki
import numpy as np

filename = 'path/to/vtk/supported/file.ext'
mesh = vtki.read(filename)
```

Creating mesh objects in VTK is also simplified by `vtki` by providing intuitive
initialization functions and attributes on the `vtki` classes that callback to
the original VTK data object.

Create polygonal data from scattered points:

```python
points = np.random.rand(100, 3)
poly = vtki.PolyData(points)
# Add a data on the nodes of the mesh
poly.point_arrays['foo'] = np.random.rand(poly.n_points)
```

Create a structured grid:

```python
# Make data
x = np.arange(-10, 10, 0.25)
y = np.arange(-10, 10, 0.25)
x, y = np.meshgrid(x, y)
r = np.sqrt(x**2 + y**2)
z = np.sin(r)
# Create and plot structured grid
sgrid = vtki.StructuredGrid(x, y, z)
# Add a data on the cells of the mesh
values = np.linspace(0, 10, sgrid.n_cells)
sgrid.cell_arrays['values'] = values.ravel()
```

Create uniform grid:

```python
# Create the spatial reference
ugrid = vtki.UniformGrid()
# Set the size of the grid
ugrid.dimensions = (20, 5, 10)
# Edit the spatial reference
ugrid.origin = (100, 33, 55.6) # The bottom left corner of the data set
ugrid.spacing = (1, 5, 2) # These are the cell sizes along each axis
# Add a data on the cells of the mesh
values = np.linspace(0, 10, ugrid.n_cells).reshape(np.array(ugrid.dimensions)-1)
ugrid.cell_arrays['values'] = values.flatten(order='F')
```


## Simplified Plotting Routines

Plotting VTK datasets using only the VTK Python package is often an ambitious
programming endeavor. For example, to read a VTK supported file and plot it, a
user would have to write the following code using VTK alone
(adapted from [this creative commons VTK example](https://vtk.org/Wiki/VTK/Examples/Python/STLReader)):

```python
import vtk

# create reader
reader = vtk.vtkSTLReader()
reader.SetFileName("myfile.stl")

mapper = vtk.vtkPolyDataMapper()
if vtk.VTK_MAJOR_VERSION <= 5:
    mapper.SetInput(reader.GetOutput())
else:
    mapper.SetInputConnection(reader.GetOutputPort())

# create actor
actor = vtk.vtkActor()
actor.SetMapper(mapper)

# Create a rendering window and renderer
ren = vtk.vtkRenderer()
renWin = vtk.vtkRenderWindow()
renWin.AddRenderer(ren)

# Create a renderwindowinteractor
iren = vtk.vtkRenderWindowInteractor()
iren.SetRenderWindow(renWin)

# Assign actor to the renderer
ren.AddActor(actor)

# Enable user interface interactor
iren.Initialize()
renWin.Render()
iren.Start()

# clean up objects
del iren
del renWin
```


`vtki` includes numerous plotting routines that are intended to be intuitive and
highly controllable with `matplotlib` [@matplotlib] similar syntax and keyword
arguments.
These plotting routines are defined to make the rendering process
straightforward and easily implemented by novice VTK users. The above
rendering example is translated to the following code in `vtki`:

```python
mesh = vtki.read('myfile.stl')
mesh.plot()
```

Notably, the `vtki.plot()` convenience method is binded to each `vtki`
data object to make visual inspection of datasets easily performed. Other
plotting routines in `vtki` are available for creating integrated and
easily manipulated scenes via the `vtki.Plotter` and `vtki.BackgroundPlotter`
classes. Creating a rendering scene and altering its properties can be performed
with the following code in `vtki`:

```python
plotter = vtki.Plotter()
plotter.add_mesh(mesh, color='yellow')
plotter.show_grid()
plotter.show()
```


## Accessing Common Analysis Routines

`vtki` wrapped data objects have a suite of common filters ready for immediate
use directly on the objects. These filters are commonly used algorithms in the
VTK library that have been made more accessible by binding a method to control
that algorithm directly onto all `vtki` datasets. These filtering algorithms are
held in the `vtki.DataSetFilters` class which is inherited by the `vtki.Common`
class giving all datasets a shared set of functionality.
Through the use of these binded methods, powerful VTK filtering algorithms can
be leveraged with intuitive control via keyword arguments in Python.
These filters can be used by calling the filtering method directly from the data
object:

```python
# Load a sample UniformGrid
from vtki import examples
dataset = examples.load_uniform()
# Apply a threshold over a data range
result = dataset.threshold([100, 500])
```

Above, an extracted version of the input dataset where the active scalar array
is between 100 and 500 is created in the new `result` object.
Documentation of the available keyword arguments to control the
filtering algorithms are described in the docstrings of each filtering method.

### Filtering Chain

In VTK, filters are often used in a pipeline where each algorithm passes its
output to the next filtering algorithm [@vtkbook].
`vtki` mimics the filtering pipeline through a chain; attaching each filter to
the last filter. In the following example using the sample dataset from above,
several filters are chained together.

1. An empty threshold filter to clean out any `NaN` values.
2. An elevation filter to generate scalar values corresponding to height.
3. A clip filter to cut the dataset in half.
4. Create three slices along each axial plane.

```python
# Apply a filtering chain
result = dataset.threshold().elevation().clip(normal='z').slice_orthogonal()
```

A complete list of common filters can be found in the
[`vtki` documentation](http://docs.vtki.org/en/latest/tools/filters.html#vtki.DataSetFilters)


## Applications

Advanced transformation and rendering of data is easily performed using `vtki`'s
NumPy interface. An example of plotting arrow glyphs from a simple numerical
function is provided below:

```python
import vtki
import numpy as np

sphere = vtki.Sphere(radius=3.14)

# make a swirly pattern
vectors = np.vstack((np.sin(sphere.points[:, 0]),
            np.cos(sphere.points[:, 1]),
            np.cos(sphere.points[:, 2]))).T

# associate and scale the vectors
sphere.vectors = vectors*0.3

# plot the arrows
sphere.arrows.plot(cmap='Blues')
```


`vtki` can also be used to create integrated visualizations of any spatially
referenced data. Using the `vtki`-based
[`omfvtk` Python package](https://github.com/OpenGeoVis/omfvtk),
users can load geospatial data into VTK data structures and create compelling
visualizations of real data in just a few lines of code:

```python
import vtki
import omfvtk

# Set a document friendly plotting theme
vtki.set_plot_theme('document')

# Load example OMF data file into a vtki.MultiBlock object
proj = omfvtk.load_project('test_file.omf')

# Grab a few elements of interest
assay = proj['wolfpass_WP_assay']
topo = proj['Topography']
dacite = proj['Dacite']
vol = proj['Block Model']

# Set the scalar attribute to use when plotting or filtering
assay.set_active_scalar('DENSITY')

# Apply a volumetric threshold on the Block Model dataset
threshed = vol.threshold_percent([0.25, 0.75])

# Create a plotting window
p = vtki.Plotter()
# Add the bounds axis with grid lines
p.show_grid()

# Add datasets
p.add_mesh(topo, texture=True, opacity=0.90, name='topo')
p.add_mesh(dacite, color='orange', opacity=0.6, name='dacite')
p.add_mesh(threshed, name='vol')

# Add the assay logs: use a tube filter that varius the radius by an attribute
# this will vary the radius by 'CU_pct' and color by the active scalar array
tubes = assay.tube(scalars='CU_pct', radius=3)
p.add_mesh(tubes, scalars='DENSITY', name='assay', cmap='viridis')

p.show(auto_close=False)
# Save a screenshot!
# p.screenshot('wolfpass.png')
p.close()

```

## References
