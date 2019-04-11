---
title: 'vtki: A Streamlined Python Interface for the Visualization Toolkit'
tags:
  - Python
  - visualization
  - 3D
authors:
  - name: C. Bane Sullivan
    orcid: 0000-0001-8628-4566
    affiliation: 1
  - name: Alexander A. Kaszynski
    orcid: 0000-0002-8232-7212
    affiliation: 2
affiliations:
 - name: Department of Geophysics, Colorado School of Mines, Golden, CO, USA
   index: 1
 - name: Universal Technology Corporation, Dayton, OH, USA
   index: 2

date: 10 February 2019
bibliography: paper.bib
---

# Summary

The Visualization Toolkit (VTK) is an excellent visualization software
library, and with Python bindings it combines the speed of C++ with
the rapid prototyping of Python [@vtkbook].  Despite this, VTK code
programmed in Python using the base package provided by Kitware is
unnecessarily complex as the Python package merely wraps existing C++
calls. This Python package seeks to simplify common mesh creation and
plotting routines without compromising on the speed of the C++ VTK
backend, enabling researchers to rapidly explore large datasets,
communicate their spatial findings, and facilitate reproducibility.
At its core, `vtki` is a pure Python helper module for VTK
that interfaces back to VTK data objects through NumPy [@numpy]
and direct array access.  This package expands upon VTK's data objects
by creating classes that extend their VTK counterpart. VTK data
objects passed to `vtki` have an added layer of functionality on top
of that object providing an accessible and intuitive interface back to
the VTK library to foster rapid prototyping, analysis, and visual
integration of spatially referenced datasets.
Figure 1 demonstrates an integrated scene of geospatial data generated
by `vtki`; to learn more about this dataset, please visit
[this website](http://forge.pvgeo.org).


![A visually integrated scene of geospatial data (FORGE Geothermal Site)](./images/forge-iso.png)
**Figure 1:** A visually integrated scene of geospatial data
(FORGE Geothermal Site). This rendering includes a land surface digital
elevation map with overlain satellite imagery and geologic map, a subsurface
temperature model, scattered points of the sampled temperature values,
geophysical well logging data, GIS site boundary, and interpreted faulting
surfaces.




## Simplified Plotting Routines

Plotting VTK datasets using only the VTK Python package or a similar
visualization library is often an ambitious programming endeavor.
Reading a VTK supported file and plotting it requires a user to write a
complicated sequence of routines to render the data object while
having to remember which VTK classes to use for file reading and dataset mapping.
An example can be found in [this creative commons VTK example](https://vtk.org/Wiki/VTK/Examples/Python/STLReader).

`vtki` includes plotting routines that are intended to be intuitive and
highly controllable with `matplotlib` [@matplotlib] similar syntax and keyword
arguments. These plotting routines are defined to make the process of
visualizing spatially referenced data straightforward and easily implemented
by novice programmers. Loading and rendering in `vtki` is implemented to take
only a few lines of code:

```python
# Obligatory set up code
import vtki
from vtki import examples
import numpy as np
# Set a document-friendly plotting theme
vtki.set_plot_theme('document')
vtki.rcParams['window_size'] = np.array([1024, 768]) * 2
```

```python
# Example data file name
filename = examples.planefile
# Read the data from the file
mesh = vtki.read(filename)
# Render the dataset
mesh.plot(show_edges=True, screenshot='./images/airplane.png')
```

![Example rendering of mesh loaded from a file](./images/airplane.png)
**Figure 2:** Rendering of an example mesh file included with `vtki`


Notably, the `vtki.plot()` convenience method is bound to each `vtki`
data object to make visual inspection of datasets easily performed. Other
plotting routines in `vtki` are available for creating integrated and
easily manipulated scenes via the `vtki.Plotter` and `vtki.BackgroundPlotter`
classes. Creating a rendering scene and altering its properties can be performed
with the following code in `vtki`:

```python
# Load a sample point cloud
point_cloud = examples.download_lidar()

# Instantiate a rendering scene
plotter = vtki.Plotter()
# Add spatial data to the scene
plotter.add_mesh(point_cloud, color='orange')
# Alter how the scene is rendered
plotter.enable_eye_dome_lighting()
# Show axes labels
plotter.show_grid()
# Render and display the scene
plotter.show(screenshot='./images/point_cloud.png')
```

![Rendering scene with eye dome lighting enabled](./images/point_cloud.png)
**Figure 3:** Rendering of an example point cloud dataset with a
non-photorealistic shading technique, Eye-Dome Lighting, applied to improve
depth perception [@edl].


## Data Types & Mesh Creation

Datasets are any spatially referenced information and usually consist of
geometrical representations of a surface or volume in 3D space.
In VTK, the abstract class `vtk.vtkDataSet` represents a set of common
functionality for spatially referenced datasets [@vtkbook].
In `vtki`, the common functionality shared across spatially referenced datasets
is shared in the `vtki.Common` class which holds methods and attributes for
quickly accessing scalar arrays associated with the dataset or easily inspecting
attributes of the dataset such as all the scalar names or number of points
present.

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


Creating mesh objects in VTK is also simplified by `vtki` by providing intuitive
initialization functions and attributes on the `vtki` classes that callback to
the original VTK data object. Loading files supported by the VTK library is also
simplified with a module level function to decide on the appropriate reader for
the file.

```python
filename = 'path/to/vtk/supported/file.ext'
mesh = vtki.read(filename)
```


## Accessing Common Analysis Routines

`vtki` wrapped data objects have a suite of common filters ready for immediate
use directly on the objects. These filters are commonly used algorithms in the
VTK library that have been made more accessible by binding a method to control
that algorithm directly onto all `vtki` datasets. These filtering algorithms are
held in the `vtki.DataSetFilters` class which is inherited by the `vtki.Common`
class giving all datasets a shared set of functionality.
Through the use of these bound methods, powerful VTK filtering algorithms can
be leveraged with intuitive control via keyword arguments.
These filters can be used by calling the filtering method directly from the
data object:

```python
# Load an example uniform grid
dataset = examples.load_uniform()
# Apply a threshold over a data range
threshed = dataset.threshold([100, 500]) # Figure 4 A
```

Above, an extracted version of the input dataset where the active scalar array
is between 100 and 500 is created in the new `threshed` object.
Available keyword arguments to control the filtering algorithms are described
in the documentation of each filtering method. Below is an example of several
common filtering algorthms.


```python
outline = dataset.outline()
contours = dataset.contour() # Figure 4 B
slices = dataset.slice_orthogonal() # Figure 4 C
glyphs = dataset.glyph(factor=1e-3, geom=vtki.Sphere()) # Figure 4 D

# Two by Two comparison
vtki.plot_compare_four(threshed, contours, slices, glyphs,
                        {'show_scalar_bar':False},
                        camera_position=[-2,5,3], outline=outline,
                        screenshot='./images/filters.png')
```

![Examples of common filters](./images/filters.png)
**Figure 4:** Examples of common filtering algorithms: A) threshold volume
extraction by scalar array, B) iso-contouring by scalar array, C) orthogonal
slicing through volume, and D) geometric glyphing at mesh nodes and scaled by
a scalar array.


### Filtering Chain

In VTK, filters are often used in a pipeline where each algorithm passes its
output to the next filtering algorithm [@vtkbook].
`vtki` mimics the filtering pipeline through a chain; attaching each filter to
the last filter. In the following example using the sample dataset from above,
several filters are chained together.

1. A threshold filter to extract a range of the active scalar array.
2. An elevation filter to generate scalar values corresponding to height.
3. A clip filter to cut the dataset in half.
4. Create three slices along each axial plane.

```python
# Apply a filtering chain
result = dataset.threshold([100, 500], invert=True).elevation().clip(normal='z').slice_orthogonal()
```


## Mentions

`vtki` is used extensively by the Air Force Research Labs (AFRL) for
data visualization and plotting.  These research articles include
figures visualizing 3D tessellated models generated from structured
light optical scanner and results from finite element analysis, both
of which are particularly suited to `vtki`.

[PVGeo](http://pvgeo.org) is Python package of VTK-based algorithms to analyze
geoscientific data and models. ``vtki`` is used to make the inputs and outputs
of PVGeo's algorithms more accessible and to streamline the process of
visualizing geoscientific data.


## References
