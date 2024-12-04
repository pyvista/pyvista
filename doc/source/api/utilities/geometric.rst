.. _geometry_api:

Geometry
--------
PyVista includes several functions to generate simple geometric
objects. The API reference for these functions is on this page. For
additional details see :ref:`geometric_example` example.

PyVista provides both geometric objects and geometric sources. These two types
serve similar but distinct purposes. Both allow you to create various types of
geometry, but they differ mainly in their level of abstraction and usage
scenarios.

If you are looking for an easy and Pythonic way to generate geometric shapes,
PyVista's geometric objects may be more appropriate. If you need more control
over the geometry or are building a complex VTK pipeline, then using geometric
sources would be more suitable.


Geometric Objects
~~~~~~~~~~~~~~~~~
* High-Level Abstraction: Geometric objects like Box, Circle, Arrow, etc., in
  PyVista are high-level Pythonic wrappers around VTK's underlying geometric
  constructs. They provide a more user-friendly way to create geometries and
  might include additional utility functions or attributes.
* Self-contained: Geometric objects are often intended to be stand-alone
  entities, fully encapsulating the properties of the shape they represent.
* Quick Prototyping: These are often used for quick and simple tasks where
  complex control over the geometry is not required.
* Specific to PyVista: While they may be built on top of VTK, these high-level
  abstractions might be unique to PyVista and not directly translatable to raw
  VTK code.

.. currentmodule:: pyvista

.. autosummary::
   :toctree: _autosummary

   Arrow
   Box
   Capsule
   Circle
   CircularArc
   CircularArcFromNormal
   Cone
   Cube
   Cylinder
   CylinderStructured
   Disc
   Dodecahedron
   Icosahedron
   Icosphere
   KochanekSpline
   Line
   MultipleLines
   Octahedron
   Plane
   PlatonicSolid
   Polygon
   Pyramid
   Rectangle
   SolidSphere
   SolidSphereGeneric
   Sphere
   Spline
   Superquadric
   Tetrahedron
   Text3D
   Triangle
   Tube
   Wavelet


Geometric Sources
~~~~~~~~~~~~~~~~~
Geometric sources are closer to the actual VTK pipeline. They serve as the
'source' nodes in a VTK pipeline and generate specific types of geometry.

* Pipeline Integration: These sources are meant to be integrated into a VTK
  pipeline, and their output can be directly connected to other pipeline stages
  like filters, mappers, etc.
* Fine Control: They often offer more parameters to control the geometry and
  may be more suitable for scenarios where you need to have fine-grained
  control over the generated geometry.
* VTK Compatible: Since they are closer to raw VTK, transitioning from PyVista
  to VTK or vice versa might be smoother when using geometric sources.

.. autosummary::
   :toctree: _autosummary

   ArrowSource
   AxesGeometrySource
   BoxSource
   ConeSource
   CubeSource
   CubeFacesSource
   CylinderSource
   DiscSource
   LineSource
   MultipleLinesSource
   OrthogonalPlanesSource
   PlaneSource
   PlatonicSolidSource
   PolygonSource
   SphereSource
   SuperquadricSource
   Text3DSource
