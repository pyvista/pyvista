"""Helper module to query vtk cell sizes upon importing."""
import vtk

vtkcell_types = [
    ['VTK_EMPTY_CELL', 'vtkEmptyCell'],
    ['VTK_VERTEX', 'vtkVertex'],
    ['VTK_POLY_VERTEX', 'vtkPolyVertex'],
    ['VTK_LINE', 'vtkLine'],
    ['VTK_POLY_LINE', 'vtkPolyLine'],
    ['VTK_TRIANGLE', 'vtkTriangle'],
    ['VTK_TRIANGLE_STRIP', 'vtkTriangleStrip'],
    ['VTK_POLYGON', 'vtkPolygon'],
    ['VTK_PIXEL', 'vtkPixel'],
    ['VTK_QUAD', 'vtkQuad'],
    ['VTK_TETRA', 'vtkTetra'],
    ['VTK_VOXEL', 'vtkVoxel'],
    ['VTK_HEXAHEDRON', 'vtkHexahedron'],
    ['VTK_WEDGE', 'vtkWedge'],
    ['VTK_PYRAMID', 'vtkPyramid'],
    ['VTK_PENTAGONAL_PRISM', 'vtkPentagonalPrism'],
    ['VTK_HEXAGONAL_PRISM', 'vtkHexagonalPrism'],
    ['VTK_QUADRATIC_EDGE', 'vtkQuadraticEdge'],
    ['VTK_QUADRATIC_TRIANGLE', 'vtkQuadraticTriangle'],
    ['VTK_QUADRATIC_QUAD', 'vtkQuadraticQuad'],
    ['VTK_QUADRATIC_POLYGON', 'vtkQuadraticPolygon'],
    ['VTK_QUADRATIC_TETRA', 'vtkQuadraticTetra'],
    ['VTK_QUADRATIC_HEXAHEDRON', 'vtkQuadraticHexahedron'],
    ['VTK_QUADRATIC_WEDGE', 'vtkQuadraticWedge'],
    ['VTK_QUADRATIC_PYRAMID', 'vtkQuadraticPyramid'],
    ['VTK_BIQUADRATIC_QUAD', 'vtkBiQuadraticQuad'],
    ['VTK_TRIQUADRATIC_HEXAHEDRON', 'vtkTriQuadraticHexahedron'],
    ['VTK_QUADRATIC_LINEAR_QUAD', 'vtkQuadraticLinearQuad'],
    ['VTK_QUADRATIC_LINEAR_WEDGE', 'vtkQuadraticLinearWedge'],
    ['VTK_BIQUADRATIC_QUADRATIC_WEDGE', 'vtkBiQuadraticQuadraticWedge'],
    ['VTK_BIQUADRATIC_QUADRATIC_HEXAHEDRON', 'vtkBiQuadraticQuadraticHexahedron'],
    ['VTK_BIQUADRATIC_TRIANGLE', 'vtkBiQuadraticTriangle'],
    ['VTK_CUBIC_LINE', 'vtkCubicLine'],
    ['VTK_CONVEX_POINT_SET', 'vtkConvexPointSet'],
    ['VTK_POLYHEDRON', 'vtkPolyhedron'],
    ['VTK_LAGRANGE_CURVE', 'vtkLagrangeCurve'],
    ['VTK_LAGRANGE_TRIANGLE', 'vtkLagrangeTriangle'],
    ['VTK_LAGRANGE_QUADRILATERAL', 'vtkLagrangeQuadrilateral'],
    ['VTK_LAGRANGE_HEXAHEDRON', 'vtkLagrangeHexahedron'],
    ['VTK_LAGRANGE_WEDGE', 'vtkLagrangeWedge'],
    ['VTK_BEZIER_CURVE', 'vtkBezierCurve'],
    ['VTK_BEZIER_TRIANGLE', 'vtkBezierTriangle'],
    ['VTK_BEZIER_QUADRILATERAL', 'vtkBezierQuadrilateral'],
    ['VTK_BEZIER_TETRAHEDRON', 'vtkBezierTetra'],
    ['VTK_BEZIER_HEXAHEDRON', 'vtkBezierHexahedron'],
    ['VTK_BEZIER_WEDGE', 'vtkBezierWedge']
]


# get the number of points in a cell for a given cell type
# compute this at runtime as this is version dependent
enum_cell_type_nr_points_map = {}
for cell_num_str, cell_str in vtkcell_types:
    if hasattr(vtk, cell_str) and hasattr(vtk, cell_num_str):
      try:
          cell_num = getattr(vtk, cell_num_str)
          n_points = getattr(vtk, cell_str)().GetNumberOfPoints()
          enum_cell_type_nr_points_map[cell_num] = n_points
      except:
          pass
