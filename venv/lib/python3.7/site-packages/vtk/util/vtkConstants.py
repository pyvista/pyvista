"""
This file is obsolete.
All the constants are part of the base vtk module.
"""

# Some constants used throughout code

VTK_FLOAT_MAX = 1.0e+38
VTK_INT_MAX = 2147483647 # 2^31 - 1

# These types are returned by GetDataType to indicate pixel type.
VTK_VOID            = 0
VTK_BIT             = 1
VTK_CHAR            = 2
VTK_SIGNED_CHAR     =15
VTK_UNSIGNED_CHAR   = 3
VTK_SHORT           = 4
VTK_UNSIGNED_SHORT  = 5
VTK_INT             = 6
VTK_UNSIGNED_INT    = 7
VTK_LONG            = 8
VTK_UNSIGNED_LONG   = 9
VTK_FLOAT           =10
VTK_DOUBLE          =11
VTK_ID_TYPE         =12

# These types are not currently supported by GetDataType, but are
# for completeness.
VTK_STRING          =13
VTK_OPAQUE          =14

VTK_LONG_LONG          =16
VTK_UNSIGNED_LONG_LONG =17

# These types are required by vtkVariant and vtkVariantArray
VTK_VARIANT =20
VTK_OBJECT  =21

# Storage for Unicode strings
VTK_UNICODE_STRING  =22


# Some constant required for correct template performance
VTK_BIT_MIN = 0
VTK_BIT_MAX = 1
VTK_CHAR_MIN = -128
VTK_CHAR_MAX = 127
VTK_UNSIGNED_CHAR_MIN = 0
VTK_UNSIGNED_CHAR_MAX = 255
VTK_SHORT_MIN = -32768
VTK_SHORT_MAX = 32767
VTK_UNSIGNED_SHORT_MIN = 0
VTK_UNSIGNED_SHORT_MAX = 65535
VTK_INT_MIN = (-VTK_INT_MAX-1)
VTK_INT_MAX = VTK_INT_MAX
#VTK_UNSIGNED_INT_MIN = 0
#VTK_UNSIGNED_INT_MAX = 4294967295
VTK_LONG_MIN = (-VTK_INT_MAX-1)
VTK_LONG_MAX = VTK_INT_MAX
#VTK_UNSIGNED_LONG_MIN = 0
#VTK_UNSIGNED_LONG_MAX = 4294967295
VTK_FLOAT_MIN = -VTK_FLOAT_MAX
VTK_FLOAT_MAX = VTK_FLOAT_MAX
VTK_DOUBLE_MIN = -1.0e+99
VTK_DOUBLE_MAX  = 1.0e+99

# These types are returned to distinguish dataset types
VTK_POLY_DATA          = 0
VTK_STRUCTURED_POINTS  = 1
VTK_STRUCTURED_GRID    = 2
VTK_RECTILINEAR_GRID   = 3
VTK_UNSTRUCTURED_GRID  = 4
VTK_PIECEWISE_FUNCTION = 5
VTK_IMAGE_DATA         = 6
VTK_DATA_OBJECT        = 7
VTK_DATA_SET           = 8
VTK_POINT_SET          = 9
VTK_UNIFORM_GRID                  = 10
VTK_COMPOSITE_DATA_SET            = 11
VTK_MULTIGROUP_DATA_SET           = 12 # OBSOLETE
VTK_MULTIBLOCK_DATA_SET           = 13
VTK_HIERARCHICAL_DATA_SET         = 14 # OBSOLETE
VTK_HIERARCHICAL_BOX_DATA_SET     = 15
VTK_GENERIC_DATA_SET              = 16
VTK_HYPER_OCTREE                  = 17
VTK_TEMPORAL_DATA_SET             = 18
VTK_TABLE                         = 19
VTK_GRAPH                         = 20
VTK_TREE                          = 21
VTK_SELECTION                     = 22

# These types define error codes for vtk functions
VTK_OK                 = 1
VTK_ERROR              = 2

# These types define different text properties
VTK_ARIAL        = 0
VTK_COURIER      = 1
VTK_TIMES        = 2
VTK_UNKNOWN_FONT = 3

VTK_TEXT_LEFT     = 0
VTK_TEXT_CENTERED = 1
VTK_TEXT_RIGHT    = 2

VTK_TEXT_BOTTOM   = 0
VTK_TEXT_TOP      = 2

VTK_TEXT_GLOBAL_ANTIALIASING_SOME  = 0
VTK_TEXT_GLOBAL_ANTIALIASING_NONE  = 1
VTK_TEXT_GLOBAL_ANTIALIASING_ALL   = 2

VTK_LUMINANCE        = 1
VTK_LUMINANCE_ALPHA  = 2
VTK_RGB              = 3
VTK_RGBA             = 4

VTK_COLOR_MODE_DEFAULT     = 0
VTK_COLOR_MODE_MAP_SCALARS = 1

# Constants for InterpolationType
VTK_NEAREST_INTERPOLATION      = 0
VTK_LINEAR_INTERPOLATION       = 1

# For volume rendering
VTK_MAX_VRCOMP    = 4

# These types define the 17 linear VTK Cell Types
# See Filtering/vtkCellType.h

# Linear cells
VTK_EMPTY_CELL       = 0
VTK_VERTEX           = 1
VTK_POLY_VERTEX      = 2
VTK_LINE             = 3
VTK_POLY_LINE        = 4
VTK_TRIANGLE         = 5
VTK_TRIANGLE_STRIP   = 6
VTK_POLYGON          = 7
VTK_PIXEL            = 8
VTK_QUAD             = 9
VTK_TETRA            = 10
VTK_VOXEL            = 11
VTK_HEXAHEDRON       = 12
VTK_WEDGE            = 13
VTK_PYRAMID          = 14
VTK_PENTAGONAL_PRISM = 15
VTK_HEXAGONAL_PRISM  = 16

# Quadratic, isoparametric cells
VTK_QUADRATIC_EDGE                   = 21
VTK_QUADRATIC_TRIANGLE               = 22
VTK_QUADRATIC_QUAD                   = 23
VTK_QUADRATIC_TETRA                  = 24
VTK_QUADRATIC_HEXAHEDRON             = 25
VTK_QUADRATIC_WEDGE                  = 26
VTK_QUADRATIC_PYRAMID                = 27
VTK_BIQUADRATIC_QUAD                 = 28
VTK_TRIQUADRATIC_HEXAHEDRON          = 29
VTK_QUADRATIC_LINEAR_QUAD            = 30
VTK_QUADRATIC_LINEAR_WEDGE           = 31
VTK_BIQUADRATIC_QUADRATIC_WEDGE      = 32
VTK_BIQUADRATIC_QUADRATIC_HEXAHEDRON = 33

# Special class of cells formed by convex group of points
VTK_CONVEX_POINT_SET = 41

# Higher order cells in parametric form
VTK_PARAMETRIC_CURVE        = 51
VTK_PARAMETRIC_SURFACE      = 52
VTK_PARAMETRIC_TRI_SURFACE  = 53
VTK_PARAMETRIC_QUAD_SURFACE = 54
VTK_PARAMETRIC_TETRA_REGION = 55
VTK_PARAMETRIC_HEX_REGION   = 56

# Higher order cells
VTK_HIGHER_ORDER_EDGE        = 60
VTK_HIGHER_ORDER_TRIANGLE    = 61
VTK_HIGHER_ORDER_QUAD        = 62
VTK_HIGHER_ORDER_POLYGON     = 63
VTK_HIGHER_ORDER_TETRAHEDRON = 64
VTK_HIGHER_ORDER_WEDGE       = 65
VTK_HIGHER_ORDER_PYRAMID     = 66
VTK_HIGHER_ORDER_HEXAHEDRON  = 67

# A macro to get the name of a type

__vtkTypeNameDict = {VTK_VOID:"void",
                     VTK_DOUBLE:"double",
                     VTK_FLOAT:"float",
                     VTK_LONG:"long",
                     VTK_UNSIGNED_LONG:"unsigned long",
                     VTK_INT:"int",
                     VTK_UNSIGNED_INT:"unsigned int",
                     VTK_SHORT:"short",
                     VTK_UNSIGNED_SHORT:"unsigned short",
                     VTK_CHAR:"char",
                     VTK_UNSIGNED_CHAR:"unsigned char",
                     VTK_SIGNED_CHAR:"signed char",
                     VTK_BIT:"bit"}

def vtkImageScalarTypeNameMacro(type):
  return __vtkTypeNameDict[type]


