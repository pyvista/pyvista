"""Helper module to query vtk cell sizes upon importing."""
import vtk
import re
import os

all_cell_types = []
all_cell_strings = []
enum_cell_type_map = {}
enum_cell_type_nr_points_map = {}
vtk_re_enums = re.compile("^(VTK_(.*?)) = (\d*),")
vtk_re_enums_wo_num = re.compile("^(VTK_(.*?))$")

def generate_all_cell_types():
  """Create the cell types by reading vtk_cell_types.txt.

  In this function, the cell types of the file vtk_cell_types are parsed line by line
  and actively looked up in the imported vtk module. If the cells are present,
  a dummy object is constructed and the number of points for the cell types are
  queried by calling GetNumberOfPoints. This information creates a mapping dictionary
  vtk type -> size for a quick lookup in get_mixed_cells and create_mixed_cells.

  See Also
  --------
    ..get_mixed_cells : Will fetch a cells dictionary for a given pyvista.UnstructuredGrid object
    ..create_mixed_cells : Creates a the necessary cell arrays from a given cells dictionary
  """
  # Using readlines()
  current_dir = os.path.dirname(__file__)
  cell_type_f = open(current_dir + '/vtk_cell_types.txt', 'r')
  lines = cell_type_f.readlines()

  # Strips the newline character
  for line in lines:
    matches = vtk_re_enums.match(line)

    #Comments and blank lines
    if not line.startswith("//") and matches is not None:
      enum_str = matches.group(1)

      try:
        enum_val = convert_cell_string_to_enum(enum_str)
        enum_cell_type_map[enum_val] = (convert_enum_string_to_cell_instancer(enum_str), enum_str)
        enum_cell_type_nr_points_map[enum_val] = enum_cell_type_map[enum_val][0]().GetNumberOfPoints()
        all_cell_types.append(enum_val)
        all_cell_strings.append(enum_str)

        assert(not enum_val in enum_cell_type_map)
      except AssertionError as err:
        pass #Continue
      except TypeError as err:
        if enum_val in enum_cell_type_map:
          del enum_cell_type_map[enum_val]

def convert_cell_string_to_enum(cell_string):
  """Convert strings of vtk types to the vtk enum.

  Parameters
  ----------
  cell_string : string
      string of the desired vtk type, e.g. "VTK_TRIANGLE"

  Returns
  -------
  vtk_enum : int
      The VTK enum value associated with cell_string

  Raises
  ------
    AssertionError
        If the searched vtk type could not be found
  """
  assert(hasattr(vtk, cell_string))
  return getattr(vtk, cell_string)

def convert_enum_string_to_cell_instancer(cell_enum_string):
  """Convert given string to a VTK cell object.

  This function is used to process each line vtk_cell_types.txt to convert the specified
  cell type, given as a string, to an instance of this cell type.
  Parameters
  ----------
  cell_enum_string : string
      An uppercase string of the desired cell type

  Returns
  -------
      A dummy instance of the cell type specified by cell_enum_string

  Raises
  ------
    AssertionError
        If the searched vtk type could not be found
  """
  matches = vtk_re_enums_wo_num.match(cell_enum_string)
  descr = matches.group(2)
  instancer_str = "".join(["vtk"] + [d.capitalize() for d in descr.split("_")])

  #Non obvious fixes to capitalization
  instancer_str = instancer_str.replace("Biquadratic", "BiQuadratic")
  instancer_str = instancer_str.replace("Triquadratic", "TriQuadratic")

  assert(hasattr(vtk, instancer_str))
  return getattr(vtk, instancer_str)

#Generate all cell types on importing this function
generate_all_cell_types()
