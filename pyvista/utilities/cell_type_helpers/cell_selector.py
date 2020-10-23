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
  assert(hasattr(vtk, cell_string))
  return getattr(vtk, cell_string)

def convert_enum_string_to_cell_instancer(cell_enum_string):
  #assert(cell_enum_string in all_cell_strings)
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
