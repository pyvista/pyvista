"""PyVista FEA

This example is a translation from a `dash-vtk` code described in that [repository](https://github.com/shkiefer/dash_vtk_unstructured) using trame and pyvista.
The data files can be found [here in the original project](https://github.com/shkiefer/dash_vtk_unstructured/tree/main/data).
"""

import io
import os

import numpy as np
import pandas as pd
from trame.app import get_server
from trame.ui.vuetify import SinglePageLayout
from trame.widgets import trame, vtk as vtk_widgets, vuetify
from vtkmodules.numpy_interface.dataset_adapter import numpyTovtkDataArray as np2da
from vtkmodules.vtkCommonCore import vtkIdList
from vtkmodules.vtkCommonDataModel import vtkCellArray
from vtkmodules.vtkFiltersCore import vtkThreshold

import pyvista as pv

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

VIEW_INTERACT = [
    {"button": 1, "action": "Rotate"},
    {"button": 2, "action": "Pan"},
    {"button": 3, "action": "Zoom", "scrollEnabled": True},
    {"button": 1, "action": "Pan", "alt": True},
    {"button": 1, "action": "Zoom", "control": True},
    {"button": 1, "action": "Pan", "shift": True},
    {"button": 1, "action": "Roll", "alt": True, "shift": True},
]

# -----------------------------------------------------------------------------
# Trame setup
# -----------------------------------------------------------------------------

server = get_server()
state, ctrl = server.state, server.controller

# -----------------------------------------------------------------------------
# PyVista pipeline
# -----------------------------------------------------------------------------

vtk_idlist = vtkIdList()
vtk_grid = pv.UnstructuredGrid()
vtk_filter = vtkThreshold()
vtk_filter.SetInputData(vtk_grid)
field_to_keep = "my_array"


@state.change("nodes_file", "elems_file", "field_file")
def update_grid(nodes_file, elems_file, field_file, **kwargs):
    state.picking_modes = []
    if not nodes_file:
        return

    if not elems_file:
        return

    nodes_bytes = nodes_file.get("content")
    elems_bytes = elems_file.get("content")

    if isinstance(nodes_bytes, list):
        nodes_bytes = b"".join(nodes_bytes)

    if isinstance(elems_bytes, list):
        elems_bytes = b"".join(elems_bytes)

    df_nodes = pd.read_csv(
        io.StringIO(nodes_bytes.decode("utf-8")),
        delim_whitespace=True,
        header=None,
        skiprows=1,
        names=["id", "x", "y", "z"],
    )

    df_nodes["id"] = df_nodes["id"].astype(int)
    df_nodes = df_nodes.set_index("id", drop=True)
    # fill missing ids in range as VTK uses position (index) to map cells to points
    df_nodes = df_nodes.reindex(
        np.arange(df_nodes.index.min(), df_nodes.index.max() + 1), fill_value=0
    )

    df_elems = pd.read_csv(
        io.StringIO(elems_bytes.decode("utf-8")),
        skiprows=1,
        header=None,
        delim_whitespace=True,
        engine="python",
        index_col=None,
    ).sort_values(0)
    # order: 0: eid, 1: eshape, 2+: nodes, iloc[:,0] is index
    df_elems.iloc[:, 0] = df_elems.iloc[:, 0].astype(int)

    n_nodes = df_elems.iloc[:, 1].map(lambda x: int("".join(i for i in x if i.isdigit())))
    df_elems.insert(2, "n_nodes", n_nodes)
    # fill missing ids in range as VTK uses position (index) to map data to cells
    new_range = np.arange(df_elems.iloc[:, 0].min(), df_elems.iloc[:, 0].max() + 1)
    df_elems = df_elems.set_index(0, drop=False).reindex(new_range, fill_value=0)

    # mapping specific to Ansys Mechanical data
    vtk_shape_id_map = {
        "Tet4": pv.CellType.TETRA,
        "Tet10": pv.CellType.QUADRATIC_TETRA,
        "Hex8": pv.CellType.HEXAHEDRON,
        "Hex20": pv.CellType.QUADRATIC_HEXAHEDRON,
        "Tri6": pv.CellType.QUADRATIC_TRIANGLE,
        "Quad8": pv.CellType.QUADRATIC_QUAD,
        "Tri3": pv.CellType.TRIANGLE,
        "Quad4": pv.CellType.QUAD,
        "Wed15": pv.CellType.QUADRATIC_WEDGE,
    }
    df_elems["cell_types"] = np.nan
    df_elems.loc[df_elems.loc[:, 0] > 0, "cell_types"] = df_elems.loc[
        df_elems.loc[:, 0] > 0, 1
    ].map(lambda x: vtk_shape_id_map[x.strip()] if x.strip() in vtk_shape_id_map.keys() else np.nan)
    df_elems = df_elems.dropna(subset=["cell_types"], axis=0)

    # convert dataframes to vtk-desired format
    points = df_nodes[["x", "y", "z"]].to_numpy()
    cell_types = df_elems["cell_types"].to_numpy()
    n_nodes = df_elems.loc[:, "n_nodes"].to_numpy()
    # subtract starting node id from all grid references in cells to avoid filling from 0 to first used node (in case mesh doesn't start at 1)
    p = df_elems.iloc[:, 3:-1].to_numpy() - df_nodes.index.min()
    # if you need to, re-order nodes here-ish
    a = np.hstack((n_nodes.reshape((len(n_nodes), 1)), p))
    # convert to flat numpy array
    cells = a.ravel()
    # remove nans (due to elements with different no. of nodes)
    cells = cells[np.logical_not(np.isnan(cells))]
    cells = cells.astype(int)

    # update grid
    vtk_grid.points = points

    vtk_cells = vtkCellArray()
    vtk_cells.SetCells(cell_types.shape[0], np2da(cells, array_type=12))
    vtk_grid.SetCells(np2da(cell_types, array_type=3), vtk_cells)

    # Add field if any
    if field_file:
        field_bytes = field_file.get("content")
        if isinstance(field_bytes, list):
            field_bytes = b"".join(field_bytes)
        df_elem_data = pd.read_csv(
            io.StringIO(field_bytes.decode("utf-8")),
            delim_whitespace=True,
            header=None,
            skiprows=1,
            names=["id", "val"],
        )
        df_elem_data = df_elem_data.sort_values("id").set_index("id", drop=True)
        # fill missing ids in range as VTK uses position (index) to map data to cells
        df_elem_data = df_elem_data.reindex(
            np.arange(df_elems.index.min(), df_elems.index.max() + 1), fill_value=0.0
        )
        np_val = df_elem_data["val"].to_numpy()
        # assign data to grid with the name 'my_array'
        vtk_array = np2da(np_val, name=field_to_keep)
        vtk_grid.GetCellData().SetScalars(vtk_array)
        state.full_range = vtk_array.GetRange()
        state.threshold_range = list(vtk_array.GetRange())
        state.picking_modes = ["hover"]

    ctrl.mesh_update()


@state.change("threshold_range")
def update_filter(threshold_range, **kwargs):
    vtk_filter.SetLowerThreshold(threshold_range[0])
    vtk_filter.SetUpperThreshold(threshold_range[1])
    ctrl.threshold_update()


def reset():
    state.update(
        {
            "mesh": None,
            "threshold": None,
            "nodes_file": None,
            "elems_file": None,
            "field_file": None,
        }
    )


@state.change("pick_data")
def update_tooltip(pick_data, pixel_ratio, **kwargs):
    state.tooltip = ""
    state.tooltip_style = {"display": "none"}
    data = pick_data

    if data:
        xyx = data["worldPosition"]
        idx = vtk_grid.FindPoint(xyx)
        field = vtk_grid.GetCellData().GetArray(0)
        if idx > -1 and field:
            messages = []
            vtk_grid.GetPointCells(idx, vtk_idlist)
            for i in range(vtk_idlist.GetNumberOfIds()):
                cell_idx = vtk_idlist.GetId(i)
                value = field.GetValue(cell_idx)
                value_str = f"{value:.2f}"
                messages.append(f"Scalar: {value_str}")

            if len(messages):
                x, y, z = data["displayPosition"]
                state.tooltip = messages[0]
                state.tooltip_style = {
                    "position": "absolute",
                    "left": f"{(x / pixel_ratio) + 10}px",
                    "bottom": f"{(y / pixel_ratio) + 10}px",
                    "zIndex": 10,
                    "pointerEvents": "none",
                }


# -----------------------------------------------------------------------------
# Web App setup
# -----------------------------------------------------------------------------

file_style = {
    "dense": True,
    "hide_details": True,
    "style": "max-width: 200px",
    "class": "mx-2",
    "small_chips": True,
    "clearable": ("false",),
    "accept": ".txt",
}

state.trame__title = "FEA - Mesh viewer"

with SinglePageLayout(server) as layout:
    layout.title.set_text("Mesh Viewer")
    layout.icon.click = reset

    # Let the server know the browser pixel ratio
    trame.ClientTriggers(mounted="pixel_ratio = window.devicePixelRatio")

    # Toolbar ----------------------------------------
    with layout.toolbar:
        vuetify.VSpacer()
        vuetify.VRangeSlider(
            thumb_size=16,
            thumb_label=True,
            label="Threshold",
            v_if=("threshold",),
            v_model=("threshold_range", [0, 1]),
            min=("full_range[0]",),
            max=("full_range[1]",),
            dense=True,
            hide_details=True,
            style="max-width: 400px",
        )
        vuetify.VFileInput(
            v_show=("!mesh",),
            prepend_icon="mdi-vector-triangle",
            v_model=("nodes_file", None),
            placeholder="Nodes",
            **file_style,
        )
        vuetify.VFileInput(
            v_show=("!mesh",),
            prepend_icon="mdi-dots-triangle",
            v_model=("elems_file", None),
            placeholder="Elements",
            **file_style,
        )
        vuetify.VFileInput(
            v_show=("!threshold",),
            prepend_icon="mdi-gradient",
            v_model=("field_file", None),
            placeholder="Field",
            **file_style,
        )
        with vuetify.VBtn(v_if=("mesh",), icon=True, click=ctrl.view_reset_camera):
            vuetify.VIcon("mdi-crop-free")

        vuetify.VProgressLinear(
            indeterminate=True, absolute=True, bottom=True, active=("trame__busy",)
        )

        trame.ClientStateChange(value="mesh", change=ctrl.view_reset_camera)

    # Content ----------------------------------------
    with layout.content:
        with vuetify.VContainer(
            fluid=True,
            classes="pa-0 fill-height",
            style="position: relative",
        ):
            with vtk_widgets.VtkView(
                ref="view",
                background=("[0.8, 0.8, 0.8]",),
                hover="pick_data = $event",
                picking_modes=("picking_modes", []),
                interactor_settings=("interactor_settings", VIEW_INTERACT),
            ) as view:
                ctrl.view_update = view.update
                ctrl.view_reset_camera = view.reset_camera
                with vtk_widgets.VtkGeometryRepresentation(
                    v_if=("mesh",),
                    property=(
                        """{
                            representation: threshold ? 1 : 2,
                            color: threshold ? [0.3, 0.3, 0.3] : [1, 1, 1],
                            opacity: threshold ? 0.2 : 1
                            }""",
                    ),
                ):
                    mesh = vtk_widgets.VtkMesh("mesh", dataset=vtk_grid)
                    ctrl.mesh_update = mesh.update

                with vtk_widgets.VtkGeometryRepresentation(
                    v_if=("threshold",),
                    color_data_range=("full_range", [0, 1]),
                ):
                    threshold = vtk_widgets.VtkMesh(
                        "threshold", dataset=vtk_filter, field_to_keep=field_to_keep
                    )
                    ctrl.threshold_update = threshold.update
            with vuetify.VCard(
                style=("tooltip_style", {"display": "none"}),
                elevation=2,
                outlined=True,
            ):
                vuetify.VCardText("<pre>{{ tooltip }}</pre>"),


# Variables not defined within HTML but used
state.update(
    {
        "pixel_ratio": 1,
        "pick_data": None,
        "tooltip": "",
    }
)

# -----------------------------------------------------------------------------
# Use --data to skip file upload
# -----------------------------------------------------------------------------

parser = server.cli
parser.add_argument("--data", help="Unstructured file path", dest="data")
args = parser.parse_args()
if args.data:
    from vtkmodules.vtkIOXML import vtkXMLUnstructuredGridReader

    reader = vtkXMLUnstructuredGridReader()
    reader.SetFileName(os.path.abspath(args.data))
    reader.Update()
    vtu = reader.GetOutput()
    vtk_grid.ShallowCopy(vtu)

    vtk_array = vtu.GetCellData().GetScalars()
    full_min, full_max = vtk_array.GetRange()
    state.full_range = [full_min, full_max]
    state.threshold_range = [full_min, full_max]
    state.picking_modes = ["hover"]
    ctrl.mesh_update()
    ctrl.threshold_update()

# -----------------------------------------------------------------------------

if __name__ == "__main__":
    server.start()
