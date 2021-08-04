"""Serialize a pyvista plotter to a pythreejs scene."""

import warnings

import numpy as np
try:
    import pythreejs as tjs
except ImportError:
    raise ImportError('Please install pythreejs to use this feature')

import pyvista as pv


def segment_poly_cells(mesh):
    """Segment lines from a mesh into line segments."""
    if not pv.is_pyvista_dataset(mesh):
        mesh = pv.wrap(mesh)
    polylines = []
    i, offset = 0, 0
    cc = mesh.lines  # fetch up front
    while i < mesh.n_cells:
        nn = cc[offset]
        polylines.append(cc[offset+1:offset+1+nn])
        offset += nn + 1
        i += 1

    lines = []
    for poly in polylines:
        lines.append(np.column_stack((poly[:-1], poly[1:])))
    return np.vstack(lines)


def buffer_normals(trimesh):
    """Extract surface normals and return a buffer attribute."""
    if 'Normals' in trimesh.point_arrays:
        normals = trimesh.point_arrays['Normals']
    else:
        normals = trimesh.point_normals
    normals = normals.astype(np.float32, copy=False)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Given trait value dtype")
        return tjs.BufferAttribute(array=normals)


def get_coloring(mapper, dataset):
    """Return the three.js coloring type for a given actor."""
    coloring = 'NoColors'
    if mapper.GetScalarModeAsString() == 'UsePointData':
        scalars = dataset.point_arrays.active_scalars
        if scalars is not None:
            coloring = 'VertexColors'
    elif mapper.GetScalarModeAsString() == 'UseCellData':
        scalars = dataset.cell_arrays.active_scalars
        if scalars is not None:
            coloring = 'FaceColors'
    return coloring


def extract_surface_mesh(obj):
    """Extract a surface mesh from a pyvista or vtk dataset.

    Parameters
    ----------
    obj : pyvista compatible object
        Any object compatible with pyvista.  Includes most ``vtk``
        objects.

    Returns
    -------
    pyvista.PolyData
        Surface mesh

    """
        # attempt to wrap non-pyvista objects
    if not pv.is_pyvista_dataset(obj):  # pragma: no cover
        mesh = pv.wrap(obj)
        if not pv.is_pyvista_dataset(mesh):
            raise TypeError(f'Object type ({type(mesh)}) cannot be converted to '
                            'a pyvista dataset')
    else:
        mesh = obj

    if not isinstance(obj, pv.PolyData):
        # unlikely case that mesh does not have extract_surface
        if not hasattr(mesh, 'extract_surface'):  # pragma: no cover
            mesh = mesh.cast_to_unstructured_grid()
        return mesh.extract_surface()

    return mesh


def get_colors(inp, colormap="viridis", normalize=True, vmin=None, vmax=None):
    """Map scalars to a RGB array."""
    from matplotlib.pyplot import cm, Normalize
    colormap = cm.get_cmap(colormap)
    if normalize:
        vmin = np.min(inp)
        vmax = np.max(inp)

    norm = Normalize(vmin, vmax)
    return colormap(norm(inp))[:, :3]


def gen_circle(width=256, height=256):
    """Generate a binary map of a circle.

    Used for plotting circles within threejs instead of squares.

    """
    xx, yy = np.mgrid[:width, :height]
    circle = (xx - width/2 + 0.5) ** 2 + (yy - height/2 + 0.5) ** 2
    array = np.ones((width, height, 4), dtype=np.float32)
    array[:, :, 0] = (circle <= width)
    array[:, :, 1] = (circle <= width)
    array[:, :, 2] = (circle <= width)
    array[:, :, 3] = circle <= width
    return array


def array_to_float_buffer(points):
    """Convert a numpy array to a pythreejs compatible point buffer."""
    # create buffered points
    points = points.astype(np.float32, copy=False)

    # ignore invalid warning.  BufferAttribute type is None and is
    # improperly recognized as float64.  Same for the rest
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Given trait value dtype")
        position = tjs.BufferAttribute(array=points, normalized=False)
    return position


def cast_to_min_size(ind, max_index):
    """Return a buffered attribute of the minimum index size."""
    ind = ind.ravel()
    if max_index < np.iinfo(np.uint16).max:
        ind = ind.astype(np.uint16)
    elif max_index < np.iinfo(np.uint32).max:
        ind = ind.astype(np.uint32)
    else:
        ind = ind.astype(np.uint64)
    return tjs.BufferAttribute(array=ind, normalized=False)


def to_surf_mesh(surf, mapper, prop, add_attr={}):
    """Convert a pyvista surface to a buffer geometry.

    General Notes
    -------------

    * THREE.BufferGeometry expects position and index attributes
      representing a triangulated mesh points and face indices or just
      a position array representing individual faces of a mesh. 
    * The normals attribute is needed for physically based rendering,
      but not for the other mesh types.
    * Colors must be a RGB array with one value per point.

    Shading Notes
    -------------
    To match VTK, the following materials are used to match VTK's shading:

    * MeshPhysicalMaterial when physically based rendering is enabled
    * MeshPhongMaterial when physically based rendering is disabled,
      but lighting is enabled.
    * MeshBasicMaterial when lighting is disabled.

    """
    # convert to an all-triangular surface
    if surf.is_all_triangles():
        trimesh = surf
    else:
        trimesh = surf.triangulate()

    position = array_to_float_buffer(trimesh.points)

    # convert to minimum index type
    face_ind = trimesh.faces.reshape(-1, 4)[:, 1:]
    index = cast_to_min_size(face_ind, trimesh.n_points)
    attr = {'position': position,
            'index': index,
    }

    if prop.GetInterpolation():  # something other than flat shading
        attr['normal'] = buffer_normals(trimesh)

    # extract point/cell scalars for coloring
    colors = None
    if mapper.GetScalarModeAsString() == 'UsePointData':
        scalars = trimesh.point_arrays.active_scalars
        colors = get_colors(scalars, mapper.cmap).astype(np.float32, copy=False)
    elif mapper.GetScalarModeAsString() == 'UseCellData':
        # special handling for RGBA
        if mapper.GetColorMode() == 2:
            scalars = trimesh.cell_arrays.active_scalars.repeat(3, axis=0)
            scalars = scalars.astype(np.float32, copy=False)
            colors = scalars[:, :3]/255  # ignore alpha
        else:
            scalars = trimesh.cell_arrays.active_scalars.repeat(3)
            colors = get_colors(scalars, mapper.cmap).astype(np.float32, copy=False)
        position = array_to_float_buffer(trimesh.points[face_ind])
        attr = {'position': position}

    if colors is not None:
        attr['color'] = array_to_float_buffer(colors)

    surf_geo = tjs.BufferGeometry(attributes=attr)

    shared_attr = {'color': color_to_hex(prop.GetColor()),
                   'vertexColors': get_coloring(mapper, trimesh),
                   'wireframe': prop.GetRepresentation() == 1,
                   'opacity': prop.GetOpacity(),
                   'wireframeLinewidth': prop.GetLineWidth(),
                   # 'side': 'DoubleSide'
                   }

    if prop.GetInterpolation() == 3:  # using physically based rendering
        material = tjs.MeshPhysicalMaterial(flatShading=False,
                                            roughness=prop.GetRoughness(),
                                            metalness=prop.GetMetallic(),
                                            reflectivity=0,
                                            **shared_attr, **add_attr)
    elif prop.GetLighting():
        material = tjs.MeshPhongMaterial(shininess=0,
                                         flatShading=prop.GetInterpolation() == 0,
                                         specular=color_to_hex(prop.GetSpecularColor()),
                                         reflectivity=0,
                                         **shared_attr,
                                         **add_attr)
    else:  # no lighting
        material = tjs.MeshBasicMaterial(**shared_attr, **add_attr)

    return tjs.Mesh(geometry=surf_geo, material=material)



def to_edge_mesh(surf, mapper, prop, use_edge_coloring=True, use_lines=False):
    """Convert a pyvista surface to a three.js edge mesh."""
    # extract all edges from the surface.  Should not use triangular
    # mesh here as mesh may contain more than triangular faces
    if use_lines:
        edges_mesh = surf
        edges = segment_poly_cells(surf)
    else:
        edges_mesh = surf.extract_all_edges()
        edges = edges_mesh.lines.reshape(-1, 3)[:, 1:]

    attr = {'position': array_to_float_buffer(edges_mesh.points),
            'index': cast_to_min_size(edges, surf.n_points),
    }

    # add in colors
    coloring = get_coloring(mapper, surf)
    if coloring != 'NoColors' and not use_edge_coloring:
        if mapper.GetScalarModeAsString() == 'UsePointData':
            edge_scalars = edges_mesh.point_arrays.active_scalars

        edge_colors = get_colors(edge_scalars, mapper.cmap)
        attr['color'] = array_to_float_buffer(edge_colors)

    edge_geo = tjs.BufferGeometry(attributes=attr)

    mesh_attr = {}
    if coloring != 'NoColors':
        mesh_attr['vertexColors'] = coloring

    if use_edge_coloring:
        edge_color = prop.GetEdgeColor()
    else:
        edge_color = prop.GetColor()

    edge_mat = tjs.LineBasicMaterial(color=color_to_hex(edge_color),
                                     linewidth=prop.GetLineWidth(),
                                     opacity=prop.GetOpacity(),
                                     side='FrontSide',
                                     **mesh_attr)
    return tjs.LineSegments(edge_geo, edge_mat)


def to_tjs_points(dataset, mapper, prop, as_circles=True):
    """Extract the points from a dataset and return a buffered geometry."""
    attr = {
        'position': array_to_float_buffer(dataset.points),
    }

    coloring = get_coloring(mapper, dataset)
    if coloring != 'NoColors':
        colors = get_colors(dataset.point_arrays.active_scalars, mapper.cmap)
        attr['color'] = array_to_float_buffer(colors)

    geo = tjs.BufferGeometry(attributes=attr)

    m_attr = {'color': color_to_hex(prop.GetColor()),
              'size': prop.GetPointSize()/100,
              'vertexColors': coloring,
              }

    if as_circles:
        m_attr['size'] *= 5
        tex = tjs.DataTexture(data=gen_circle(128, 128),
                              format="RGBAFormat",
                              type="FloatType")

        m_attr["map"] = tex
        m_attr["alphaTest"] = 0.5
        m_attr["transparency"] = True

    point_mat = tjs.PointsMaterial(**m_attr)
    return tjs.Points(geo, point_mat)


def pvcamera_to_threejs_camera(pv_camera, lights, aspect):
    """Return an ipygany camera dict from a ``pyvista.Plotter`` object."""
    # scene will be centered at focal_point, so adjust the position
    position = np.array(pv_camera.position) - np.array(pv_camera.focal_point)
    far = np.linalg.norm(position)*2

    return tjs.PerspectiveCamera(up=pv_camera.up,
                                 children=lights,
                                 position=position.tolist(),
                                 fov=pv_camera.view_angle,
                                 aspect=aspect,
                                 far=far,
                                 )


def color_to_hex(color):
    """Convert a 0 - 1 RGB color tuple to a HTML hex color."""
    color = tuple((np.array(color)*255).astype(np.uint8))
    return '#%02x%02x%02x' % color


def pvlight_to_threejs_light(pvlight):
    """Convert a pyvista headlight into a three.js directional light."""
    if pvlight.is_camera_light or pvlight.is_headlight:
        # extend the position of the light to make "near infinite"
        position = np.array(pvlight.position)*100000
        return tjs.DirectionalLight(color=color_to_hex(pvlight.diffuse_color),
                                    position=position.tolist(),
                                    intensity=pvlight.intensity*0.8,
                                    )


def extract_lights_from_plotter(pl):
    """Extract and convert all pyvista lights to pythreejs compatible lights."""
    return [pvlight_to_threejs_light(pvlight) for pvlight in pl.renderer.lights]


def actor_to_mesh(actor, focal_point):
    """Convert a VTK actor to a threejs mesh or meshes."""
    mapper = actor.GetMapper()
    if mapper is None:
        return

    dataset = mapper.GetInputAsDataSet()
    has_faces = True
    if hasattr(dataset, 'faces'):
        has_faces = np.any(dataset.faces)

    prop = actor.GetProperty()
    rep_type = prop.GetRepresentationAsString()

    meshes = []
    if rep_type == 'Surface' and has_faces:
        surf = extract_surface_mesh(dataset)
        add_attr = {}
        if prop.GetEdgeVisibility():
            # must offset polygons to have mesh render property with lines
            add_attr = {'polygonOffset': True,
                        'polygonOffsetFactor': 1,
                        'polygonOffsetUnits': 1}

            meshes.append(to_edge_mesh(surf, mapper, prop, use_edge_coloring=True))

        meshes.append(to_surf_mesh(surf, mapper, prop, add_attr))

    elif rep_type == 'Points':
        meshes.append(to_tjs_points(dataset, mapper, prop))
    else:  # wireframe
        if has_faces:
            surf = extract_surface_mesh(dataset)
            mesh = to_edge_mesh(surf, mapper, prop, use_edge_coloring=False)
        elif np.any(dataset.lines):
            mesh = to_edge_mesh(dataset, mapper, prop, use_edge_coloring=False,
                                use_lines=True)
        else:  # empty mesh
            return

        meshes.append(mesh)

    # the camera in three.js has no concept of a "focal point".  In
    # three.js, the scene is always centered at the origin, which
    # serves as the focal point of the camera.  Therefore, we need to
    # shift the entire scene by the focal point of the pyvista camera
    for mesh in meshes:
        mesh.position = -focal_point[0], -focal_point[1], -focal_point[2]

    return meshes


def meshes_from_actors(actors, focal_point):
    """Convert a pyvista plotter to a scene."""
    meshes = []
    for actor in actors:
        mesh = actor_to_mesh(actor, focal_point)
        if mesh is not None:
            meshes.extend(mesh)

    return meshes


def renderer_from_plotter(pl):
    """Convert a pyvista Plotter to a pythreejs renderer."""
    # verify plotter hasn't been closed
    if not hasattr(pl, 'ren_win'):
        raise AttributeError('This plotter is closed and unable to export to html.\n'
                             'Please run this before showing or closing the plotter.')

    children = meshes_from_actors(pl.renderer.actors.values(),
                                  pl.camera.focal_point)
    lights = extract_lights_from_plotter(pl)

    width, height = pl.window_size
    aspect = width/height
    camera = pvcamera_to_threejs_camera(pl.camera, lights, aspect)

    children.append(camera)

    if pl.renderer.axes_enabled:
        children.append(tjs.AxesHelper(0.1))

    scene = tjs.Scene(children=children,
                      background=color_to_hex(pl.background_color)
    )

    renderer = tjs.Renderer(camera=camera,
                            scene=scene,
                            alpha=True,
                            clearOpacity=0,
                            controls=[tjs.OrbitControls(controlling=camera)],
                            width=width,
                            height=height,
                            antialias=pl.renderer.GetUseFXAA(),
    )

    # for now, we can't dynamically size the render windows.  If
    # unset, the renderer widget will attempt to resize and the
    # threejs renderer will not resize.
    renderer.layout.width = f'{width}px'
    renderer.layout.height = f'{height}px'

    return renderer
