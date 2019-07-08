"""Export PyVista scenes to K3D scenes: https://github.com/K3D-tools/K3D-jupyter
"""
import numpy as np
import vtk

import pyvista

from .colors import rgb_to_int, parse_color



def export_k3d(plotter, **kwargs):
    import k3d
    renderers = plotter.ren_win.GetRenderers()
    renderer = renderers.GetItemAsObject(0) # TODO

    bnds = plotter.bounds
    kgrid = bnds[0], bnds[2], bnds[4], bnds[1], bnds[3], bnds[5]

    background_color = rgb_to_int(parse_color('white'))

    k3d_scene = k3d.plot(axes=['xtitle', 'ytitle', 'ztitle'],
                         menu_visibility=True,
                         height=plotter.window_size[1]//2,
                         background_color=background_color,
                         grid_visible=True,
                         grid=kgrid,
                         lighting=1.0,
                        )

    ren_props = renderer.GetViewProps()

    for rpidx in range(ren_props.GetNumberOfItems()):
        actor = ren_props.GetItemAsObject(rpidx)
        property = actor.GetProperty()
        if not actor.GetVisibility():
            continue
        # krep = property.GetRepresentation()

        dataset = None
        if hasattr(actor, 'GetMapper') and actor.GetMapper() is not None:
            mapper = actor.GetMapper()
            dataObject = mapper.GetInputDataObject(0, 0)
            if dataObject is None:
                continue
            if dataObject.IsA('vtkCompositeDataSet'):
                if dataObject.GetNumberOfBlocks() == 1:
                    dataset = dataObject.GetBlock(0)
                else:
                    gf = vtk.vtkCompositeDataGeometryFilter()
                    gf.SetInputData(dataObject)
                    gf.Update()
                    dataset = pyvista.filters._get_output(gf)
            else:
                dataset = pyvista.wrap(mapper.GetInput())

            if dataset and not isinstance(dataset, (vtk.vtkPolyData,)):# TODO vtk.vtkImageData)):
                # All data must be PolyData surfaces
                gf = vtk.vtkGeometryFilter()
                gf.SetInputData(dataset)
                gf.Update()
                dataset = pyvista.filters._get_output(gf)

        if dataset is None:
            continue

        # NOTE: k3d cannot handle cell data - only points
        dataset = dataset.cell_data_to_point_data()

        name = '{}'.format(str(hex(id(dataset))))

        scalar_visibility = mapper.GetScalarVisibility()
        color_mode = mapper.GetColorMode()
        scalar_mode = mapper.GetScalarMode()
        lookup_table = mapper.GetLookupTable()

        kcmap = None
        color_attribute = None
        if scalar_visibility and scalar_mode in [0, 1]:
            lookup_table.Build()
            kcmap=[]
            nlut = lookup_table.GetNumberOfTableValues()
            for i in range(nlut):
                r,g,b,a = lookup_table.GetTableValue(i)
                kcmap += [i/nlut, r,g,b]
            color_attribute = (dataset.active_scalar_name, *dataset.get_data_range())
        else:
            # Solid color/no scalars plotted
            pass

        if isinstance(dataset, vtk.vtkPolyData):
            # TODO: Handle Points/lines differently than surface meshes
            if dataset.area > 0:
                kobj = k3d.vtk_poly_data(dataset,
                                         name=name,
                                         color=rgb_to_int(property.GetColor()),
                                         color_attribute=color_attribute,
                                         color_map=kcmap,
                                         opacity=property.GetOpacity(),
                                         # wireframe=(krep==1),
                                         )
                # Add data to the scene
                k3d_scene += kobj
            else:

                # Check if line/poly line cells
                if False:#TODO len(dataset.lines) > 0:
                    shader = 'thick' if property.GetRenderLinesAsTubes() else 'simple'
                    dataset = dataset.connectivity()
                    line_indices = dataset.point_arrays['RegionId']
                    for index in np.unique(line_indices):
                        subset = dataset.threshold([index-0.5, index+0.5], scalars='RegionId', preference='point')
                        colors = k3d.helpers.map_colors(subset.active_scalar, kcmap,
                                        subset.get_data_range()).astype(np.uint32)
                        kobj = k3d.line(subset.points,
                                        color=rgb_to_int(property.GetColor()),
                                        colors=colors,
                                        opacity=property.GetOpacity(),
                                        shader=shader,
                                        # width=property.GetLineWidth(), # TODO: this is not be done correctly
                                    )
                        # Add data to the scene
                        k3d_scene += kobj

                else: # handle like point cloud
                    colors = k3d.helpers.map_colors(dataset.active_scalar, kcmap,
                                    dataset.get_data_range()).astype(np.uint32)
                    shader = '3d' if property.GetRenderPointsAsSpheres() else 'flat'
                    kobj = k3d.points(dataset.points,
                                      color=rgb_to_int(property.GetColor()),
                                      colors=colors,
                                      opacity=property.GetOpacity(),
                                      shader=shader,
                                      point_size=property.GetPointSize(), # TODO: this might not be done correctly
                                      )
                    # Add data to the scene
                    k3d_scene += kobj


        elif isinstance(dataset, vtk.vtkImageData):
            raise NotImplementedError('ImageData not yet able to be sent to K3D.')
        else:
            raise NotImplementedError('Type ({}) not able to be sent to K3D'.format(type(dataset)))

        if property.GetInterpolation() == 0:
            kobj.flat_shading = True

    # All done.
    return k3d_scene
