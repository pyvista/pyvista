"""Export a rendering window to a VTKjs file.

This module holds a set of tools for exporting a VTK rendering window
to a VTKjs file that can be viewed in a web browser.

PVGeo has a webveiwer_ set up to load these files.

.. _webviewer: http://viewer.pyvista.org


Much of this export script was adopted from the
`VTKjs export script for ParaView`_.

.. VTKjs export script for ParaView: https://github.com/Kitware/vtk-js/blob/master/Utilities/ParaView/export-scene-macro.py

The license for the original export script is as follows:

Copyright (c) 2016, Kitware Inc.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the <organization> nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT
HOLDER> BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""
import errno
import gzip
import hashlib
import json
import os
import shutil
import sys
import time
import zipfile

import vtk

FILENAME_EXTENSION = '.vtkjs'

arrayTypesMapping = '  bBhHiIlLfdL'  # last one is idtype

jsMapping = {
    'b': 'Int8Array',
    'B': 'Uint8Array',
    'h': 'Int16Array',
    'H': 'Int16Array',
    'i': 'Int32Array',
    'I': 'Uint32Array',
    'l': 'Int32Array',
    'L': 'Uint32Array',
    'f': 'Float32Array',
    'd': 'Float64Array'
}

writer_mapping = {}

# -----------------------------------------------------------------------------


def get_range_info(array, component):
    """Get the data range of the array's component."""
    r = array.GetRange(component)
    comp_range = {}
    comp_range['min'] = r[0]
    comp_range['max'] = r[1]
    comp_range['component'] = array.GetComponentName(component)
    return comp_range

# -----------------------------------------------------------------------------


def get_ref(dest_dir, md5):
    """Get reference."""
    ref = {}
    ref['id'] = md5
    ref['encode'] = 'BigEndian' if sys.byteorder == 'big' else 'LittleEndian'
    ref['basepath'] = dest_dir
    return ref

# -----------------------------------------------------------------------------


objIds = []  # type: ignore


def get_object_id(obj):
    """Get object identifier."""
    try:
        idx = objIds.index(obj)
        return idx + 1
    except ValueError:
        objIds.append(obj)
        return len(objIds)


# -----------------------------------------------------------------------------

def dump_data_array(dataset_dir, data_dir, array, root=None, compress=True):
    """Dump vtkjs data array."""
    if root is None:
        root = {}
    if not array:
        return None

    if array.GetDataType() == 12:
        # IdType need to be converted to Uint32
        array_size = array.GetNumberOfTuples() * array.GetNumberOfComponents()
        new_array = vtk.vtkTypeUInt32Array()
        new_array.SetNumberOfTuples(array_size)
        for i in range(array_size):
            new_array.SetValue(i, -1 if array.GetValue(i) < 0 else array.GetValue(i))
        pbuffer = memoryview(new_array)
    else:
        pbuffer = memoryview(array)

    pMd5 = hashlib.md5(pbuffer).hexdigest()
    ppath = os.path.join(data_dir, pMd5)
    with open(ppath, 'wb') as f:
        f.write(pbuffer)

    if compress:
        with open(ppath, 'rb') as f_in, gzip.open(os.path.join(data_dir, pMd5 + '.gz'), 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
        # Close then remove.
        os.remove(ppath)

    root['ref'] = get_ref(os.path.relpath(data_dir, dataset_dir), pMd5)
    root['vtkClass'] = 'vtkDataArray'
    root['name'] = array.GetName()
    root['dataType'] = jsMapping[arrayTypesMapping[array.GetDataType()]]
    root['numberOfComponents'] = array.GetNumberOfComponents()
    root['size'] = array.GetNumberOfComponents() * array.GetNumberOfTuples()
    root['ranges'] = []
    if root['numberOfComponents'] > 1:
        for i in range(root['numberOfComponents']):
            root['ranges'].append(get_range_info(array, i))
        root['ranges'].append(get_range_info(array, -1))
    else:
        root['ranges'].append(get_range_info(array, 0))

    return root

# -----------------------------------------------------------------------------


def dump_color_array(dataset_dir, data_dir, color_array_info, root=None, compress=True):
    """Dump vtkjs color array."""
    if root is None:
        root = {}
    root['pointData'] = {
        'vtkClass': 'vtkDataSetAttributes',
        "activeGlobalIds": -1,
        "activeNormals": -1,
        "activePedigreeIds": -1,
        "activeScalars": -1,
        "activeTCoords": -1,
        "activeTensors": -1,
        "activeVectors": -1,
        "arrays": []
    }
    root['cellData'] = {
        'vtkClass': 'vtkDataSetAttributes',
        "activeGlobalIds": -1,
        "activeNormals": -1,
        "activePedigreeIds": -1,
        "activeScalars": -1,
        "activeTCoords": -1,
        "activeTensors": -1,
        "activeVectors": -1,
        "arrays": []
    }
    root['fieldData'] = {
        'vtkClass': 'vtkDataSetAttributes',
        "activeGlobalIds": -1,
        "activeNormals": -1,
        "activePedigreeIds": -1,
        "activeScalars": -1,
        "activeTCoords": -1,
        "activeTensors": -1,
        "activeVectors": -1,
        "arrays": []
    }

    colorArray = color_array_info['colorArray']
    location = color_array_info['location']

    dumped_array = dump_data_array(dataset_dir, data_dir, colorArray, {}, compress)

    if dumped_array:
        root[location]['activeScalars'] = 0
        root[location]['arrays'].append({'data': dumped_array})

    return root

# -----------------------------------------------------------------------------


def dump_t_coords(dataset_dir, data_dir, dataset, root=None, compress=True):
    """Dump vtkjs texture coordinates."""
    if root is None:
        root = {}
    tcoords = dataset.GetPointData().GetTCoords()
    if tcoords:
        dumped_array = dump_data_array(dataset_dir, data_dir, tcoords, {}, compress)
        root['pointData']['activeTCoords'] = len(root['pointData']['arrays'])
        root['pointData']['arrays'].append({'data': dumped_array})

# -----------------------------------------------------------------------------


def dump_normals(dataset_dir, data_dir, dataset, root=None, compress=True):
    """Dump vtkjs normal vectors."""
    if root is None:
        root = {}
    normals = dataset.GetPointData().GetNormals()
    if normals:
        dumped_array = dump_data_array(dataset_dir, data_dir, normals, {}, compress)
        root['pointData']['activeNormals'] = len(root['pointData']['arrays'])
        root['pointData']['arrays'].append({'data': dumped_array})

# -----------------------------------------------------------------------------


def dump_all_arrays(dataset_dir, data_dir, dataset, root=None, compress=True):
    """Dump all data arrays to vtkjs."""
    if root is None:
        root = {}
    root['pointData'] = {
        'vtkClass': 'vtkDataSetAttributes',
        "activeGlobalIds": -1,
        "activeNormals": -1,
        "activePedigreeIds": -1,
        "activeScalars": -1,
        "activeTCoords": -1,
        "activeTensors": -1,
        "activeVectors": -1,
        "arrays": []
    }
    root['cellData'] = {
        'vtkClass': 'vtkDataSetAttributes',
        "activeGlobalIds": -1,
        "activeNormals": -1,
        "activePedigreeIds": -1,
        "activeScalars": -1,
        "activeTCoords": -1,
        "activeTensors": -1,
        "activeVectors": -1,
        "arrays": []
    }
    root['fieldData'] = {
        'vtkClass': 'vtkDataSetAttributes',
        "activeGlobalIds": -1,
        "activeNormals": -1,
        "activePedigreeIds": -1,
        "activeScalars": -1,
        "activeTCoords": -1,
        "activeTensors": -1,
        "activeVectors": -1,
        "arrays": []
    }

    # Point data
    pd = dataset.GetPointData()
    pd_size = pd.GetNumberOfArrays()
    for i in range(pd_size):
        array = pd.GetArray(i)
        if array:
            dumped_array = dump_data_array(
                dataset_dir, data_dir, array, {}, compress)
            root['pointData']['activeScalars'] = 0
            root['pointData']['arrays'].append({'data': dumped_array})

    # Cell data
    cd = dataset.GetCellData()
    cd_size = pd.GetNumberOfArrays()
    for i in range(cd_size):
        array = cd.GetArray(i)
        if array:
            dumped_array = dump_data_array(
                dataset_dir, data_dir, array, {}, compress)
            root['cellData']['activeScalars'] = 0
            root['cellData']['arrays'].append({'data': dumped_array})

    return root

# -----------------------------------------------------------------------------


def dump_poly_data(dataset_dir, data_dir, dataset, color_array_info, root=None, compress=True):
    """Dump poly data object to vtkjs."""
    if root is None:
        root = {}
    root['vtkClass'] = 'vtkPolyData'
    container = root

    # Points
    points = dump_data_array(dataset_dir, data_dir,
                             dataset.GetPoints().GetData(), {}, compress)
    points['vtkClass'] = 'vtkPoints'
    container['points'] = points

    # Cells
    _cells = container

    # Verts
    if dataset.GetVerts() and dataset.GetVerts().GetData().GetNumberOfTuples() > 0:
        _verts = dump_data_array(dataset_dir, data_dir,
                                 dataset.GetVerts().GetData(), {}, compress)
        _cells['verts'] = _verts
        _cells['verts']['vtkClass'] = 'vtkCellArray'

    # Lines
    if dataset.GetLines() and dataset.GetLines().GetData().GetNumberOfTuples() > 0:
        _lines = dump_data_array(dataset_dir, data_dir,
                                 dataset.GetLines().GetData(), {}, compress)
        _cells['lines'] = _lines
        _cells['lines']['vtkClass'] = 'vtkCellArray'

    # Polys
    if dataset.GetPolys() and dataset.GetPolys().GetData().GetNumberOfTuples() > 0:
        _polys = dump_data_array(dataset_dir, data_dir,
                                 dataset.GetPolys().GetData(), {}, compress)
        _cells['polys'] = _polys
        _cells['polys']['vtkClass'] = 'vtkCellArray'

    # Strips
    if dataset.GetStrips() and dataset.GetStrips().GetData().GetNumberOfTuples() > 0:
        _strips = dump_data_array(dataset_dir, data_dir,
                                  dataset.GetStrips().GetData(), {}, compress)
        _cells['strips'] = _strips
        _cells['strips']['vtkClass'] = 'vtkCellArray'

    dump_color_array(dataset_dir, data_dir, color_array_info, container, compress)

    # PointData TCoords
    dump_t_coords(dataset_dir, data_dir, dataset, container, compress)
    # dump_normals(dataset_dir, data_dir, dataset, container, compress)

    return root


# -----------------------------------------------------------------------------
writer_mapping['vtkPolyData'] = dump_poly_data
# -----------------------------------------------------------------------------


def dump_image_data(dataset_dir, data_dir, dataset, color_array_info, root=None, compress=True):
    """Dump image data object to vtkjs."""
    if root is None:
        root = {}
    root['vtkClass'] = 'vtkImageData'
    container = root

    container['spacing'] = dataset.GetSpacing()
    container['origin'] = dataset.GetOrigin()
    container['extent'] = dataset.GetExtent()

    dump_all_arrays(dataset_dir, data_dir, dataset, container, compress)

    return root


# -----------------------------------------------------------------------------
writer_mapping['vtkImageData'] = dump_image_data
# -----------------------------------------------------------------------------


def write_data_set(file_path, dataset, output_dir, color_array_info, new_name=None, compress=True):
    """Write dataset to vtkjs."""
    fileName = new_name if new_name else os.path.basename(file_path)
    dataset_dir = os.path.join(output_dir, fileName)
    data_dir = os.path.join(dataset_dir, 'data')

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    root = {}
    root['metadata'] = {}
    root['metadata']['name'] = fileName

    writer = writer_mapping[dataset.GetClassName()]
    if writer:
        writer(dataset_dir, data_dir, dataset, color_array_info, root, compress)
    else:
        print(dataset.GetClassName(), 'is not supported')

    with open(os.path.join(dataset_dir, "index.json"), 'w') as f:
        f.write(json.dumps(root, indent=2))

    return dataset_dir

### ----------------------------------------------------------------------- ###
###                          Main script contents                           ###
### ----------------------------------------------------------------------- ###

def mkdir_p(path):
    """Make directory at path."""
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def export_plotter_vtkjs(plotter, filename, compress_arrays=False):
    """Export a plotter's rendering window to the VTKjs format."""
    sceneName = os.path.split(filename)[1]
    doCompressArrays = compress_arrays

    # Generate timestamp and use it to make subdirectory within the top level output dir
    timeStamp = time.strftime("%a-%d-%b-%Y-%H-%M-%S")
    root_output_directory = os.path.split(filename)[0]
    output_dir = os.path.join(root_output_directory, timeStamp)
    mkdir_p(output_dir)

    renderers = plotter.ren_win.GetRenderers()

    scDirs = []
    sceneComponents = []
    textureToSave = {}

    for rIdx in range(renderers.GetNumberOfItems()):
        renderer = renderers.GetItemAsObject(rIdx)
        renProps = renderer.GetViewProps()
        for rpIdx in range(renProps.GetNumberOfItems()):
            renProp = renProps.GetItemAsObject(rpIdx)
            if not renProp.GetVisibility():
                continue
            if hasattr(renProp, 'GetMapper') and renProp.GetMapper() is not None:
                mapper = renProp.GetMapper()
                dataObject = mapper.GetInputDataObject(0, 0)
                dataset = None
                if dataObject is None:
                    continue
                if dataObject.IsA('vtkCompositeDataSet'):
                    if dataObject.GetNumberOfBlocks() == 1:
                        dataset = dataObject.GetBlock(0)
                    else:
                        gf = vtk.vtkCompositeDataGeometryFilter()
                        gf.SetInputData(dataObject)
                        gf.Update()
                        dataset = gf.GetOutput()
                else:
                    dataset = mapper.GetInput()

                if dataset and not isinstance(dataset, (vtk.vtkPolyData, vtk.vtkImageData)):
                    # All data must be PolyData surfaces
                    gf = vtk.vtkGeometryFilter()
                    gf.SetInputData(dataset)
                    gf.Update()
                    dataset = gf.GetOutputDataObject(0)


                if dataset:# and dataset.GetPoints(): # NOTE: vtkImageData does not have points
                    componentName = f'data_{rIdx}_{rpIdx}' # getComponentName(renProp)
                    scalarVisibility = mapper.GetScalarVisibility()
                    #arrayAccessMode = mapper.GetArrayAccessMode()
                    #colorArrayName = mapper.GetArrayName() #TODO: if arrayAccessMode == 1 else mapper.GetArrayId()
                    colorMode = mapper.GetColorMode()
                    scalarMode = mapper.GetScalarMode()
                    lookupTable = mapper.GetLookupTable()

                    dsAttrs = None
                    arrayLocation = ''

                    if scalarVisibility:
                        if scalarMode == 3 or scalarMode == 1:  # VTK_SCALAR_MODE_USE_POINT_FIELD_DATA or VTK_SCALAR_MODE_USE_POINT_DATA
                            dsAttrs = dataset.GetPointData()
                            arrayLocation = 'pointData'
                        # VTK_SCALAR_MODE_USE_CELL_FIELD_DATA or VTK_SCALAR_MODE_USE_CELL_DATA
                        elif scalarMode == 4 or scalarMode == 2:
                            dsAttrs = dataset.GetCellData()
                            arrayLocation = 'cellData'

                    colorArray = None
                    dataArray = None

                    if dsAttrs:
                        dataArray = dsAttrs.GetArray(0) # Force getting the active array

                    if dataArray:
                        # component = -1 => let specific instance get scalar from vector before mapping
                        colorArray = lookupTable.MapScalars(
                            dataArray, colorMode, -1)
                        colorArrayName = '__CustomRGBColorArray__'
                        colorArray.SetName(colorArrayName)
                        colorMode = 0
                    else:
                        colorArrayName = ''

                    color_array_info = {
                        'colorArray': colorArray,
                        'location': arrayLocation
                    }

                    scDirs.append(write_data_set('', dataset, output_dir,
                                                 color_array_info,
                                                 new_name=componentName,
                                                 compress=doCompressArrays))

                    # Handle texture if any
                    textureName = None
                    if renProp.GetTexture() and renProp.GetTexture().GetInput():
                        textureData = renProp.GetTexture().GetInput()
                        textureName = f'texture_{get_object_id(textureData)}'
                        textureToSave[textureName] = textureData

                    representation = renProp.GetProperty().GetRepresentation(
                    ) if hasattr(renProp, 'GetProperty') else 2
                    colorToUse = renProp.GetProperty().GetDiffuseColor(
                    ) if hasattr(renProp, 'GetProperty') else [1, 1, 1]
                    if representation == 1:
                        colorToUse = renProp.GetProperty().GetColor() if hasattr(
                            renProp, 'GetProperty') else [1, 1, 1]
                    pointSize = renProp.GetProperty().GetPointSize(
                    ) if hasattr(renProp, 'GetProperty') else 1.0
                    opacity = renProp.GetProperty().GetOpacity() if hasattr(
                        renProp, 'GetProperty') else 1.0
                    edgeVisibility = renProp.GetProperty().GetEdgeVisibility(
                    ) if hasattr(renProp, 'GetProperty') else False

                    p3dPosition = renProp.GetPosition() if renProp.IsA(
                        'vtkProp3D') else [0, 0, 0]
                    p3dScale = renProp.GetScale() if renProp.IsA(
                        'vtkProp3D') else [1, 1, 1]
                    p3dOrigin = renProp.GetOrigin() if renProp.IsA(
                        'vtkProp3D') else [0, 0, 0]
                    p3dRotateWXYZ = renProp.GetOrientationWXYZ(
                    ) if renProp.IsA('vtkProp3D') else [0, 0, 0, 0]

                    sceneComponents.append({
                        "name": componentName,
                        "type": "httpDataSetReader",
                        "httpDataSetReader": {
                            "url": componentName
                        },
                        "actor": {
                            "origin": p3dOrigin,
                            "scale": p3dScale,
                            "position": p3dPosition,
                        },
                        "actorRotation": p3dRotateWXYZ,
                        "mapper": {
                            "colorByArrayName": colorArrayName,
                            "colorMode": colorMode,
                            "scalarMode": scalarMode
                        },
                        "property": {
                            "representation": representation,
                            "edgeVisibility": edgeVisibility,
                            "diffuseColor": colorToUse,
                            "pointSize": pointSize,
                            "opacity": opacity
                        },
                        "lookupTable": {
                            "tableRange": lookupTable.GetRange(),
                            "hueRange": lookupTable.GetHueRange() if hasattr(lookupTable, 'GetHueRange') else [0.5, 0]
                        }
                    })

                    if textureName:
                        sceneComponents[-1]['texture'] = textureName

    # Save texture data if any
    for key, val in textureToSave.items():
        write_data_set('', val, output_dir, None, new_name=key,
                       compress=doCompressArrays)

    cameraClippingRange = plotter.camera.clipping_range

    sceneDescription = {
        "fetchGzip": doCompressArrays,
        "background": plotter.background_color,
        "camera": {
            "focalPoint": plotter.camera.focal_point,
            "position": plotter.camera.position,
            "viewUp": plotter.camera.up,
            "clippingRange": [elt for elt in cameraClippingRange],
        },
        "centerOfRotation": plotter.camera.focal_point,
        "scene": sceneComponents
    }

    indexFilePath = os.path.join(output_dir, 'index.json')
    with open(indexFilePath, 'w') as outfile:
        json.dump(sceneDescription, outfile, indent=4)

# -----------------------------------------------------------------------------

    # Now zip up the results and get rid of the temp directory
    sceneFileName = os.path.join(
        root_output_directory, f'{sceneName}{FILENAME_EXTENSION}')

    try:
        import zlib
        compression = zipfile.ZIP_DEFLATED
    except:
        compression = zipfile.ZIP_STORED

    zf = zipfile.ZipFile(sceneFileName, mode='w')

    try:
        for dirName, subdirList, fileList in os.walk(output_dir):
            for fname in fileList:
                fullPath = os.path.join(dirName, fname)
                relPath = f'{sceneName}/{os.path.relpath(fullPath, output_dir)}'
                zf.write(fullPath, arcname=relPath, compress_type=compression)
    finally:
        zf.close()

    shutil.rmtree(output_dir)

    print('Finished exporting dataset to: ', sceneFileName)


def convert_dropbox_url(url):
    """Convert dropbox url to direct download link."""
    return url.replace("https://www.dropbox.com", "https://dl.dropbox.com")


def generate_viewer_url(dataURL):
    """Generate viewer url with data link."""
    viewerURL = "http://viewer.pyvista.org/"
    return viewerURL + f'?fileURL={dataURL}'


def get_vtkjs_url(*args):
    """Provide shareable link from the vtkjs script.

    After using ``export_vtkjs()`` to create a ``.vtkjs`` file from a
    data scene which is uploaded to an online file hosting service like Dropbox,
    use this method to get a shareable link to that scene on the
    `PVGeo VTKjs viewer`_.

    .. _PVGeo VTKjs viewer: http://viewer.pyvista.org

    **Current file hosts supported:**

    - Dropbox

    Args:
        host (str): the name of the file hosting service.
        inURL (str): the web URL to the ``.vtkjs`` file.

    """
    if len(args) == 1:
        host = 'dropbox'
        inURL = args[0]
    elif len(args) == 2:
        host = args[0]
        inURL = args[1]
    else:
        raise TypeError('Arguments not understood.')
    if host.lower() == "dropbox":
        convertURL = convert_dropbox_url(inURL)
    else:
        print("--> Warning: Web host not specified or supported. URL is simply appended to standalone scene loader link.")
        convertURL = inURL
    #print("--> Your link: %s" % generate_viewer_url(convertURL))
    return generate_viewer_url(convertURL)
