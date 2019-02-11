"""THis module holds a set of tools for exporting a VTK rendering window to
a VTKjs file that can be viewed in a web browser.

PVGeo has a webveiwer_ set up to load these files.

.. _webviewer: http://viewer.pvgeo.org


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

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""

import hashlib
import shutil
import gzip
import json
import errno
import time
import os
import sys
import vtk
import zipfile


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

writerMapping = {}

# -----------------------------------------------------------------------------


def get_range_info(array, component):
    r = array.GetRange(component)
    compRange = {}
    compRange['min'] = r[0]
    compRange['max'] = r[1]
    compRange['component'] = array.GetComponentName(component)
    return compRange

# -----------------------------------------------------------------------------


def get_ref(destDirectory, md5):
    ref = {}
    ref['id'] = md5
    ref['encode'] = 'BigEndian' if sys.byteorder == 'big' else 'LittleEndian'
    ref['basepath'] = destDirectory
    return ref

# -----------------------------------------------------------------------------


objIds = []


def get_object_id(obj):
    try:
        idx = objIds.index(obj)
        return idx + 1
    except ValueError:
        objIds.append(obj)
        return len(objIds)


# -----------------------------------------------------------------------------

def dump_data_array(datasetDir, dataDir, array, root={}, compress=True):
    if not array:
        return None

    if array.GetDataType() == 12:
        # IdType need to be converted to Uint32
        arraySize = array.GetNumberOfTuples() * array.GetNumberOfComponents()
        newArray = vtk.vtkTypeUInt32Array()
        newArray.SetNumberOfTuples(arraySize)
        for i in range(arraySize):
            newArray.SetValue(i, -1 if array.GetValue(i) <
                              0 else array.GetValue(i))
        pBuffer = memoryview(newArray)
    else:
        pBuffer = memoryview(array)

    pMd5 = hashlib.md5(pBuffer).hexdigest()
    pPath = os.path.join(dataDir, pMd5)
    with open(pPath, 'wb') as f:
        f.write(pBuffer)

    if compress:
        with open(pPath, 'rb') as f_in, gzip.open(os.path.join(dataDir, pMd5 + '.gz'), 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
            os.remove(pPath)

    root['ref'] = get_ref(os.path.relpath(dataDir, datasetDir), pMd5)
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


def dump_color_array(datasetDir, dataDir, colorArrayInfo, root={}, compress=True):
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

    colorArray = colorArrayInfo['colorArray']
    location = colorArrayInfo['location']

    dumpedArray = dump_data_array(datasetDir, dataDir, colorArray, {}, compress)

    if dumpedArray:
        root[location]['activeScalars'] = 0
        root[location]['arrays'].append({'data': dumpedArray})

    return root

# -----------------------------------------------------------------------------


def dump_t_coords(datasetDir, dataDir, dataset, root={}, compress=True):
    tcoords = dataset.GetPointData().GetTCoords()
    if tcoords:
        dumpedArray = dump_data_array(datasetDir, dataDir, tcoords, {}, compress)
        root['pointData']['activeTCoords'] = len(root['pointData']['arrays'])
        root['pointData']['arrays'].append({'data': dumpedArray})

# -----------------------------------------------------------------------------


def dump_normals(datasetDir, dataDir, dataset, root={}, compress=True):
    normals = dataset.GetPointData().GetNormals()
    if normals:
        dumpedArray = dump_data_array(datasetDir, dataDir, normals, {}, compress)
        root['pointData']['activeNormals'] = len(root['pointData']['arrays'])
        root['pointData']['arrays'].append({'data': dumpedArray})

# -----------------------------------------------------------------------------


def dump_all_arrays(datasetDir, dataDir, dataset, root={}, compress=True):
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
            dumpedArray = dump_data_array(
                datasetDir, dataDir, array, {}, compress)
            root['pointData']['activeScalars'] = 0
            root['pointData']['arrays'].append({'data': dumpedArray})

    # Cell data
    cd = dataset.GetCellData()
    cd_size = pd.GetNumberOfArrays()
    for i in range(cd_size):
        array = cd.GetArray(i)
        if array:
            dumpedArray = dump_data_array(
                datasetDir, dataDir, array, {}, compress)
            root['cellData']['activeScalars'] = 0
            root['cellData']['arrays'].append({'data': dumpedArray})

    return root

# -----------------------------------------------------------------------------


def dump_poly_data(datasetDir, dataDir, dataset, colorArrayInfo, root={}, compress=True):
    root['vtkClass'] = 'vtkPolyData'
    container = root

    # Points
    points = dump_data_array(datasetDir, dataDir,
                           dataset.GetPoints().GetData(), {}, compress)
    points['vtkClass'] = 'vtkPoints'
    container['points'] = points

    # Cells
    _cells = container

    # Verts
    if dataset.GetVerts() and dataset.GetVerts().GetData().GetNumberOfTuples() > 0:
        _verts = dump_data_array(datasetDir, dataDir,
                               dataset.GetVerts().GetData(), {}, compress)
        _cells['verts'] = _verts
        _cells['verts']['vtkClass'] = 'vtkCellArray'

    # Lines
    if dataset.GetLines() and dataset.GetLines().GetData().GetNumberOfTuples() > 0:
        _lines = dump_data_array(datasetDir, dataDir,
                               dataset.GetLines().GetData(), {}, compress)
        _cells['lines'] = _lines
        _cells['lines']['vtkClass'] = 'vtkCellArray'

    # Polys
    if dataset.GetPolys() and dataset.GetPolys().GetData().GetNumberOfTuples() > 0:
        _polys = dump_data_array(datasetDir, dataDir,
                               dataset.GetPolys().GetData(), {}, compress)
        _cells['polys'] = _polys
        _cells['polys']['vtkClass'] = 'vtkCellArray'

    # Strips
    if dataset.GetStrips() and dataset.GetStrips().GetData().GetNumberOfTuples() > 0:
        _strips = dump_data_array(datasetDir, dataDir,
                                dataset.GetStrips().GetData(), {}, compress)
        _cells['strips'] = _strips
        _cells['strips']['vtkClass'] = 'vtkCellArray'

    dump_color_array(datasetDir, dataDir, colorArrayInfo, container, compress)

    # PointData TCoords
    dump_t_coords(datasetDir, dataDir, dataset, container, compress)
    # dump_normals(datasetDir, dataDir, dataset, container, compress)

    return root


# -----------------------------------------------------------------------------
writerMapping['vtkPolyData'] = dump_poly_data
# -----------------------------------------------------------------------------


def dump_image_data(datasetDir, dataDir, dataset, colorArrayInfo, root={}, compress=True):
    root['vtkClass'] = 'vtkImageData'
    container = root

    container['spacing'] = dataset.GetSpacing()
    container['origin'] = dataset.GetOrigin()
    container['extent'] = dataset.GetExtent()

    dump_all_arrays(datasetDir, dataDir, dataset, container, compress)

    return root


# -----------------------------------------------------------------------------
writerMapping['vtkImageData'] = dump_image_data
# -----------------------------------------------------------------------------


def write_data_set(filePath, dataset, outputDir, colorArrayInfo, newDSName=None, compress=True):
    fileName = newDSName if newDSName else os.path.basename(filePath)
    datasetDir = os.path.join(outputDir, fileName)
    dataDir = os.path.join(datasetDir, 'data')

    if not os.path.exists(dataDir):
        os.makedirs(dataDir)

    root = {}
    root['metadata'] = {}
    root['metadata']['name'] = fileName

    writer = writerMapping[dataset.GetClassName()]
    if writer:
        writer(datasetDir, dataDir, dataset, colorArrayInfo, root, compress)
    else:
        print(dataObject.GetClassName(), 'is not supported')

    with open(os.path.join(datasetDir, "index.json"), 'w') as f:
        f.write(json.dumps(root, indent=2))

    return datasetDir

### ----------------------------------------------------------------------- ###
###                          Main script contents                           ###
### ----------------------------------------------------------------------- ###

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def export_plotter_vtkjs(plotter, filename, compress_arrays=False):
    """Export a plotter's rendering window to the VTKjs format.
    """
    sceneName = filename
    doCompressArrays = compress_arrays

    # Generate timestamp and use it to make subdirectory within the top level output dir
    timeStamp = time.strftime("%a-%d-%b-%Y-%H-%M-%S")
    root_output_directory = os.path.split(filename)[0]
    outputDir = os.path.join(root_output_directory, timeStamp)
    mkdir_p(outputDir)

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
                    componentName = 'data_%d_%d' % (
                        rIdx, rpIdx)  # getComponentName(renProp)
                    scalarVisibility = mapper.GetScalarVisibility()
                    arrayAccessMode = mapper.GetArrayAccessMode()
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

                    colorArrayInfo = {
                        'colorArray': colorArray,
                        'location': arrayLocation
                    }

                    scDirs.append(write_data_set('', dataset, outputDir, colorArrayInfo,
                                               newDSName=componentName, compress=doCompressArrays))

                    # Handle texture if any
                    textureName = None
                    if renProp.GetTexture() and renProp.GetTexture().GetInput():
                        textureData = renProp.GetTexture().GetInput()
                        textureName = 'texture_%d' % get_object_id(textureData)
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
                    ) if hasattr(renProp, 'GetProperty') else false

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
        write_data_set('', val, outputDir, None, newDSName=key,
                     compress=doCompressArrays)

    cameraClippingRange = plotter.renderer.GetActiveCamera().GetClippingRange()

    sceneDescription = {
      "fetchGzip": doCompressArrays,
      "background": plotter.background_color,
      "camera": {
        "focalPoint": plotter.camera.GetFocalPoint(),
        "position": plotter.camera.GetPosition(),
        "viewUp": plotter.camera.GetViewUp(),
        "clippingRange": [ elt for elt in cameraClippingRange ]
      },
      "centerOfRotation": plotter.camera.GetFocalPoint(),
      "scene": sceneComponents
    }

    indexFilePath = os.path.join(outputDir, 'index.json')
    with open(indexFilePath, 'w') as outfile:
      json.dump(sceneDescription, outfile, indent=4)

# -----------------------------------------------------------------------------

    # Now zip up the results and get rid of the temp directory
    sceneFileName = os.path.join(
        root_output_directory, '%s%s' % (sceneName, FILENAME_EXTENSION))

    try:
        import zlib
        compression = zipfile.ZIP_DEFLATED
    except:
        compression = zipfile.ZIP_STORED

    zf = zipfile.ZipFile(sceneFileName, mode='w')

    try:
        for dirName, subdirList, fileList in os.walk(outputDir):
            for fname in fileList:
                fullPath = os.path.join(dirName, fname)
                relPath = '%s/%s' % (sceneName,
                                     os.path.relpath(fullPath, outputDir))
                zf.write(fullPath, arcname=relPath, compress_type=compression)
    finally:
        zf.close()

    shutil.rmtree(outputDir)

    print('Finished exporting dataset to: ', sceneFileName)



def convert_dropbox_url(url):
    return url.replace("https://www.dropbox.com", "https://dl.dropbox.com")

def generate_viewer_url(dataURL):
    viewerURL = "http://viewer.pvgeo.org/"
    return viewerURL + '%s%s' % ("?fileURL=", dataURL)

def get_vtkjs_url(*args):
    """After using ``export_vtkjs()`` to create a ``.vtkjs`` file from a
    data scene which is uploaded to an online file hosting service like Dropbox,
    use this method to get a shareable link to that scene on the
    `PVGeo VTKjs viewer`_.

    .. _PVGeo VTKjs viewer: http://viewer.pvgeo.org

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
        raise RuntimeError('Arguments not understood.')
    if host.lower() == "dropbox":
        convertURL = convert_dropbox_url(inURL)
    else:
        print("--> Warning: Web host not specified or supported. URL is simply appended to standalone scene loader link.")
        convertURL = inURL
    #print("--> Your link: %s" % generate_viewer_url(convertURL))
    return generate_viewer_url(convertURL)
