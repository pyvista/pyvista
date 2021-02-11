"""Functions to download sample datasets from the VTK data repository."""

import os
import shutil
import sys
import zipfile

import numpy as np
import vtk

import pyvista
from pyvista.core.errors import DeprecationError

# Helpers:

def _check_examples_path():
    """Check if the examples path exists."""
    if not pyvista.EXAMPLES_PATH:
        raise FileNotFoundError('EXAMPLES_PATH does not exist.  Try setting the '
                                'environment variable `PYVISTA_USERDATA_PATH` '
                                'to a writable path and restarting python')


def delete_downloads():
    """Delete all downloaded examples to free space or update the files."""
    _check_examples_path()
    shutil.rmtree(pyvista.EXAMPLES_PATH)
    os.makedirs(pyvista.EXAMPLES_PATH)
    return True


def _decompress(filename):
    _check_examples_path()
    zip_ref = zipfile.ZipFile(filename, 'r')
    zip_ref.extractall(pyvista.EXAMPLES_PATH)
    return zip_ref.close()


def _get_vtk_file_url(filename):
    return f'https://github.com/pyvista/vtk-data/raw/master/Data/{filename}'


def _retrieve_file(url, filename):
    _check_examples_path()
    # First check if file has already been downloaded
    local_path = os.path.join(pyvista.EXAMPLES_PATH, os.path.basename(filename))
    local_path_no_zip = local_path.replace('.zip', '')
    if os.path.isfile(local_path_no_zip) or os.path.isdir(local_path_no_zip):
        return local_path_no_zip, None
    # grab the correct url retriever
    if sys.version_info < (3,):
        import urllib
        urlretrieve = urllib.urlretrieve
    else:
        import urllib.request
        urlretrieve = urllib.request.urlretrieve
    # Perform download
    saved_file, resp = urlretrieve(url)
    # new_name = saved_file.replace(os.path.basename(saved_file), os.path.basename(filename))
    # Make sure folder exists!
    if not os.path.isdir(os.path.dirname((local_path))):
        os.makedirs(os.path.dirname((local_path)))
    shutil.move(saved_file, local_path)
    if pyvista.get_ext(local_path) in ['.zip']:
        _decompress(local_path)
        local_path = local_path[:-4]
    return local_path, resp


def _download_file(filename):
    url = _get_vtk_file_url(filename)
    return _retrieve_file(url, filename)


def _download_and_read(filename, texture=False, file_format=None):
    saved_file, _ = _download_file(filename)
    if texture:
        return pyvista.read_texture(saved_file)
    return pyvista.read(saved_file, file_format=file_format)


###############################################################################

def download_masonry_texture():
    """Download masonry texture."""
    return _download_and_read('masonry.bmp', texture=True)


def download_usa_texture():
    """Download usa texture."""
    return _download_and_read('usa_image.jpg', texture=True)


def download_puppy_texture():
    """Download puppy texture."""
    return _download_and_read('puppy.jpg', texture=True)


def download_puppy():
    """Download puppy dataset."""
    return _download_and_read('puppy.jpg')


def download_usa():
    """Download usa dataset."""
    return _download_and_read('usa.vtk')


def download_st_helens():
    """Download Saint Helens dataset."""
    return _download_and_read('SainteHelens.dem')


def download_bunny():
    """Download bunny dataset."""
    return _download_and_read('bunny.ply')


def download_bunny_coarse():
    """Download coarse bunny dataset."""
    return _download_and_read('Bunny.vtp')


def download_cow():
    """Download cow dataset."""
    return _download_and_read('cow.vtp')


def download_cow_head():
    """Download cow head dataset."""
    return _download_and_read('cowHead.vtp')


def download_faults():
    """Download faults dataset."""
    return _download_and_read('faults.vtk')


def download_tensors():
    """Download tensors dataset."""
    return _download_and_read('tensors.vtk')


def download_head():
    """Download head dataset."""
    _download_file('HeadMRVolume.raw')
    return _download_and_read('HeadMRVolume.mhd')


def download_bolt_nut():
    """Download bolt nut dataset."""
    blocks = pyvista.MultiBlock()
    blocks['bolt'] = _download_and_read('bolt.slc')
    blocks['nut'] = _download_and_read('nut.slc')
    return blocks


def download_clown():
    """Download clown dataset."""
    return _download_and_read('clown.facet')


def download_topo_global():
    """Download topo dataset."""
    return _download_and_read('EarthModels/ETOPO_10min_Ice.vtp')


def download_topo_land():
    """Download topo land dataset."""
    return _download_and_read('EarthModels/ETOPO_10min_Ice_only-land.vtp')


def download_coastlines():
    """Download coastlines dataset."""
    return _download_and_read('EarthModels/Coastlines_Los_Alamos.vtp')


def download_knee():
    """Download knee dataset."""
    return _download_and_read('DICOM_KNEE.dcm')


def download_knee_full():
    """Download full knee dataset."""
    return _download_and_read('vw_knee.slc')


def download_lidar():
    """Download lidar dataset."""
    return _download_and_read('kafadar-lidar-interp.vtp')


def download_exodus():
    """Sample ExodusII data file."""
    return _download_and_read('mesh_fs8.exo')


def download_nefertiti():
    """Download mesh of Queen Nefertiti."""
    return _download_and_read('Nefertiti.obj.zip')


def download_blood_vessels():
    """Download data representing the bifurcation of blood vessels."""
    local_path, _ = _download_file('pvtu_blood_vessels/blood_vessels.zip')
    filename = os.path.join(local_path, 'T0000000500.pvtu')
    mesh = pyvista.read(filename)
    mesh.set_active_vectors('velocity')
    return mesh


def download_iron_pot():  # pragma: no cover
    """Download iron protein dataset.

    DEPRECATED: Please use ``download_iron_protein``.

    """
    raise DeprecationError('DEPRECATED: Please use ``download_iron_protein``')


def download_iron_protein():
    """Download iron protein dataset."""
    return _download_and_read('ironProt.vtk')


def download_tetrahedron():
    """Download tetrahedron dataset."""
    return _download_and_read('Tetrahedron.vtu')


def download_saddle_surface():
    """Download saddle surface dataset."""
    return _download_and_read('InterpolatingOnSTL_final.stl')


def download_sparse_points():
    """Download sparse points dataset.

    Used with ``download_saddle_surface``.

    """
    saved_file, _ = _download_file('sparsePoints.txt')
    points_reader = vtk.vtkDelimitedTextReader()
    points_reader.SetFileName(saved_file)
    points_reader.DetectNumericColumnsOn()
    points_reader.SetFieldDelimiterCharacters('\t')
    points_reader.SetHaveHeaders(True)
    table_points = vtk.vtkTableToPolyData()
    table_points.SetInputConnection(points_reader.GetOutputPort())
    table_points.SetXColumn('x')
    table_points.SetYColumn('y')
    table_points.SetZColumn('z')
    table_points.Update()
    return pyvista.wrap(table_points.GetOutput())


def download_foot_bones():
    """Download foot bones dataset."""
    return _download_and_read('fsu/footbones.ply')


def download_guitar():
    """Download guitar dataset."""
    return _download_and_read('fsu/stratocaster.ply')


def download_quadratic_pyramid():
    """Download quadratic pyramid dataset."""
    return _download_and_read('QuadraticPyramid.vtu')


def download_bird():
    """Download bird dataset."""
    return _download_and_read('Pileated.jpg')


def download_bird_texture():
    """Download bird texture."""
    return _download_and_read('Pileated.jpg', texture=True)


def download_office():
    """Download office dataset."""
    return _download_and_read('office.binary.vtk')


def download_horse_points():
    """Download horse points dataset."""
    return _download_and_read('horsePoints.vtp')


def download_horse():
    """Download horse dataset."""
    return _download_and_read('horse.vtp')


def download_cake_easy():
    """Download cake dataset."""
    return _download_and_read('cake_easy.jpg')


def download_cake_easy_texture():
    """Download cake texture."""
    return _download_and_read('cake_easy.jpg', texture=True)


def download_rectilinear_grid():
    """Download rectilinear grid dataset."""
    return _download_and_read('RectilinearGrid.vtr')


def download_gourds(zoom=False):
    """Download gourds dataset."""
    if zoom:
        return _download_and_read('Gourds.png')
    return _download_and_read('Gourds2.jpg')


def download_gourds_texture(zoom=False):
    """Download gourds texture."""
    if zoom:
        return _download_and_read('Gourds.png', texture=True)
    return _download_and_read('Gourds2.jpg', texture=True)


def download_unstructured_grid():
    """Download unstructured grid dataset."""
    return _download_and_read('uGridEx.vtk')


def download_letter_k():
    """Download letter k dataset."""
    return _download_and_read('k.vtk')


def download_letter_a():
    """Download letter a dataset."""
    return _download_and_read('a_grid.vtk')


def download_poly_line():
    """Download polyline dataset."""
    return _download_and_read('polyline.vtk')


def download_cad_model():
    """Download cad dataset."""
    return _download_and_read('42400-IDGH.stl')


def download_frog():
    """Download frog dataset."""
    # TODO: there are other files with this
    _download_file('froggy/frog.zraw')
    return _download_and_read('froggy/frog.mhd')


def download_prostate():
    """Download prostate dataset."""
    return _download_and_read('prostate.img')


def download_filled_contours():
    """Download filled contours dataset."""
    return _download_and_read('filledContours.vtp')


def download_doorman():
    """Download doorman dataset."""
    # TODO: download textures as well
    return _download_and_read('doorman/doorman.obj')


def download_mug():
    """Download mug dataset."""
    return _download_and_read('mug.e')


def download_oblique_cone():
    """Download oblique cone dataset."""
    return _download_and_read('ObliqueCone.vtp')


def download_emoji():
    """Download emoji dataset."""
    return _download_and_read('emote.jpg')


def download_emoji_texture():
    """Download emoji texture."""
    return _download_and_read('emote.jpg', texture=True)


def download_teapot():
    """Download teapot dataset."""
    return _download_and_read('teapot.g')


def download_brain():
    """Download brain dataset."""
    return _download_and_read('brain.vtk')


def download_structured_grid():
    """Download structured grid dataset."""
    return _download_and_read('StructuredGrid.vts')


def download_structured_grid_two():
    """Download structured grid two dataset."""
    return _download_and_read('SampleStructGrid.vtk')


def download_trumpet():
    """Download trumpet dataset."""
    return _download_and_read('trumpet.obj')


def download_face():
    """Download face dataset."""
    # TODO: there is a texture with this
    return _download_and_read('fran_cut.vtk')


def download_sky_box_nz():
    """Download skybox-nz dataset."""
    return _download_and_read('skybox-nz.jpg')


def download_sky_box_nz_texture():
    """Download skybox-nz texture."""
    return _download_and_read('skybox-nz.jpg', texture=True)


def download_disc_quads():
    """Download disc quads dataset."""
    return _download_and_read('Disc_BiQuadraticQuads_0_0.vtu')


def download_honolulu():
    """Download honolulu dataset."""
    return _download_and_read('honolulu.vtk')


def download_motor():
    """Download motor dataset."""
    return _download_and_read('motor.g')


def download_tri_quadratic_hexahedron():
    """Download tri quadratic hexahedron dataset."""
    return _download_and_read('TriQuadraticHexahedron.vtu')


def download_human():
    """Download human dataset."""
    return _download_and_read('Human.vtp')


def download_vtk():
    """Download vtk dataset."""
    return _download_and_read('vtk.vtp')


def download_spider():
    """Download spider dataset."""
    return _download_and_read('spider.ply')


def download_carotid():
    """Download carotid dataset."""
    mesh = _download_and_read('carotid.vtk')
    mesh.set_active_scalars('scalars')
    mesh.set_active_vectors('vectors')
    return mesh


def download_blow():
    """Download blow dataset."""
    return _download_and_read('blow.vtk')


def download_shark():
    """Download shark dataset."""
    return _download_and_read('shark.ply')


def download_dragon():
    """Download dragon dataset."""
    return _download_and_read('dragon.ply')


def download_armadillo():
    """Download armadillo dataset."""
    return _download_and_read('Armadillo.ply')


def download_gears():
    """Download gears dataset."""
    return _download_and_read('gears.stl')


def download_torso():
    """Download torso dataset."""
    return _download_and_read('Torso.vtp')


def download_kitchen(split=False):
    """Download structured grid of kitchen with velocity field.

    Use the ``split`` argument to extract all of the furniture in the kitchen.

    """
    mesh = _download_and_read('kitchen.vtk')
    if not split:
        return mesh
    extents = {
        'door': (27, 27, 14, 18, 0, 11),
        'window1': (0, 0, 9, 18, 6, 12),
        'window2': (5, 12, 23, 23, 6, 12),
        'klower1': (17, 17, 0, 11, 0, 6),
        'klower2': (19, 19, 0, 11, 0, 6),
        'klower3': (17, 19, 0, 0, 0, 6),
        'klower4': (17, 19, 11, 11, 0, 6),
        'klower5': (17, 19, 0, 11, 0, 0),
        'klower6': (17, 19, 0, 7, 6, 6),
        'klower7': (17, 19, 9, 11, 6, 6),
        'hood1': (17, 17, 0, 11, 11, 16),
        'hood2': (19, 19, 0, 11, 11, 16),
        'hood3': (17, 19, 0, 0, 11, 16),
        'hood4': (17, 19, 11, 11, 11, 16),
        'hood5': (17, 19, 0, 11, 16, 16),
        'cookingPlate': (17, 19, 7, 9, 6, 6),
        'furniture': (17, 19, 7, 9, 11, 11),
    }
    kitchen = pyvista.MultiBlock()
    for key, extent in extents.items():
        alg = vtk.vtkStructuredGridGeometryFilter()
        alg.SetInputDataObject(mesh)
        alg.SetExtent(extent)
        alg.Update()
        result = pyvista.filters._get_output(alg)
        kitchen[key] = result
    return kitchen


def download_tetra_dc_mesh():
    """Download two meshes defining an electrical inverse problem.

    This contains a high resolution forward modeled mesh and a coarse
    inverse modeled mesh.

    """
    local_path, _ = _download_file('dc-inversion.zip')
    filename = os.path.join(local_path, 'mesh-forward.vtu')
    fwd = pyvista.read(filename)
    fwd.set_active_scalars('Resistivity(log10)-fwd')
    filename = os.path.join(local_path, 'mesh-inverse.vtu')
    inv = pyvista.read(filename)
    inv.set_active_scalars('Resistivity(log10)')
    return pyvista.MultiBlock({'forward':fwd, 'inverse':inv})


def download_model_with_variance():
    """Download model with variance dataset."""
    return _download_and_read("model_with_variance.vtu")


def download_thermal_probes():
    """Download thermal probes dataset."""
    return _download_and_read("probes.vtp")


def download_carburator():
    """Download scan of a carburator.

    https://www.laserdesign.com/sample-files/carburetor/

    """
    url = "http://3dgallery.gks.com/2012/carburator/carburator2.php"
    filename, _ = _retrieve_file(url, 'carburator.ply')
    return pyvista.read(filename)


def download_woman():
    """Download scan of a woman.

    https://www.laserdesign.com/sample-files/full-body-scan-with-texture/

    """
    url = "http://3dgallery.gks.com/2012/bodyscan/bodyscan3.php"
    filename, _ = _retrieve_file(url, 'woman.stl')
    return pyvista.read(filename)


def download_lobster():
    """Download scan of a lobster.

    https://www.laserdesign.com/lobster-scan-data

    """
    url = "http://3dgallery.gks.com/2016/lobster/index2.php"
    filename, _ = _retrieve_file(url, 'lobster.ply')
    return pyvista.read(filename)


def download_face2():
    """Download scan of a man's face.

    https://www.laserdesign.com/sample-files/mans-face/

    """
    url = "http://3dgallery.gks.com/2012/face/"
    filename, _ = _retrieve_file(url, 'man_face.stl')
    return pyvista.read(filename)


def download_urn():
    """Download scan of a burial urn.

    https://www.laserdesign.com/sample-files/burial-urn/

    """
    url = "http://3dgallery.gks.com/2012/urn/urn3.php"
    filename, _ = _retrieve_file(url, 'urn.stl')
    return pyvista.read(filename)


def download_pepper():
    """Download scan of a pepper (capsicum).

    https://www.laserdesign.com/sample-files/hot-red-pepper/

    """
    url = "http://3dgallery.gks.com/2012/redpepper/redpepper2.php"
    filename, _ = _retrieve_file(url, 'pepper.ply')
    return pyvista.read(filename)


def download_drill():
    """Download scan of a power drill.

    https://www.laserdesign.com/drill-scan-data

    """
    url = "http://3dgallery.gks.com/2015/ryobi/index1.php"
    filename, _ = _retrieve_file(url, 'pepper.obj')
    return pyvista.read(filename)


def download_action_figure():
    """Download scan of an action figure.

    https://www.laserdesign.com/sample-files/action-figure/

    """
    url = "http://3dgallery.gks.com/2013/tigerfighter"
    filename, _ = _retrieve_file(url, 'tigerfighter.obj')
    return pyvista.read(filename)


def download_turbine_blade():
    """Download scan of a turbine blade.

    https://www.laserdesign.com/sample-files/blade/

    """
    url = "http://3dgallery.gks.com/2012/blade/blade.php"
    filename, _ = _retrieve_file(url, 'turbine_blade.stl')
    return pyvista.read(filename)


def download_pine_roots():
    """Download pine roots dataset."""
    return _download_and_read('pine_root.tri')


def download_crater_topo():
    """Download crater dataset."""
    return _download_and_read('Ruapehu_mag_dem_15m_NZTM.vtk')


def download_crater_imagery():
    """Download crater texture."""
    return _download_and_read('BJ34_GeoTifv1-04_crater_clip.tif', texture=True)


def download_dolfin():
    """Download dolfin mesh."""
    return _download_and_read('dolfin_fine.xml', file_format="dolfin-xml")


def download_damavand_volcano():
    """Download damavand volcano model."""
    volume = _download_and_read("damavand-volcano.vtk")
    volume.rename_array("None", "data")
    return volume


def download_delaunay_example():
    """Download a pointset for the Delaunay example."""
    return _download_and_read('250.vtk')


def download_embryo():
    """Download a volume of an embryo."""
    return _download_and_read('embryo.slc')


def download_antarctica_velocity():
    """Download the antarctica velocity simulation results."""
    return _download_and_read("antarctica_velocity.vtp")


def download_room_surface_mesh():
    """Download the room surface mesh.

    This mesh is for demonstrating the difference that depth peeling can
    provide whenn rendering translucent geometries.

    This mesh is courtesy of `Sam Potter <https://github.com/sampotter>`_.
    """
    return _download_and_read("room_surface_mesh.obj")


def download_beach():
    """Download the beach NRRD image."""
    return _download_and_read("beach.nrrd")


def download_rgba_texture():
    """Download a texture with an alpha channel."""
    return _download_and_read("alphachannel.png", texture=True)


def download_vtk_logo():
    """Download a texture of the VTK logo."""
    return _download_and_read("vtk.png", texture=True)


def download_backward_facing_step():
    """Download an ensight gold case of a fluid simulation."""
    folder, _ = _download_file('EnSight.zip')
    filename = os.path.join(folder, "foam_case_0_0_0_0.case")
    return pyvista.read(filename)


def download_gpr_data_array():
    """Download GPR example data array."""
    saved_file, _ = _download_file("gpr-example/data.npy")
    return np.load(saved_file)


def download_gpr_path():
    """Download GPR example path."""
    saved_file, _ = _download_file("gpr-example/path.txt")
    path = np.loadtxt(saved_file, skiprows=1)
    return pyvista.PolyData(path)
