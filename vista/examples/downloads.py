"""Functions to download sampe datasets from the VTK data repository
"""
import shutil
import os
import sys
import vtk
import zipfile

import pyvista

# Helpers:

def delete_downloads():
    """Delete all downloaded examples to free space or update the files"""
    shutil.rmtree(pyvista.EXAMPLES_PATH)
    os.makedirs(pyvista.EXAMPLES_PATH)
    return True


def _decompress(filename):
    zip_ref = zipfile.ZipFile(filename, 'r')
    zip_ref.extractall(pyvista.EXAMPLES_PATH)
    return zip_ref.close()

def _get_vtk_file_url(filename):
    return 'https://github.com/pyvista/vtk-data/raw/master/Data/{}'.format(filename)

def _retrieve_file(url, filename):
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
    # Perfrom download
    saved_file, resp = urlretrieve(url)
    # new_name = saved_file.replace(os.path.basename(saved_file), os.path.basename(filename))
    shutil.move(saved_file, local_path)
    if pyvista.get_ext(local_path) in ['.zip']:
        _decompress(local_path)
        local_path = local_path[:-4]
    return local_path, resp

def _download_file(filename):
    url = _get_vtk_file_url(filename)
    return _retrieve_file(url, filename)

def _download_and_read(filename, texture=False):
    saved_file, _ = _download_file(filename)
    if texture:
        return pyvista.read_texture(saved_file)
    return pyvista.read(saved_file)


###############################################################################

def download_masonry_texture():
    return _download_and_read('masonry.bmp', texture=True)

def download_usa_texture():
    return _download_and_read('usa_image.jpg', texture=True)

def download_puppy_texture():
    return _download_and_read('puppy.jpg', texture=True)

def download_puppy():
    return _download_and_read('puppy.jpg')

def download_usa():
    return _download_and_read('usa.vtk')

def download_st_helens():
    return _download_and_read('SainteHelens.dem')

def download_bunny():
    return _download_and_read('bunny.ply')

def download_bunny_coarse():
    return _download_and_read('Bunny.vtp')

def download_cow():
    return _download_and_read('cow.vtp')

def download_cow_head():
    return _download_and_read('cowHead.vtp')

def download_faults():
    return _download_and_read('faults.vtk')

def download_tensors():
    return _download_and_read('tensors.vtk')

def download_head():
    _download_file('HeadMRVolume.raw')
    return _download_and_read('HeadMRVolume.mhd')

def download_bolt_nut():
    blocks = pyvista.MultiBlock()
    blocks['bolt'] =  _download_and_read('bolt.slc')
    blocks['nut'] = _download_and_read('nut.slc')
    return blocks

def download_clown():
    return _download_and_read('clown.facet')

def download_topo_global():
    return _download_and_read('EarthModels/ETOPO_10min_Ice.vtp')

def download_topo_land():
    return _download_and_read('EarthModels/ETOPO_10min_Ice_only-land.vtp')

def download_coastlines():
    return _download_and_read('EarthModels/Coastlines_Los_Alamos.vtp')

def download_knee():
    return _download_and_read('DICOM_KNEE.dcm')

def download_knee_full():
    return _download_and_read('vw_knee.slc')

def download_lidar():
    return _download_and_read('kafadar-lidar-interp.vtp')

def download_exodus():
    """Sample ExodusII data file"""
    return _download_and_read('mesh_fs8.exo')

def download_nefertiti():
    """Download mesh of Queen Nefertiti"""
    return _download_and_read('Nefertiti.obj.zip')

def download_blood_vessels():
    """data representing the bifurcation of blood vessels."""
    local_path, _ = _download_file('pvtu_blood_vessels/blood_vessels.zip')
    filename = os.path.join(local_path, 'T0000000500.pvtu')
    mesh = pyvista.read(filename)
    mesh.set_active_vectors('velocity')
    return mesh

def download_iron_pot():
    return _download_and_read('ironProt.vtk')

def download_tetrahedron():
    return _download_and_read('Tetrahedron.vtu')

def download_saddle_surface():
    return _download_and_read('InterpolatingOnSTL_final.stl')

def download_sparse_points():
    """Used with ``download_saddle_surface``"""
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
    return _download_and_read('fsu/footbones.ply')

def download_guitar():
    return _download_and_read('fsu/stratocaster.ply')

def download_quadratic_pyramid():
    return _download_and_read('QuadraticPyramid.vtu')

def download_bird():
    return _download_and_read('Pileated.jpg')

def download_bird_texture():
    return _download_and_read('Pileated.jpg', texture=True)

def download_office():
    return _download_and_read('office.binary.vtk')

def download_horse_points():
    return _download_and_read('horsePoints.vtp')

def download_horse():
    return _download_and_read('horse.vtp')

def download_cake_easy():
    return _download_and_read('cake_easy.jpg')

def download_cake_easy_texture():
    return _download_and_read('cake_easy.jpg', texture=True)

def download_rectilinear_grid():
    return _download_and_read('RectilinearGrid.vtr')

def download_gourds(zoom=False):
    if zoom:
        return _download_and_read('Gourds.png')
    return _download_and_read('Gourds2.jpg')

def download_gourds_texture(zoom=False):
    if zoom:
        return _download_and_read('Gourds.png', texture=True)
    return _download_and_read('Gourds2.jpg', texture=True)

def download_unstructured_grid():
    return _download_and_read('uGridEx.vtk')

def download_letter_k():
    return _download_and_read('k.vtk')

def download_letter_a():
    return _download_and_read('a.vtp')

def download_poly_line():
    return _download_and_read('polyline.vtk')

def download_cad_model():
    return _download_and_read('42400-IDGH.stl')

def download_frog():
    # TODO: there are other files with this
    return _download_and_read('Frog/frog.mhd')

def download_prostate():
    return _download_and_read('prostate.img')

def download_filled_contours():
    return _download_and_read('filledContours.vtp')

def download_doorman():
    # TODO: download textures as well
    return _download_and_read('doorman/doorman.obj')

def download_mug():
    return _download_and_read('mug.e')

def download_oblique_cone():
    return _download_and_read('ObliqueCone.vtp')

def download_emoji():
    return _download_and_read('emote.jpg')

def download_emoji_texture():
    return _download_and_read('emote.jpg', texture=True)

def download_teapot():
    return _download_and_read('teapot.g')

def download_brain():
    return _download_and_read('brain.vtk')

def download_structured_grid():
    return _download_and_read('StructuredGrid.vts')

def download_structured_grid_two():
    return _download_and_read('SampleStructGrid.vtk')

def download_trumpet():
    return _download_and_read('trumpet.obj')

def download_face():
    # TODO: there is a texture with this
    return _download_and_read('fran_cut.vtk')

def download_sky_box_nz():
    return _download_and_read('skybox-nz.jpg')

def download_sky_box_nz_texture():
    return _download_and_read('skybox-nz.jpg', texture=True)

def download_disc_quads():
    return _download_and_read('Disc_BiQuadraticQuads_0_0.vtu')

def download_honolulu():
    return _download_and_read('honolulu.vtk')

def download_motor():
    return _download_and_read('motor.g')

def download_tri_quadratic_hexahedron():
    return _download_and_read('TriQuadraticHexahedron.vtu')

def download_human():
    return _download_and_read('Human.vtp')

def download_vtk():
    return _download_and_read('vtk.vtp')

def download_spider():
    return _download_and_read('spider.ply')

def download_carotid():
    mesh = _download_and_read('carotid.vtk')
    mesh.set_active_scalar('scalars')
    mesh.set_active_vectors('vectors')
    return mesh

def download_blow():
    return _download_and_read('blow.vtk')

def download_shark():
    return _download_and_read('shark.ply')

def download_dragon():
    return _download_and_read('dragon.ply')

def download_armadillo():
    return _download_and_read('Armadillo.ply')

def download_gears():
    return _download_and_read('gears.stl')

def download_torso():
    return _download_and_read('Torso.vtp')

def download_kitchen(split=False):
    """Download structured grid of kitchen with velocity field. Use the
    ``split`` argument to extract all of the furniture in the kitchen.
    """
    mesh =  _download_and_read('kitchen.vtk')
    if not split:
        return mesh
    extents = {
        'door' : (27, 27, 14, 18, 0, 11),
        'window1' : (0, 0, 9, 18, 6, 12),
        'window2' : (5, 12, 23, 23, 6, 12),
        'klower1' : (17, 17, 0, 11, 0, 6),
        'klower2' : (19, 19, 0, 11, 0, 6),
        'klower3' : (17, 19, 0, 0, 0, 6),
        'klower4' : (17, 19, 11, 11, 0, 6),
        'klower5' : (17, 19, 0, 11, 0, 0),
        'klower6' : (17, 19, 0, 7, 6, 6),
        'klower7' : (17, 19, 9, 11, 6, 6),
        'hood1' : (17, 17, 0, 11, 11, 16),
        'hood2' : (19, 19, 0, 11, 11, 16),
        'hood3' : (17, 19, 0, 0, 11, 16),
        'hood4' : (17, 19, 11, 11, 11, 16),
        'hood5' : (17, 19, 0, 11, 16, 16),
        'cookingPlate' : (17, 19, 7, 9, 6, 6),
        'furniture' : (17, 19, 7, 9, 11, 11),
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
    """Download two meshes defining an electrical inverse problem. This contains
    a high resolution forward modeled mesh and a coarse inverse modeled mesh
    """
    local_path, _ = _download_file('dc-inversion.zip')
    filename = os.path.join(local_path, 'mesh-forward.vtu')
    fwd = pyvista.read(filename)
    fwd.set_active_scalar('Resistivity(log10)-fwd')
    filename = os.path.join(local_path, 'mesh-inverse.vtu')
    inv = pyvista.read(filename)
    inv.set_active_scalar('Resistivity(log10)')
    return pyvista.MultiBlock({'forward':fwd, 'inverse':inv})
