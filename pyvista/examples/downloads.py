"""Functions to download sample datasets from the VTK data repository."""

from functools import partial
import os
import shutil
import sys
import zipfile

import numpy as np

import pyvista
from pyvista import _vtk


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


def _http_request(url):
    # grab the correct url retriever
    if sys.version_info < (3,):
        import urllib
        urlretrieve = urllib.urlretrieve
    else:
        import urllib.request
        urlretrieve = urllib.request.urlretrieve
    # Perform download
    return urlretrieve(url)


def _repo_file_request(repo_path, filename):
    return os.path.join(repo_path, 'Data', filename), None


def _retrieve_file(retriever, filename):
    """Retrieve file and cache it in pyvsita.EXAMPLES_PATH.

    Parameters
    ----------
    retriever : str or callable
        If str, it is treated as a url.
        If callable, the function must take no arguments and must
        return a tuple like (file_path, resp), where file_path is
        the path to the file to use.
    filename : str
        The name of the file
    """
    _check_examples_path()
    # First check if file has already been downloaded
    local_path = os.path.join(pyvista.EXAMPLES_PATH, os.path.basename(filename))
    local_path_no_zip = local_path.replace('.zip', '')
    if os.path.isfile(local_path_no_zip) or os.path.isdir(local_path_no_zip):
        return local_path_no_zip, None
    if isinstance(retriever, str):
        retriever = partial(_http_request, retriever)
    saved_file, resp = retriever()
    # new_name = saved_file.replace(os.path.basename(saved_file), os.path.basename(filename))
    # Make sure folder exists!
    if not os.path.isdir(os.path.dirname((local_path))):
        os.makedirs(os.path.dirname((local_path)))
    if pyvista.VTK_DATA_PATH is None:
        shutil.move(saved_file, local_path)
    else:
        shutil.copy(saved_file, local_path)
    if pyvista.get_ext(local_path) in ['.zip']:
        _decompress(local_path)
        local_path = local_path[:-4]
    return local_path, resp


def _download_file(filename):
    if pyvista.VTK_DATA_PATH is None:
        url = _get_vtk_file_url(filename)
        retriever = partial(_http_request, url)
    else:
        if not os.path.isdir(pyvista.VTK_DATA_PATH):
            raise FileNotFoundError(f'VTK data repository path does not exist at:\n\n{pyvista.VTK_DATA_PATH}')
        if not os.path.isdir(os.path.join(pyvista.VTK_DATA_PATH, 'Data')):
            raise FileNotFoundError(f'VTK data repository does not have "Data" folder at:\n\n{pyvista.VTK_DATA_PATH}')
        retriever = partial(_repo_file_request, pyvista.VTK_DATA_PATH, filename)
    return _retrieve_file(retriever, filename)


def _download_and_read(filename, texture=False, file_format=None, load=True):
    saved_file, _ = _download_file(filename)
    if not load:
        return saved_file
    if texture:
        return pyvista.read_texture(saved_file)
    return pyvista.read(saved_file, file_format=file_format)


###############################################################################

def download_masonry_texture(load=True):
    """Download masonry texture."""
    return _download_and_read('masonry.bmp', texture=True, load=load)


def download_usa_texture(load=True):
    """Download usa texture."""
    return _download_and_read('usa_image.jpg', texture=True, load=load)


def download_puppy_texture(load=True):
    """Download puppy texture."""
    return _download_and_read('puppy.jpg', texture=True, load=load)


def download_puppy(load=True):
    """Download puppy dataset."""
    return _download_and_read('puppy.jpg', load=load)


def download_usa(load=True):
    """Download usa dataset."""
    return _download_and_read('usa.vtk', load=load)


def download_st_helens(load=True):
    """Download Saint Helens dataset."""
    return _download_and_read('SainteHelens.dem', load=load)


def download_bunny(load=True):
    """Download bunny dataset."""
    return _download_and_read('bunny.ply', load=load)


def download_bunny_coarse(load=True):
    """Download coarse bunny dataset."""
    return _download_and_read('Bunny.vtp', load=load)


def download_cow(load=True):
    """Download cow dataset."""
    return _download_and_read('cow.vtp', load=load)


def download_cow_head(load=True):
    """Download cow head dataset."""
    return _download_and_read('cowHead.vtp', load=load)


def download_faults(load=True):
    """Download faults dataset."""
    return _download_and_read('faults.vtk', load=load)


def download_tensors(load=True):
    """Download tensors dataset."""
    return _download_and_read('tensors.vtk', load=load)


def download_head(load=True):
    """Download head dataset."""
    _download_file('HeadMRVolume.raw')
    return _download_and_read('HeadMRVolume.mhd', load=load)


def download_bolt_nut(load=True):
    """Download bolt nut dataset."""
    if not load:
        return (
            _download_and_read('bolt.slc', load=load),
            _download_and_read('nut.slc', load=load)
        )
    blocks = pyvista.MultiBlock()
    blocks['bolt'] = _download_and_read('bolt.slc')
    blocks['nut'] = _download_and_read('nut.slc')
    return blocks


def download_clown(load=True):
    """Download clown dataset."""
    return _download_and_read('clown.facet', load=load)


def download_topo_global(load=True):
    """Download topo dataset."""
    return _download_and_read('EarthModels/ETOPO_10min_Ice.vtp', load=load)


def download_topo_land(load=True):
    """Download topo land dataset."""
    return _download_and_read('EarthModels/ETOPO_10min_Ice_only-land.vtp', load=load)


def download_coastlines(load=True):
    """Download coastlines dataset."""
    return _download_and_read('EarthModels/Coastlines_Los_Alamos.vtp', load=load)


def download_knee(load=True):
    """Download knee dataset."""
    return _download_and_read('DICOM_KNEE.dcm', load=load)


def download_knee_full(load=True):
    """Download full knee dataset."""
    return _download_and_read('vw_knee.slc', load=load)


def download_lidar(load=True):
    """Download lidar dataset."""
    return _download_and_read('kafadar-lidar-interp.vtp', load=load)


def download_exodus(load=True):
    """Sample ExodusII data file."""
    return _download_and_read('mesh_fs8.exo', load=load)


def download_nefertiti(load=True):
    """Download mesh of Queen Nefertiti."""
    return _download_and_read('nefertiti.ply.zip', load=load)


def download_blood_vessels(load=True):
    """Download data representing the bifurcation of blood vessels."""
    local_path, _ = _download_file('pvtu_blood_vessels/blood_vessels.zip')
    filename = os.path.join(local_path, 'T0000000500.pvtu')
    if not load:
        return filename
    mesh = pyvista.read(filename)
    mesh.set_active_vectors('velocity')
    return mesh


def download_iron_protein(load=True):
    """Download iron protein dataset."""
    return _download_and_read('ironProt.vtk', load=load)


def download_tetrahedron(load=True):
    """Download tetrahedron dataset."""
    return _download_and_read('Tetrahedron.vtu', load=load)


def download_saddle_surface(load=True):
    """Download saddle surface dataset."""
    return _download_and_read('InterpolatingOnSTL_final.stl', load=load)


def download_sparse_points(load=True):
    """Download sparse points dataset.

    Used with ``download_saddle_surface``.

    """
    saved_file, _ = _download_file('sparsePoints.txt')
    if not load:
        return saved_file
    points_reader = _vtk.vtkDelimitedTextReader()
    points_reader.SetFileName(saved_file)
    points_reader.DetectNumericColumnsOn()
    points_reader.SetFieldDelimiterCharacters('\t')
    points_reader.SetHaveHeaders(True)
    table_points = _vtk.vtkTableToPolyData()
    table_points.SetInputConnection(points_reader.GetOutputPort())
    table_points.SetXColumn('x')
    table_points.SetYColumn('y')
    table_points.SetZColumn('z')
    table_points.Update()
    return pyvista.wrap(table_points.GetOutput())


def download_foot_bones(load=True):
    """Download foot bones dataset."""
    return _download_and_read('fsu/footbones.ply', load=load)


def download_guitar(load=True):
    """Download guitar dataset."""
    return _download_and_read('fsu/stratocaster.ply', load=load)


def download_quadratic_pyramid(load=True):
    """Download quadratic pyramid dataset."""
    return _download_and_read('QuadraticPyramid.vtu', load=load)


def download_bird(load=True):
    """Download bird dataset."""
    return _download_and_read('Pileated.jpg', load=load)


def download_bird_texture(load=True):
    """Download bird texture."""
    return _download_and_read('Pileated.jpg', texture=True, load=load)


def download_office(load=True):
    """Download office dataset."""
    return _download_and_read('office.binary.vtk', load=load)


def download_horse_points(load=True):
    """Download horse points dataset."""
    return _download_and_read('horsePoints.vtp', load=load)


def download_horse(load=True):
    """Download horse dataset."""
    return _download_and_read('horse.vtp', load=load)


def download_cake_easy(load=True):
    """Download cake dataset."""
    return _download_and_read('cake_easy.jpg', load=load)


def download_cake_easy_texture(load=True):
    """Download cake texture."""
    return _download_and_read('cake_easy.jpg', texture=True, load=load)


def download_rectilinear_grid(load=True):
    """Download rectilinear grid dataset."""
    return _download_and_read('RectilinearGrid.vtr', load=load)


def download_gourds(zoom=False, load=True):
    """Download gourds dataset."""
    if zoom:
        return _download_and_read('Gourds.png', load=load)
    return _download_and_read('Gourds2.jpg', load=load)


def download_gourds_texture(zoom=False, load=True):
    """Download gourds texture."""
    if zoom:
        return _download_and_read('Gourds.png', texture=True, load=load)
    return _download_and_read('Gourds2.jpg', texture=True, load=load)


def download_unstructured_grid(load=True):
    """Download unstructured grid dataset."""
    return _download_and_read('uGridEx.vtk', load=load)


def download_letter_k(load=True):
    """Download letter k dataset."""
    return _download_and_read('k.vtk', load=load)


def download_letter_a(load=True):
    """Download letter a dataset."""
    return _download_and_read('a_grid.vtk', load=load)


def download_poly_line(load=True):
    """Download polyline dataset."""
    return _download_and_read('polyline.vtk', load=load)


def download_cad_model(load=True):
    """Download cad dataset."""
    return _download_and_read('42400-IDGH.stl', load=load)


def download_frog(load=True):
    """Download frog dataset."""
    # TODO: there are other files with this
    _download_file('froggy/frog.zraw')
    return _download_and_read('froggy/frog.mhd', load=load)


def download_prostate(load=True):
    """Download prostate dataset."""
    return _download_and_read('prostate.img', load=load)


def download_filled_contours(load=True):
    """Download filled contours dataset."""
    return _download_and_read('filledContours.vtp', load=load)


def download_doorman(load=True):
    """Download doorman dataset."""
    # TODO: download textures as well
    return _download_and_read('doorman/doorman.obj', load=load)


def download_mug(load=True):
    """Download mug dataset."""
    return _download_and_read('mug.e', load=load)


def download_oblique_cone(load=True):
    """Download oblique cone dataset."""
    return _download_and_read('ObliqueCone.vtp', load=load)


def download_emoji(load=True):
    """Download emoji dataset."""
    return _download_and_read('emote.jpg', load=load)


def download_emoji_texture(load=True):
    """Download emoji texture."""
    return _download_and_read('emote.jpg', texture=True, load=load)


def download_teapot(load=True):
    """Download teapot dataset."""
    return _download_and_read('teapot.g', load=load)


def download_brain(load=True):
    """Download brain dataset."""
    return _download_and_read('brain.vtk', load=load)


def download_structured_grid(load=True):
    """Download structured grid dataset."""
    return _download_and_read('StructuredGrid.vts', load=load)


def download_structured_grid_two(load=True):
    """Download structured grid two dataset."""
    return _download_and_read('SampleStructGrid.vtk', load=load)


def download_trumpet(load=True):
    """Download trumpet dataset."""
    return _download_and_read('trumpet.obj', load=load)


def download_face(load=True):
    """Download face dataset."""
    # TODO: there is a texture with this
    return _download_and_read('fran_cut.vtk', load=load)


def download_sky_box_nz(load=True):
    """Download skybox-nz dataset."""
    return _download_and_read('skybox-nz.jpg', load=load)


def download_sky_box_nz_texture(load=True):
    """Download skybox-nz texture."""
    return _download_and_read('skybox-nz.jpg', texture=True, load=load)


def download_disc_quads(load=True):
    """Download disc quads dataset."""
    return _download_and_read('Disc_BiQuadraticQuads_0_0.vtu', load=load)


def download_honolulu(load=True):
    """Download honolulu dataset."""
    return _download_and_read('honolulu.vtk', load=load)


def download_motor(load=True):
    """Download motor dataset."""
    return _download_and_read('motor.g', load=load)


def download_tri_quadratic_hexahedron(load=True):
    """Download tri quadratic hexahedron dataset."""
    return _download_and_read('TriQuadraticHexahedron.vtu', load=load)


def download_human(load=True):
    """Download human dataset."""
    return _download_and_read('Human.vtp', load=load)


def download_vtk(load=True):
    """Download vtk dataset."""
    return _download_and_read('vtk.vtp', load=load)


def download_spider(load=True):
    """Download spider dataset."""
    return _download_and_read('spider.ply', load=load)


def download_carotid(load=True):
    """Download carotid dataset."""
    mesh = _download_and_read('carotid.vtk', load=load)
    if not load:
        return mesh
    mesh.set_active_scalars('scalars')
    mesh.set_active_vectors('vectors')
    return mesh


def download_blow(load=True):
    """Download blow dataset."""
    return _download_and_read('blow.vtk', load=load)


def download_shark(load=True):
    """Download shark dataset."""
    return _download_and_read('shark.ply', load=load)


def download_dragon(load=True):
    """Download dragon dataset."""
    return _download_and_read('dragon.ply', load=load)


def download_armadillo(load=True):
    """Download armadillo dataset."""
    return _download_and_read('Armadillo.ply', load=load)


def download_gears(load=True):
    """Download gears dataset."""
    return _download_and_read('gears.stl', load=load)


def download_torso(load=True):
    """Download torso dataset."""
    return _download_and_read('Torso.vtp', load=load)


def download_kitchen(split=False, load=True):
    """Download structured grid of kitchen with velocity field.

    Use the ``split`` argument to extract all of the furniture in the kitchen.

    """
    mesh = _download_and_read('kitchen.vtk', load=load)
    if not load:
         return mesh
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
        alg = _vtk.vtkStructuredGridGeometryFilter()
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
    return pyvista.MultiBlock({'forward': fwd, 'inverse': inv})


def download_model_with_variance(load=True):
    """Download model with variance dataset."""
    return _download_and_read("model_with_variance.vtu", load=load)


def download_thermal_probes(load=True):
    """Download thermal probes dataset."""
    return _download_and_read("probes.vtp", load=load)


def download_carburator(load=True):
    """Download scan of a carburator."""
    return _download_and_read("carburetor.ply", load=load)


def download_turbine_blade(load=True):
    """Download scan of a turbine blade."""
    return _download_and_read('turbineblade.ply', load=load)


def download_pine_roots(load=True):
    """Download pine roots dataset."""
    return _download_and_read('pine_root.tri', load=load)


def download_crater_topo(load=True):
    """Download crater dataset."""
    return _download_and_read('Ruapehu_mag_dem_15m_NZTM.vtk', load=load)


def download_crater_imagery(load=True):
    """Download crater texture."""
    return _download_and_read('BJ34_GeoTifv1-04_crater_clip.tif', texture=True, load=load)


def download_dolfin(load=True):
    """Download dolfin mesh."""
    return _download_and_read('dolfin_fine.xml', file_format="dolfin-xml", load=load)


def download_damavand_volcano(load=True):
    """Download damavand volcano model."""
    volume = _download_and_read("damavand-volcano.vtk", load=load)
    if not load:
        return volume
    volume.rename_array("None", "data")
    return volume


def download_delaunay_example(load=True):
    """Download a pointset for the Delaunay example."""
    return _download_and_read('250.vtk', load=load)


def download_embryo(load=True):
    """Download a volume of an embryo."""
    return _download_and_read('embryo.slc', load=load)


def download_antarctica_velocity(load=True):
    """Download the antarctica velocity simulation results."""
    return _download_and_read("antarctica_velocity.vtp", load=load)


def download_room_surface_mesh(load=True):
    """Download the room surface mesh.

    This mesh is for demonstrating the difference that depth peeling can
    provide whenn rendering translucent geometries.

    This mesh is courtesy of `Sam Potter <https://github.com/sampotter>`_.
    """
    return _download_and_read("room_surface_mesh.obj", load=load)


def download_beach(load=True):
    """Download the beach NRRD image."""
    return _download_and_read("beach.nrrd", load=load)


def download_rgba_texture(load=True):
    """Download a texture with an alpha channel."""
    return _download_and_read("alphachannel.png", texture=True, load=load)


def download_vtk_logo(load=True):
    """Download a texture of the VTK logo."""
    return _download_and_read("vtk.png", texture=True, load=load)


def download_sky_box_cube_map():
    """Download a skybox cube map texture."""
    prefix = 'skybox2-'
    sets = ['posx', 'negx', 'posy', 'negy', 'posz', 'negz']
    images = [prefix + suffix + '.jpg' for suffix in sets]
    for image in images:
        _download_file(image)

    return pyvista.cubemap(pyvista.EXAMPLES_PATH, prefix)


def download_backward_facing_step(load=True):
    """Download an ensight gold case of a fluid simulation."""
    folder, _ = _download_file('EnSight.zip')
    filename = os.path.join(folder, "foam_case_0_0_0_0.case")
    if not load:
        return filename
    return pyvista.read(filename)


def download_gpr_data_array(load=True):
    """Download GPR example data array."""
    saved_file, _ = _download_file("gpr-example/data.npy")
    if not load:
        return saved_file
    return np.load(saved_file)


def download_gpr_path(load=True):
    """Download GPR example path."""
    saved_file, _ = _download_file("gpr-example/path.txt")
    if not load:
        return saved_file
    path = np.loadtxt(saved_file, skiprows=1)
    return pyvista.PolyData(path)


def download_woman(load=True):
    """Download scan of a woman.

    https://www.laserdesign.com/sample-files/full-body-scan-with-texture/

    """
    return _download_and_read('woman.stl', load=load)


def download_lobster(load=True):
    """Download scan of a lobster.

    https://www.laserdesign.com/lobster-scan-data

    """
    return _download_and_read('lobster.ply', load=load)


def download_face2(load=True):
    """Download scan of a man's face.

    https://www.laserdesign.com/sample-files/mans-face/

    """
    return _download_and_read('man_face.stl', load=load)


def download_urn(load=True):
    """Download scan of a burial urn.

    https://www.laserdesign.com/sample-files/burial-urn/

    """
    return _download_and_read('urn.stl', load=load)


def download_pepper(load=True):
    """Download scan of a pepper (capsicum).

    https://www.laserdesign.com/sample-files/hot-red-pepper/

    """
    return _download_and_read('pepper.ply', load=load)


def download_drill(load=True):
    """Download scan of a power drill.

    https://www.laserdesign.com/drill-scan-data

    """
    return _download_and_read('drill.obj', load=load)


def download_action_figure(load=True):
    """Download scan of an action figure.

    https://www.laserdesign.com/sample-files/action-figure/

    """
    return _download_and_read('tigerfighter.obj', load=load)


def download_mars_jpg():
    """Download and return the path of ``'mars.jpg'``."""
    return _download_file('mars.jpg')[0]


def download_stars_jpg():
    """Download and return the path of ``'stars.jpg'``."""
    return _download_file('stars.jpg')[0]


def download_notch_stress(load=True):
    """Download the FEA stress result from a notched beam.

    Notes
    -----
    This file may have issues being read in on VTK 8.1.2

    """
    return _download_and_read('notch_stress.vtk', load=load)


def download_notch_displacement(load=True):
    """Download the FEA displacement result from a notched beam."""
    return _download_and_read('notch_disp.vtu', load=load)


def download_louis_louvre(load=True):
    """Download the Louis XIV de France statue at the Louvre, Paris.

    Statue found in the Napoléon Courtyard of Louvre Palace. It is a
    copy in plomb of the original statue in Versailles, made by
    Bernini and Girardon.

    Credit goes to
    https://sketchfab.com/3d-models/louis-xiv-de-france-louvre-paris-a0cc0e7eee384c99838dff2857b8158c

    """
    return _download_and_read('louis.ply', load=load)


def download_cylinder_crossflow(load=True):
    """Download CFD result for cylinder in cross flow at Re=35."""
    filename, _ = _download_file('EnSight/CylinderCrossflow/cylinder_Re35.case')
    _download_file('EnSight/CylinderCrossflow/cylinder_Re35.geo')
    _download_file('EnSight/CylinderCrossflow/cylinder_Re35.scl1')
    _download_file('EnSight/CylinderCrossflow/cylinder_Re35.scl2')
    _download_file('EnSight/CylinderCrossflow/cylinder_Re35.vel')
    if not load:
        return filename
    return pyvista.read(filename)


def download_naca(load=True):
    filename, _ = _download_file('EnSight/naca.bin.case')
    _download_file('EnSight/naca.gold.bin.DENS_1')
    _download_file('EnSight/naca.gold.bin.DENS_3')
    _download_file('EnSight/naca.gold.bin.geo')
    if not load:
        return filename
    return pyvista.read(filename)


def download_wavy(load=True):
    folder, _ = _download_file('PVD/wavy.zip')
    filename = os.path.join(folder, 'wavy.pvd')
    if not load:
        return filename
    return pyvista.PVDReader(filename).read()
