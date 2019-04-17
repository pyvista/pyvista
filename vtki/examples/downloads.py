"""Functions to download sampe datasets from the VTK data repository
"""
import shutil
import os
import sys
import zipfile

import vtki

# Helpers:

def delete_downloads():
    """Delete all downloaded examples to free space or update the files"""
    shutil.rmtree(vtki.EXAMPLES_PATH)
    os.makedirs(vtki.EXAMPLES_PATH)
    return True


def _decompress(filename):
    zip_ref = zipfile.ZipFile(filename, 'r')
    zip_ref.extractall(vtki.EXAMPLES_PATH)
    return zip_ref.close()

def _get_vtk_file_url(filename):
    return 'https://github.com/vtkiorg/vtk-data/raw/master/Data/{}'.format(filename)

def _retrieve_file(url, filename):
    # First check if file has already been downloaded
    local_path = os.path.join(vtki.EXAMPLES_PATH, os.path.basename(filename))
    if os.path.isfile(local_path.replace('.zip', '')):
        return local_path.replace('.zip', ''), None
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
    return local_path, resp

def _download_file(filename):
    url = _get_vtk_file_url(filename)
    return _retrieve_file(url, filename)

def _download_and_read(filename, texture=False):
    saved_file, _ = _download_file(filename)
    if vtki.get_ext(saved_file) in ['.zip']:
        _decompress(saved_file)
        saved_file = saved_file[:-4]
    if texture:
        return vtki.read_texture(saved_file)
    return vtki.read(saved_file)


# Textures:

def download_masonry_texture():
    return _download_and_read('masonry.bmp', texture=True)

def download_usa_texture():
    return _download_and_read('usa_image.jpg', texture=True)

def download_puppy_texture():
    return _download_and_read('puppy.jpg', texture=True)


# Examples:

def download_puppy():
    return _download_and_read('puppy.jpg')

def download_usa():
    return _download_and_read('usa.vtk')

def download_st_helens():
    return _download_and_read('SainteHelens.dem')

def download_bunny():
    return _download_and_read('bunny.ply')

def download_cow():
    return _download_and_read('cow.vtp')

def download_faults():
    return _download_and_read('faults.vtk')

def download_tensors():
    return _download_and_read('tensors.vtk')

def download_head():
    _download_file('HeadMRVolume.raw')
    return _download_and_read('HeadMRVolume.mhd')

def download_bolt_nut():
    blocks = vtki.MultiBlock()
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

def download_lidar():
    return _download_and_read('kafadar-lidar-interp.vtp')

def download_exodus():
    """Sample ExodusII data file"""
    return _download_and_read('mesh_fs8.exo')


def download_nefertiti():
    """ Download mesh of Queen Nefertiti """
    return _download_and_read('Nefertiti.obj.zip')
