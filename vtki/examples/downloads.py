"""Functions to download sampe datasets from the VTK data repository
"""
import shutil
import os
import sys

import vtki


DOWNLOAD_TEMP_FOLDER = True
"""
A flag to save downloads in a temporary directory or the current working
directory.
"""

# Helpers:

def _get_vtk_file_url(filename):
    return 'https://github.com/vtkiorg/vtk-data/raw/master/Data/{}'.format(filename)

def _retrieve_file(url, filename):
    # grab the correct url retriever
    if sys.version_info < (3,):
        import urllib
        urlretrieve = urllib.urlretrieve
    else:
        import urllib.request
        urlretrieve = urllib.request.urlretrieve
    if DOWNLOAD_TEMP_FOLDER:
        saved_file, resp = urlretrieve(url)
        # rename saved file:
        new_name = saved_file.replace(os.path.basename(saved_file), os.path.basename(filename))
        shutil.move(saved_file, new_name)
        return new_name, resp
    return urlretrieve(url, filename)

def _download_file(filename):
    url = _get_vtk_file_url(filename)
    return _retrieve_file(url, filename)

def _download_and_read(filename, texture=False):
    saved_file, _ = _download_file(filename)
    if texture:
        return vtki.read_texture(saved_file)
    return vtki.read(saved_file)


# Textures:

def download_masonry_texture():
    return _download_and_read('masonry.bmp', texture=True)

def download_usa_texture():
    return _download_and_read('usa_image.jpg', texture=True)


# Examples:

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
