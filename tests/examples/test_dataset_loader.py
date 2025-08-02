from __future__ import annotations

from dataclasses import dataclass
import inspect
from itertools import starmap
import os
from pathlib import Path
import shutil
from types import FunctionType
from types import ModuleType
from typing import TYPE_CHECKING
from typing import Any

import numpy as np
import pytest

import pyvista as pv
from pyvista.examples import downloads
from pyvista.examples import examples
from pyvista.examples._dataset_loader import _DatasetLoader
from pyvista.examples._dataset_loader import _Downloadable
from pyvista.examples._dataset_loader import _DownloadableFile
from pyvista.examples._dataset_loader import _format_file_size
from pyvista.examples._dataset_loader import _load_and_merge
from pyvista.examples._dataset_loader import _load_as_cubemap
from pyvista.examples._dataset_loader import _load_as_multiblock
from pyvista.examples._dataset_loader import _MultiFileDownloadableDatasetLoader
from pyvista.examples._dataset_loader import _SingleFileDatasetLoader
from pyvista.examples._dataset_loader import _SingleFileDownloadableDatasetLoader
from pyvista.examples.planets import _download_dataset_texture

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass
class DatasetLoaderTestCase:
    dataset_name: str
    dataset_function: tuple[str, FunctionType]
    dataset_loader: tuple[str, _DatasetLoader]


def _generate_dataset_loader_test_cases_from_module(
    module: ModuleType,
) -> list[DatasetLoaderTestCase]:
    # Generate test cases by module with all dataset functions and their respective file loaders.
    test_cases_dict: dict = {}

    def add_to_dict(func: str, dataset_function: Callable[[], Any]):
        # Function for stuffing example functions into a dict.
        # We use a dict to allow for any entry to be made based on example name alone.
        # This way, we can defer checking for any mismatch between the download functions
        # and file loaders to test time.
        nonlocal test_cases_dict
        if func.startswith('_dataset_'):
            dataset_name = func.split('_dataset_')[1]
            key = 'dataset_loader'
        elif func.startswith('download_'):
            dataset_name = func.split('download_')[1]
            key = 'dataset_function'
        elif func.startswith('load_'):
            dataset_name = func.split('load_')[1]
            key = 'dataset_function'
        else:
            msg = f'Invalid case specified: {(func, dataset_function)}'
            raise RuntimeError(msg)
        test_cases_dict.setdefault(dataset_name, {})
        test_cases_dict[dataset_name][key] = (func, dataset_function)

    module_members = dict(inspect.getmembers(module))

    # Collect all `download_<name> or `load_<name>` functions
    def _is_dataset_function(name, item):
        return isinstance(item, FunctionType) and name.startswith(('download_', 'load_'))

    dataset_functions = {
        name: item for name, item in module_members.items() if _is_dataset_function(name, item)
    }
    # Remove special case which is not a dataset function
    dataset_functions.pop('download_file', None)
    list(starmap(add_to_dict, dataset_functions.items()))

    # Collect all `_dataset_<name>` file loaders
    dataset_file_loaders = {
        name: item
        for name, item in module_members.items()
        if name.startswith('_dataset_') and isinstance(item, _DatasetLoader)
    }
    list(starmap(add_to_dict, dataset_file_loaders.items()))

    # Flatten dict
    test_cases_list: list[DatasetLoaderTestCase] = []
    for name, content in sorted(test_cases_dict.items()):
        dataset_function = content.setdefault('dataset_function', None)
        dataset_loader = content.setdefault('dataset_loader', None)
        test_case = DatasetLoaderTestCase(
            dataset_name=name,
            dataset_function=dataset_function,
            dataset_loader=dataset_loader,
        )
        test_cases_list.append(test_case)

    return test_cases_list


def _get_mismatch_fail_msg(test_case: DatasetLoaderTestCase):
    if test_case.dataset_function is None:
        return (
            f"A file loader:\n\t'{test_case.dataset_loader[0]}'"
            f'\n\t{test_case.dataset_loader[1]}\n'
            f'was found but is missing a corresponding download function.\n\n'
            f'Expected to find a function named:'
            f"\n\t'download_{test_case.dataset_name}'\nGot: {test_case.dataset_function}"
        )
    elif test_case.dataset_loader is None:
        return (
            f"A download function:\n\t'{test_case.dataset_function[0]}'"
            f'\n\t{test_case.dataset_function[1]}\n'
            f'was found but is missing a corresponding file loader.\n\n'
            f'Expected to find a loader named:'
            f"\n\t'_dataset_{test_case.dataset_name}'\nGot: {test_case.dataset_loader}"
        )
    else:
        return None


@pytest.fixture
def examples_local_repository_tmp_dir(tmp_path):
    """Create a local repository with a bunch of datasets available for download."""
    # setup
    repository_path = os.path.join(tmp_path, 'repo')
    os.mkdir(repository_path)

    downloadable_basenames = [
        'airplane.ply',
        'hexbeam.vtk',
        'sphere.ply',
        'uniform.vtk',
        'rectilinear.vtk',
        'globe.vtk',
        '2k_earth_daymap.jpg',
        'channels.vti',
        'pyvista_logo.png',
    ]

    # copy datasets from the pyvista repo to the local repository

    [
        shutil.copyfile(os.path.join(examples.dir_path, base), os.path.join(repository_path, base))
        for base in downloadable_basenames
    ]

    # create a zipped copy of the datasets and include the zip with repository
    shutil.make_archive(os.path.join(tmp_path, 'archive'), 'zip', repository_path)
    shutil.move(
        os.path.join(tmp_path, 'archive.zip'),
        os.path.join(repository_path, 'archive.zip'),
    )
    downloadable_basenames.append('archive.zip')

    # initialize downloads fetcher
    for base in downloadable_basenames:
        downloads.FETCHER.registry[base] = None
    downloads.FETCHER.base_url = str(repository_path) + '/'
    downloads._FILE_CACHE = True

    # make sure any "downloaded" files (moved from repo -> cache) are cleared
    cached_paths = [os.path.join(downloads.FETCHER.path, base) for base in downloadable_basenames]
    [os.remove(file) for file in cached_paths if os.path.isfile(file)]

    yield repository_path

    # teardown
    downloads.FETCHER.base_url = 'https://github.com/pyvista/vtk-data/raw/master/Data/'
    downloads._FILE_CACHE = False
    [downloads.FETCHER.registry.pop(base, None) for base in downloadable_basenames]

    # make sure any "downloaded" files (moved from repo -> cache) are cleared afterward
    [os.remove(file) for file in cached_paths if os.path.isfile(file)]


@pytest.mark.usefixtures('examples_local_repository_tmp_dir')
@pytest.mark.parametrize('use_archive', [True, False])
@pytest.mark.parametrize(
    'file_loader',
    [_SingleFileDatasetLoader, _DownloadableFile, _SingleFileDownloadableDatasetLoader],
)
def test_single_file_loader(file_loader, use_archive):
    basename = 'pyvista_logo.png'
    if use_archive and isinstance(file_loader, _Downloadable):
        file_loader = file_loader('archive.zip', target_file=basename)
        expected_path_is_absolute = False
    else:
        file_loader = file_loader(basename)
        expected_path_is_absolute = True

    # test initial path
    path = file_loader.path
    assert os.path.basename(path) == basename
    assert not os.path.isfile(path)

    if expected_path_is_absolute:
        assert os.path.isabs(path)
    else:
        assert not os.path.isabs(path)

    # test download
    if isinstance(file_loader, (_DownloadableFile, _SingleFileDownloadableDatasetLoader)):
        assert isinstance(file_loader, _Downloadable)
        path_download = file_loader.download()
        assert os.path.isfile(path_download)
        assert os.path.isabs(path_download)
        assert file_loader.path == path_download
        assert 'https://github.com/pyvista/vtk-data/raw/master/Data/' in file_loader.source_url_raw
        assert (
            'https://github.com/pyvista/vtk-data/blob/master/Data/' in file_loader.source_url_blob
        )
    else:
        with pytest.raises(AttributeError):
            file_loader.download()
        # download manually to continue test
        downloads.download_file(basename)

    # test load
    if isinstance(file_loader, (_SingleFileDatasetLoader, _SingleFileDownloadableDatasetLoader)):
        assert isinstance(file_loader, _DatasetLoader)
        dataset = file_loader.load()
        assert file_loader.dataset is None
        file_loader.load_and_store_dataset()
        assert isinstance(dataset, pv.DataSet)
    else:
        with pytest.raises(AttributeError):
            file_loader.load()

    assert os.path.isfile(file_loader.path)


def test_single_file_loader_from_directory(examples_local_repository_tmp_dir):
    # Make sure directory only contains dataset files
    Path(examples_local_repository_tmp_dir, 'archive.zip').unlink()

    # Test all datasets loaded as multiblock
    filenames = sorted(
        [Path(fname).stem for fname in Path(examples_local_repository_tmp_dir).iterdir()],
    )
    loader = _SingleFileDatasetLoader(examples_local_repository_tmp_dir)
    dataset = loader.load()
    assert isinstance(dataset, pv.MultiBlock)
    assert dataset.n_blocks == len(filenames)
    assert dataset.keys() == filenames


@pytest.mark.parametrize('load_func', [_load_as_multiblock, _load_and_merge, None])
@pytest.mark.usefixtures('examples_local_repository_tmp_dir')
def test_multi_file_loader(load_func):
    basename_loaded1 = 'airplane.ply'
    basename_loaded2 = 'hexbeam.vtk'
    basename_not_loaded = 'pyvista_logo.png'

    file_loaded1 = _SingleFileDownloadableDatasetLoader(basename_loaded1)
    file_loaded2 = _SingleFileDownloadableDatasetLoader(basename_loaded2)
    file_not_loaded = _DownloadableFile(basename_not_loaded)

    expected_airplane = examples.load_airplane()
    expected_hexbeam = examples.load_hexbeam()

    def files_func():
        return file_loaded1, file_loaded2, file_not_loaded

    multi_file_loader = _MultiFileDownloadableDatasetLoader(files_func, load_func=load_func)
    # test files func is not called when initialized
    assert multi_file_loader._file_loaders_ is None

    path = multi_file_loader.path
    assert multi_file_loader._file_loaders_ is not None
    assert isinstance(path, tuple)
    assert all(os.path.isabs(file) for file in path)
    assert len(path) == 3

    path_loadable = multi_file_loader.path_loadable
    assert isinstance(path_loadable, tuple)
    assert all(os.path.isabs(file) for file in path_loadable)
    assert len(path_loadable) == 2
    assert basename_not_loaded not in path_loadable

    # test download
    path_download = multi_file_loader.download()
    assert path_download == path
    assert all(os.path.isfile(file) for file in path_download)
    assert all(
        'https://github.com/pyvista/vtk-data/raw/master/Data/' in url
        for url in multi_file_loader.source_url_raw
    )
    assert all(
        'https://github.com/pyvista/vtk-data/blob/master/Data/' in url
        for url in multi_file_loader.source_url_blob
    )

    # test load
    # test calling load does not store the dataset internally
    assert multi_file_loader.dataset is None
    dataset_loaded = multi_file_loader.load()
    assert multi_file_loader.dataset is None

    # test load and store the dataset
    dataset_stored = multi_file_loader.load_and_store_dataset()
    # same dataset is stored, but different instance from previously loaded
    assert dataset_loaded is not dataset_stored
    if isinstance(dataset_loaded, pv.MultiBlock):
        assert np.array_equal(dataset_loaded[0].points, dataset_stored[0].points)
    else:
        assert np.array_equal(dataset_loaded.points, dataset_stored.points)
    # test calling load() again still returns yet another instance
    dataset_loaded2 = multi_file_loader.load()
    assert dataset_loaded2 is not dataset_stored
    assert dataset_loaded2 is not dataset_loaded

    if load_func is _load_as_multiblock or None:
        assert isinstance(dataset_loaded, pv.MultiBlock)
        assert dataset_loaded.keys() == ['airplane', 'hexbeam']
        assert np.array_equal(dataset_loaded[0].points, expected_airplane.points)
        assert np.array_equal(dataset_loaded[1].points, expected_hexbeam.points)
        assert len(dataset_loaded) == 2
    elif load_func is _load_and_merge:
        assert isinstance(dataset_loaded, pv.UnstructuredGrid)
        expected = pv.merge((expected_airplane, expected_hexbeam))
        assert np.array_equal(dataset_loaded.points, expected.points)


@pytest.fixture
def dataset_loader_one_file_local():
    # Test 'download' for a local built-in dataset
    loader = _SingleFileDownloadableDatasetLoader(examples.antfile)
    loader.download()
    loader.load_and_store_dataset()
    return loader


def test_dataset_loader_one_file_local(dataset_loader_one_file_local):
    loader = dataset_loader_one_file_local
    assert isinstance(loader.path, str)
    assert loader.num_files == 1
    assert loader._total_size_bytes == 17941
    assert loader.total_size == '17.9 KB'
    assert loader.unique_extension == '.ply'
    assert isinstance(loader._reader, pv.PLYReader)
    assert loader.unique_reader_type is pv.PLYReader
    assert isinstance(loader.dataset, pv.PolyData)
    assert isinstance(loader.dataset_iterable[0], pv.PolyData)
    assert loader.unique_dataset_type is pv.PolyData
    assert loader.source_name == 'ant.ply'
    assert (
        loader.source_url_raw
        == 'https://github.com/pyvista/pyvista/raw/main/pyvista/examples/ant.ply'
    )
    assert (
        loader.source_url_blob
        == 'https://github.com/pyvista/pyvista/blob/main/pyvista/examples/ant.ply'
    )
    assert loader.unique_cell_types == (pv.CellType.TRIANGLE,)


@pytest.fixture
def dataset_loader_one_file():
    loader = _SingleFileDownloadableDatasetLoader('cow.vtp')
    loader.download()
    loader.load_and_store_dataset()
    return loader


def test_dataset_loader_one_file(dataset_loader_one_file):
    loader = dataset_loader_one_file
    assert isinstance(loader.path, str)
    assert loader.num_files == 1
    assert loader._total_size_bytes == 60449
    assert loader.total_size == '60.4 KB'
    assert loader.unique_extension == '.vtp'
    assert isinstance(loader._reader, pv.XMLPolyDataReader)
    assert loader.unique_reader_type is pv.XMLPolyDataReader
    assert isinstance(loader.dataset, pv.PolyData)
    assert isinstance(loader.dataset_iterable[0], pv.PolyData)
    assert loader.unique_dataset_type is pv.PolyData
    assert loader.source_name == 'cow.vtp'
    assert loader.source_url_raw == 'https://github.com/pyvista/vtk-data/raw/master/Data/cow.vtp'
    assert loader.source_url_blob == 'https://github.com/pyvista/vtk-data/blob/master/Data/cow.vtp'
    assert loader.unique_cell_types == (
        pv.CellType.TRIANGLE,
        pv.CellType.POLYGON,
        pv.CellType.QUAD,
    )


@pytest.fixture
def dataset_loader_two_files_one_loadable():
    def _files_func():
        loadable = _SingleFileDownloadableDatasetLoader('HeadMRVolume.mhd')
        not_loadable = _DownloadableFile('HeadMRVolume.raw')
        return loadable, not_loadable

    loader = _MultiFileDownloadableDatasetLoader(_files_func)
    loader.download()
    loader.load_and_store_dataset()
    return loader


def test_dataset_loader_two_files_one_loadable(dataset_loader_two_files_one_loadable):
    loader = dataset_loader_two_files_one_loadable
    assert len(loader.path) == 2
    assert loader.num_files == 2
    assert loader._total_size_bytes == 125223
    assert loader.total_size == '125.2 KB'
    assert loader.unique_extension == ('.mhd', '.raw')
    assert pv.get_ext(loader.path[0]) == '.mhd'
    assert pv.get_ext(loader.path[1]) == '.raw'
    assert len(loader._reader) == 2
    assert isinstance(loader._reader[0], pv.MetaImageReader)
    assert loader._reader[1] is None
    assert loader.unique_reader_type is pv.MetaImageReader
    assert isinstance(loader.dataset, pv.ImageData)
    assert isinstance(loader.dataset_iterable[0], pv.ImageData)
    assert loader.unique_dataset_type is pv.ImageData
    assert loader.source_name == ('HeadMRVolume.mhd', 'HeadMRVolume.raw')
    assert loader.source_url_raw == (
        'https://github.com/pyvista/vtk-data/raw/master/Data/HeadMRVolume.mhd',
        'https://github.com/pyvista/vtk-data/raw/master/Data/HeadMRVolume.raw',
    )
    assert loader.source_url_blob == (
        'https://github.com/pyvista/vtk-data/blob/master/Data/HeadMRVolume.mhd',
        'https://github.com/pyvista/vtk-data/blob/master/Data/HeadMRVolume.raw',
    )
    assert loader.unique_cell_types == (pv.CellType.VOXEL,)


@pytest.fixture
def dataset_loader_two_files_both_loadable():
    def _files_func():
        loadable1 = _SingleFileDownloadableDatasetLoader('bolt.slc')
        loadable2 = _SingleFileDownloadableDatasetLoader('nut.slc')
        return loadable1, loadable2

    loader = _MultiFileDownloadableDatasetLoader(_files_func)
    loader.download()
    loader.load_and_store_dataset()
    return loader


def test_dataset_loader_two_files_both_loadable(dataset_loader_two_files_both_loadable):
    loader = dataset_loader_two_files_both_loadable
    assert len(loader.path) == 2
    assert loader.num_files == 2
    assert loader._total_size_bytes == 132818
    assert loader.total_size == '132.8 KB'
    assert loader.unique_extension == '.slc'
    assert pv.get_ext(loader.path[0]) == '.slc'
    assert pv.get_ext(loader.path[1]) == '.slc'
    assert len(loader._reader) == 2
    assert isinstance(loader._reader[0], pv.SLCReader)
    assert isinstance(loader._reader[1], pv.SLCReader)
    assert loader.unique_reader_type is pv.SLCReader
    assert isinstance(loader.dataset, pv.MultiBlock)
    dataset1, dataset2 = loader.dataset
    assert isinstance(dataset1, pv.ImageData)
    assert isinstance(dataset2, pv.ImageData)
    assert isinstance(loader.dataset_iterable[0], pv.MultiBlock)
    assert isinstance(loader.dataset_iterable[1], pv.ImageData)
    assert isinstance(loader.dataset_iterable[2], pv.ImageData)
    assert loader.unique_dataset_type == (pv.MultiBlock, pv.ImageData)
    assert loader.source_name == ('bolt.slc', 'nut.slc')
    assert loader.source_url_raw == (
        'https://github.com/pyvista/vtk-data/raw/master/Data/bolt.slc',
        'https://github.com/pyvista/vtk-data/raw/master/Data/nut.slc',
    )
    assert loader.source_url_blob == (
        'https://github.com/pyvista/vtk-data/blob/master/Data/bolt.slc',
        'https://github.com/pyvista/vtk-data/blob/master/Data/nut.slc',
    )
    assert loader.unique_cell_types == (pv.CellType.VOXEL,)


@pytest.fixture
def dataset_loader_cubemap():
    loader = _SingleFileDownloadableDatasetLoader(
        'cubemap_park/cubemap_park.zip',
        read_func=_load_as_cubemap,
    )
    loader.download()
    loader.load_and_store_dataset()
    return loader


def test_dataset_loader_cubemap(dataset_loader_cubemap):
    loader = dataset_loader_cubemap
    assert os.path.isdir(loader.path)
    assert loader.num_files == 6
    assert loader._total_size_bytes == 606113
    assert loader.total_size == '606.1 KB'
    assert loader.unique_extension == '.jpg'
    assert isinstance(loader.dataset, pv.Texture)
    assert isinstance(loader.dataset_iterable[0], pv.Texture)
    assert loader.unique_dataset_type is pv.Texture
    assert loader.source_name == 'cubemap_park/cubemap_park.zip'
    assert (
        loader.source_url_raw
        == 'https://github.com/pyvista/vtk-data/raw/master/Data/cubemap_park/cubemap_park.zip'
    )
    assert (
        loader.source_url_blob
        == 'https://github.com/pyvista/vtk-data/blob/master/Data/cubemap_park/cubemap_park.zip'
    )

    assert loader.unique_cell_types == (pv.CellType.PIXEL,)


@pytest.fixture
def dataset_loader_dicom():
    loader = _SingleFileDownloadableDatasetLoader('DICOM_Stack/data.zip', target_file='data')
    loader.download()
    loader.load_and_store_dataset()
    return loader


def test_dataset_loader_dicom(dataset_loader_dicom):
    loader = dataset_loader_dicom
    assert os.path.isdir(loader.path)
    assert loader.num_files == 3
    assert loader._total_size_bytes == 1583688
    assert loader.total_size == '1.6 MB'
    assert loader.unique_extension == '.dcm'
    assert isinstance(loader._reader, pv.DICOMReader)
    assert loader.unique_reader_type is pv.DICOMReader
    assert isinstance(loader.dataset, pv.ImageData)
    assert loader.unique_dataset_type is pv.ImageData
    assert loader.source_name == 'DICOM_Stack/data.zip'
    assert (
        loader.source_url_raw
        == 'https://github.com/pyvista/vtk-data/raw/master/Data/DICOM_Stack/data.zip'
    )
    assert (
        loader.source_url_blob
        == 'https://github.com/pyvista/vtk-data/blob/master/Data/DICOM_Stack/data.zip'
    )
    assert loader.unique_cell_types == (pv.CellType.VOXEL,)


def test_dataset_loader_from_nested_files_and_directory(
    dataset_loader_one_file,
    dataset_loader_two_files_one_loadable,
    dataset_loader_dicom,
):
    # test complex multiple file case with separate ext and reader, which are loaded as a tuple
    # piece together new dataset from existing ones
    def files_func():
        return (
            dataset_loader_one_file,
            dataset_loader_two_files_one_loadable,
            dataset_loader_dicom,
        )

    loader = _MultiFileDownloadableDatasetLoader(files_func, load_func=_load_as_multiblock)
    loader.download()
    assert len(loader.path) == 4
    assert loader.num_files == 6
    assert os.path.isfile(loader.path[0])
    assert os.path.isfile(loader.path[1])
    assert os.path.isfile(loader.path[2])
    assert os.path.isdir(loader.path[3])
    assert loader._filesize_bytes == (60449, 231, 124992, 1583688)
    assert loader._filesize_format == ('60.4 KB', '231 B', '125.0 KB', '1.6 MB')
    assert loader._total_size_bytes == 1769360
    assert loader.total_size == '1.8 MB'
    assert loader.unique_extension == ('.dcm', '.mhd', '.raw', '.vtp')
    assert len(loader._reader) == 4
    assert isinstance(loader._reader[0], pv.XMLPolyDataReader)
    assert isinstance(loader._reader[1], pv.MetaImageReader)
    assert loader._reader[2] is None
    assert isinstance(loader._reader[3], pv.DICOMReader)
    assert set(loader.unique_reader_type) == {
        pv.XMLPolyDataReader,
        pv.DICOMReader,
        pv.MetaImageReader,
    }
    assert loader.dataset is None
    assert loader.unique_dataset_type is type(None)
    loader.load_and_store_dataset()
    assert type(loader.dataset) is pv.MultiBlock
    assert isinstance(loader.dataset_iterable[0], pv.MultiBlock)
    assert isinstance(loader.dataset_iterable[1], pv.PolyData)
    assert isinstance(loader.dataset_iterable[2], pv.ImageData)
    assert isinstance(loader.dataset_iterable[3], pv.ImageData)
    assert set(loader.unique_dataset_type) == {pv.MultiBlock, pv.ImageData, pv.PolyData}
    assert loader.dataset.keys() == ['cow', 'HeadMRVolume', 'data']
    assert loader.source_name == (
        'cow.vtp',
        'HeadMRVolume.mhd',
        'HeadMRVolume.raw',
        'DICOM_Stack/data.zip',
    )
    assert loader.source_url_raw == (
        'https://github.com/pyvista/vtk-data/raw/master/Data/cow.vtp',
        'https://github.com/pyvista/vtk-data/raw/master/Data/HeadMRVolume.mhd',
        'https://github.com/pyvista/vtk-data/raw/master/Data/HeadMRVolume.raw',
        'https://github.com/pyvista/vtk-data/raw/master/Data/DICOM_Stack/data.zip',
    )
    assert loader.source_url_blob == (
        'https://github.com/pyvista/vtk-data/blob/master/Data/cow.vtp',
        'https://github.com/pyvista/vtk-data/blob/master/Data/HeadMRVolume.mhd',
        'https://github.com/pyvista/vtk-data/blob/master/Data/HeadMRVolume.raw',
        'https://github.com/pyvista/vtk-data/blob/master/Data/DICOM_Stack/data.zip',
    )
    assert loader.unique_cell_types == (
        pv.CellType.TRIANGLE,
        pv.CellType.POLYGON,
        pv.CellType.QUAD,
        pv.CellType.VOXEL,
    )


@pytest.fixture
def dataset_loader_nested_multiblock():
    loader = _SingleFileDownloadableDatasetLoader('mesh_fs8.exo')
    loader.download()
    loader.load_and_store_dataset()
    return loader


def test_dataset_loader_from_nested_multiblock(dataset_loader_nested_multiblock):
    loader = dataset_loader_nested_multiblock
    assert loader.num_files == 1
    assert os.path.isfile(loader.path)
    assert loader._filesize_bytes == 69732
    assert loader._filesize_format == '69.7 KB'
    assert loader._total_size_bytes == 69732
    assert loader.total_size == '69.7 KB'
    assert loader.unique_extension == '.exo'
    assert isinstance(loader._reader, pv.ExodusIIReader)
    assert loader.unique_reader_type is pv.ExodusIIReader
    assert type(loader.dataset) is pv.MultiBlock
    assert isinstance(loader.dataset_iterable[0], pv.MultiBlock)
    assert len(loader.dataset_iterable) == 12
    assert loader.unique_dataset_type == (pv.MultiBlock, pv.UnstructuredGrid)
    assert loader.source_name == 'mesh_fs8.exo'
    assert (
        loader.source_url_raw == 'https://github.com/pyvista/vtk-data/raw/master/Data/mesh_fs8.exo'
    )
    assert (
        loader.source_url_blob
        == 'https://github.com/pyvista/vtk-data/blob/master/Data/mesh_fs8.exo'
    )
    assert loader.unique_cell_types == (
        pv.CellType.TRIANGLE,
        pv.CellType.QUAD,
        pv.CellType.WEDGE,
    )


def test_load_dataset_no_reader():
    # Test using dataset with .npy file
    dataset = downloads._dataset_cloud_dark_matter

    dataset.download()
    match = '`pyvista.get_reader` does not support a file with the .npy extension'
    with pytest.raises(ValueError, match=match):
        pv.get_reader(dataset.path)
    assert dataset.unique_extension == '.npy'
    assert dataset._reader is None
    assert dataset.unique_reader_type is None

    # try loading .npy file directly
    loader = _SingleFileDatasetLoader(dataset.path)
    match = 'Error loading dataset from path'
    with pytest.raises(RuntimeError, match=match):
        loader.load()


def test_unique_cell_types_explicit_structured_grid():
    loader = examples._dataset_explicit_structured
    loader.load_and_store_dataset()
    assert loader.unique_cell_types == (pv.CellType.HEXAHEDRON,)


def test_format_file_size():
    assert _format_file_size(999) == '999 B'
    assert _format_file_size(1000) == '1.0 KB'

    assert _format_file_size(999949) == '999.9 KB'
    assert _format_file_size(999950) == '1.0 MB'
    assert _format_file_size(1000000) == '1.0 MB'

    assert _format_file_size(999949000) == '999.9 MB'
    assert _format_file_size(999950000) == '1.0 GB'
    assert _format_file_size(1000000000) == '1.0 GB'

    assert _format_file_size(999949000000) == '999.9 GB'
    assert _format_file_size(999950000000) == '1000.0 GB'
    assert _format_file_size(1000000000000) == '1000.0 GB'


def test_download_dataset_texture():
    loader = _SingleFileDownloadableDatasetLoader(
        'beach.nrrd',
    )
    loaded = _download_dataset_texture(loader, texture=True, load=True)
    assert isinstance(loaded, pv.Texture)

    loaded = _download_dataset_texture(loader, texture=False, load=True)
    assert isinstance(loaded, pv.ImageData)

    loaded = _download_dataset_texture(loader, texture=False, load=False)
    assert isinstance(loaded, str)
