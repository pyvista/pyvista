"""Test the images generated from building the documentation."""

from __future__ import annotations

import glob
import os
from pathlib import Path
import shutil
from typing import Literal
from typing import NamedTuple
import warnings
from xml.etree.ElementTree import parse

from PIL import Image
import pytest

import pyvista as pv

ROOT_DIR = str(Path(__file__).parent.parent.parent)
BUILD_DIR = str(Path(ROOT_DIR) / 'doc' / '_build')
HTML_DIR = str(Path(BUILD_DIR) / 'html')
BUILD_IMAGE_DIR = str(Path(HTML_DIR) / '_images')
DEBUG_IMAGE_DIR = str(Path(ROOT_DIR) / '_doc_debug_images')
DEBUG_IMAGE_FAILED_DIR = str(Path(ROOT_DIR) / '_doc_debug_images_failed')
BUILD_IMAGE_CACHE = str(Path(__file__).parent / 'doc_image_cache')
FLAKY_IMAGE_DIR = str(Path(__file__).parent / 'flaky_tests')
FLAKY_TEST_CASES = [path.name for path in Path(FLAKY_IMAGE_DIR).iterdir() if path.is_dir()]

MAX_VTKSZ_FILE_SIZE_MB = 50

# Same value as `sphinx_gallery_conf['junit']` in `conf.py`
SPHINX_GALLERY_CONF_JUNIT = Path('sphinx-gallery') / 'junit-results.xml'
SPHINX_GALLERY_EXAMPLE_MAX_TIME = 150.0  # Measured in seconds
XML_FILE = Path(HTML_DIR) / SPHINX_GALLERY_CONF_JUNIT
assert XML_FILE.is_file()

pytestmark = [pytest.mark.filterwarnings(r'always:.*\n.*THIS IS A FLAKY TEST.*:UserWarning')]


class _TestCaseTuple(NamedTuple):
    test_name: str
    docs_image_path: str
    cached_image_path: str


def _get_file_paths(dir_: str, ext: str):
    """Get all paths of files with a specific extension inside a directory tree."""
    pattern = str(Path(dir_) / '**' / ('*.' + ext))
    return glob.glob(pattern, recursive=True)  # noqa: PTH207


def _flatten_path(path: str):
    return '_'.join(os.path.split(path))[1:]


def _preprocess_build_images(build_images_dir: str, output_dir: str):
    """Read images from the build dir, resize them, and save as JPG to a flat output dir.

    All PNG and GIF files from the build are included, and are saved as JPG.

    """
    input_png = _get_file_paths(build_images_dir, ext='png')
    input_gif = _get_file_paths(build_images_dir, ext='gif')
    input_jpg = _get_file_paths(build_images_dir, ext='jpg')
    output_paths = []
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for input_path in input_png + input_gif + input_jpg:
        # input image from the docs may come from a nested directory,
        # so we flatten the file's relative path
        output_file_name = _flatten_path(os.path.relpath(input_path, build_images_dir))
        output_file_name = Path(output_file_name).with_suffix('.jpg')
        output_path = str(Path(output_dir) / output_file_name)
        output_paths.append(output_path)

        # Ensure image size is max 400x400 and save to output
        with Image.open(input_path) as im:
            im = im.convert('RGB') if im.mode != 'RGB' else im  # noqa: PLW2901
            if not (im.size[0] <= 400 and im.size[1] <= 400):
                im.thumbnail(size=(400, 400))
            im.save(output_path, quality='keep') if im.format == 'JPEG' else im.save(output_path)

    return output_paths


def _generate_test_cases():
    """Generate a list of image test cases.
    This function:
        (1) Generates a list of test images from the docs
        (2) Generates a list of cached images
        (3) Merges the two lists together and returns separate test cases to
            comparing all docs images to all cached images
    """
    test_cases_dict: dict = {}

    def add_to_dict(filepath: str, key: str):
        # Function for stuffing image paths into a dict.
        # We use a dict to allow for any entry to be made based on image path alone.
        # This way, we can defer checking for any mismatch between the cached and docs
        # images to test time.
        nonlocal test_cases_dict
        test_name = Path(filepath).stem
        try:
            test_cases_dict[test_name]
        except KeyError:
            test_cases_dict[test_name] = {}
        test_cases_dict[test_name].setdefault(key, filepath)

    # process test images
    test_image_paths = _preprocess_build_images(BUILD_IMAGE_DIR, DEBUG_IMAGE_DIR)
    [add_to_dict(path, 'docs') for path in test_image_paths]

    # process cached images
    cached_image_paths = _get_file_paths(BUILD_IMAGE_CACHE, ext='jpg')
    [add_to_dict(path, 'cached') for path in cached_image_paths]

    # flatten dict
    test_cases_list = []
    for test_name, content in sorted(test_cases_dict.items()):
        doc = content.get('docs', None)
        cache = content.get('cached', None)
        test_case = _TestCaseTuple(
            test_name=test_name,
            docs_image_path=doc,
            cached_image_path=cache,
        )
        test_cases_list.append(test_case)

    return test_cases_list


def pytest_generate_tests(metafunc):
    """Generate parametrized tests."""
    if 'test_case' in metafunc.fixturenames:
        # Generate a separate test case for each image being tested
        test_cases = _generate_test_cases()
        ids = [case.test_name for case in test_cases]
        metafunc.parametrize('test_case', test_cases, ids=ids)

    if 'vtksz_file' in metafunc.fixturenames:
        # Generate a separate test case for each vtksz file
        files = sorted(_get_file_paths(BUILD_IMAGE_DIR, ext='vtksz'))
        ids = [str(Path(file).stem) for file in files]
        metafunc.parametrize('vtksz_file', files, ids=ids)


def _save_failed_test_image(source_path, category: Literal['warnings', 'errors', 'flaky']):
    """Save test image from cache or build to the failed image dir."""
    parent_dir = Path(category)
    if Path(source_path).parent == Path(BUILD_IMAGE_CACHE):
        dest_dirname = 'from_cache'
    else:
        dest_dirname = 'from_build'
    Path(DEBUG_IMAGE_FAILED_DIR).mkdir(exist_ok=True)
    Path(DEBUG_IMAGE_FAILED_DIR, parent_dir).mkdir(exist_ok=True)
    dest_dir = Path(DEBUG_IMAGE_FAILED_DIR, parent_dir, dest_dirname)
    dest_dir.mkdir(exist_ok=True)
    dest_path = Path(dest_dir, Path(source_path).name)
    shutil.copy(source_path, dest_path)


def test_static_images(test_case: _TestCaseTuple):
    fail_msg, fail_source = _test_both_images_exist(*test_case)
    if fail_msg:
        _save_failed_test_image(fail_source, 'errors')
        pytest.fail(fail_msg)

    warn_msg, fail_msg = _test_compare_images(*test_case)
    if fail_msg:
        _save_failed_test_image(test_case.docs_image_path, 'errors')
        _save_failed_test_image(test_case.cached_image_path, 'errors')
        pytest.fail(fail_msg)

    if warn_msg:
        parent_dir = (
            'flaky' if Path(test_case.cached_image_path).stem in FLAKY_TEST_CASES else 'warnings'
        )
        _save_failed_test_image(test_case.docs_image_path, parent_dir)
        _save_failed_test_image(test_case.cached_image_path, parent_dir)
        warnings.warn(warn_msg)


def _test_both_images_exist(filename, docs_image_path, cached_image_path):
    if docs_image_path is None or cached_image_path is None:
        if docs_image_path is None:
            assert cached_image_path is not None
            source_path = cached_image_path
            exists = 'cache'
            missing = 'docs build'
            exists_path = cached_image_path
            missing_path = BUILD_IMAGE_DIR
        else:
            assert docs_image_path is not None
            source_path = docs_image_path
            exists = 'docs build'
            missing = 'cache'
            exists_path = BUILD_IMAGE_DIR
            missing_path = BUILD_IMAGE_CACHE

        msg = (
            f'Test setup failed for test image:\n'
            f'\t{filename}\n'
            f'The image exists in the {exists} directory:\n'
            f'\t{exists_path}\n'
            f'but is missing from the {missing} directory:\n'
            f'\t{missing_path}\n'
        )
        return msg, source_path
    return None, None


def _test_compare_images(test_name, docs_image_path, cached_image_path):
    try:
        docs_image = pv.read(docs_image_path)
        cached_image = pv.read(cached_image_path)

        # Check if test should fail or warn
        error = pv.compare_images(docs_image, cached_image)
        fail_msg = _check_compare_fail(test_name, error)
        warn_msg = _check_compare_warn(test_name, error)
        if fail_msg:
            # Check if test case is flaky test
            if test_name in FLAKY_TEST_CASES:
                # Compare build image to other known valid versions
                success_path = _is_false_positive(test_name, docs_image)
                if success_path:
                    # Convert failure into a warning
                    warn_msg = fail_msg + (
                        '\nTHIS IS A FLAKY TEST. It initially failed (as above) but passed when '
                        f'compared to:\n\t{success_path}'
                    )
                    fail_msg = None
                else:
                    # Test still fails
                    fail_msg += (
                        '\nTHIS IS A FLAKY TEST. It initially failed (as above) and failed again '
                        f'for all images in \n\t{Path(FLAKY_IMAGE_DIR, test_name)!s}.'
                    )
    except RuntimeError as e:
        warn_msg = None
        fail_msg = repr(e)
    return warn_msg, fail_msg


def _check_compare_fail(filename, error_, allowed_error=500.0):
    if error_ > allowed_error:
        return (
            f'{filename} Exceeded image regression error of '
            f'{allowed_error} with an image error equal to: {error_}'
        )
    return None


def _check_compare_warn(filename, error_, allowed_warning=200.0):
    if error_ > allowed_warning:
        return (
            f'{filename} Exceeded image regression warning of '
            f'{allowed_warning} with an image error of '
            f'{error_}'
        )
    return None


def _is_false_positive(test_name, docs_image):
    """Compare against other image in the flaky image dir."""
    paths = _get_file_paths(str(Path(FLAKY_IMAGE_DIR, test_name)), 'jpg')
    for path in paths:
        error = pv.compare_images(docs_image, pv.read(path))
        if _check_compare_fail(test_name, error) is None:
            return path
    return None


def test_interactive_plot_file_size(vtksz_file: str):
    filepath = Path(vtksz_file)
    assert filepath.is_file()
    size_bytes = filepath.stat().st_size
    size_megabytes = round(size_bytes / 1_000_000)
    if size_megabytes > MAX_VTKSZ_FILE_SIZE_MB:
        rel_path = filepath.relative_to(ROOT_DIR)
        msg = (
            f'The generated interactive plot file is too large: '
            f'\n\t{rel_path}\n'
            f'Its size is {size_megabytes} MB, but must be less than {MAX_VTKSZ_FILE_SIZE_MB} MB.'
            f'\nConsider reducing the complexity of the plot or forcing it to be static.'
        )
        pytest.fail(msg)


xml_root = parse(XML_FILE).getroot()
test_cases = [dict(case.attrib) for case in xml_root.iterfind('testcase')]
test_ids = [case['classname'] for case in test_cases]


@pytest.mark.parametrize('testcase', test_cases, ids=test_ids)
def test_sphinx_gallery_execution_times(testcase):
    if float(testcase['time']) > SPHINX_GALLERY_EXAMPLE_MAX_TIME:
        pytest.fail(
            f'Gallery example {testcase["name"]!r} from {testcase["file"]!r}\n'
            f'Took too long to run: '
            f'Duration {testcase["time"]}s > {SPHINX_GALLERY_EXAMPLE_MAX_TIME}s',
        )
