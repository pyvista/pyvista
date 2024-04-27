"""Test the images generated from building the documentation."""

import glob
import os
from pathlib import Path
from typing import Dict, NamedTuple
import warnings

from PIL import Image
import pytest

import pyvista as pv

ROOT_DIR = str(Path(__file__).parent.parent.parent)
BUILD_DIR = str(Path(ROOT_DIR) / 'doc' / '_build')
BUILD_IMAGE_DIR = str(Path(BUILD_DIR) / 'html' / '_images')
DEBUG_IMAGE_DIR = str(Path(ROOT_DIR) / '_doc_debug_images')
BUILD_IMAGE_CACHE = str(Path(__file__).parent / 'doc_image_cache')


class _TestCaseTuple(NamedTuple):
    filename: str
    docs_image_path: str
    cached_image_path: str


def _get_file_paths(dir_: str, ext: str):
    """Get all paths of files with a specific extension inside a directory tree."""
    pattern = str(Path(dir_) / '**' / ('*.' + ext))
    file_paths = glob.glob(pattern, recursive=True)  # noqa: PTH207
    return file_paths


def _flatten_path(path: str):
    return '_'.join(os.path.split(path))[1:]


def _preprocess_build_images(build_images_dir: str, output_dir: str):
    """Read png images from the build dir, resize them, and save to flat output dir."""
    input_paths = _get_file_paths(build_images_dir, ext='png')
    output_paths = []
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for input_path in input_paths:
        # input image from the docs may come from a nested directory,
        # so we flatten the file's relative path
        output_file_name = _flatten_path(os.path.relpath(input_path, build_images_dir))
        output_path = str(Path(output_dir) / output_file_name)
        output_paths.append(output_path)

        # Ensure image size is max 400x400 and save to output
        im = Image.open(input_path)
        im.thumbnail(size=(400, 400))
        im.save(output_path)

    return output_paths


def _generate_test_cases():
    """Generate a list of image test cases.
    This function:
        (1) Generates a list of test images from the docs
        (2) Generates a list of cached images
        (3) Merges the two lists together and returns separate test cases to
            comparing all docs images to all cached images
    """
    test_cases_dict: Dict = {}

    def add_to_dict(filepath: str, key: str):
        # Function for stuffing image paths into a dict.
        # We use a dict to allow for any entry to be made based on image path alone.
        # This way, we can defer checking for any mismatch between the cached and docs
        # images to test time.
        nonlocal test_cases_dict
        filename = Path(filepath).name
        try:
            test_cases_dict[filename]
        except KeyError:
            test_cases_dict[filename] = {}
        test_cases_dict[filename].setdefault(key, filepath)

    # process test images
    test_image_paths = _preprocess_build_images(BUILD_IMAGE_DIR, DEBUG_IMAGE_DIR)
    [add_to_dict(path, 'docs') for path in test_image_paths]

    # process cached images
    cached_image_paths = _get_file_paths(BUILD_IMAGE_CACHE, ext='png')
    [add_to_dict(path, 'cached') for path in cached_image_paths]

    # flatten dict
    test_cases_list = []
    for filename, content in sorted(test_cases_dict.items()):
        doc = content.get('docs', None)
        cache = content.get('cached', None)
        test_case = _TestCaseTuple(filename=filename, docs_image_path=doc, cached_image_path=cache)
        test_cases_list.append(test_case)

    return test_cases_list


def pytest_generate_tests(metafunc):
    """Generate parametrized tests."""
    if 'test_case' in metafunc.fixturenames:
        # Generate a separate test case for each image being tested
        test_cases = _generate_test_cases()
        ids = [case.filename for case in test_cases]
        metafunc.parametrize('test_case', test_cases, ids=ids)


def test_docs(test_case):
    filename, docs_image_path, cached_image_path = test_case
    if docs_image_path is None or cached_image_path is None:
        if docs_image_path is None:
            exists = 'cache'
            missing = 'docs'
        else:
            exists = 'docs'
            missing = 'cache'
        pytest.fail(
            f"Test setup failed for test case:\n"
            f"\t{filename}\n"
            f"The image exists in the {exists}, but is missing from the {missing}.",
        )

    docs_image = pv.read(docs_image_path)
    cached_image = pv.read(cached_image_path)
    error = pv.compare_images(docs_image, cached_image)

    allowed_error = 500.0
    allowed_warning = 200.0
    if error > allowed_error:
        pytest.fail(
            f"{filename} Exceeded image regression error of "
            f"{allowed_error} with an image error equal to: {error}",
        )
    if error > allowed_warning:
        warnings.warn(
            f"{filename} Exceeded image regression warning of "
            f"{allowed_warning} with an image error of "
            f"{error}",
        )
