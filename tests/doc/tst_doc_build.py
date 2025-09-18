"""Test the images generated from building the documentation."""

from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import NamedTuple
from xml.etree.ElementTree import parse

import pytest

ROOT_DIR = str(Path(__file__).parent.parent.parent)
BUILD_DIR = str(Path(ROOT_DIR) / 'doc' / '_build')
HTML_DIR = str(Path(BUILD_DIR) / 'html')
BUILD_IMAGE_DIR = str(Path(HTML_DIR) / '_images')

MAX_VTKSZ_FILE_SIZE_MB = 50

# Same value as `sphinx_gallery_conf['junit']` in `conf.py`
SPHINX_GALLERY_CONF_JUNIT = Path('sphinx-gallery') / 'junit-results.xml'
SPHINX_GALLERY_EXAMPLE_MAX_TIME = 150.0  # Measured in seconds
XML_FILE = HTML_DIR / SPHINX_GALLERY_CONF_JUNIT
assert XML_FILE.is_file()


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


def pytest_generate_tests(metafunc):
    """Generate parametrized tests."""
    if 'vtksz_file' in metafunc.fixturenames:
        # Generate a separate test case for each vtksz file
        files = sorted(_get_file_paths(BUILD_IMAGE_DIR, ext='vtksz'))
        ids = [str(Path(file).stem) for file in files]
        metafunc.parametrize('vtksz_file', files, ids=ids)


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
