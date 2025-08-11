"""Tests for tinypages build using pyvista's plot_directive extension."""

from __future__ import annotations

import os
from pathlib import Path
from subprocess import PIPE
from subprocess import Popen
import sys

import pytest

from pyvista.ext.plot_directive import hash_plot_code
from pyvista.plotting import system_supports_plotting
from tests.conftest import flaky_test

pytest.importorskip('sphinx')

# skip all tests if unable to render
if not system_supports_plotting():
    pytestmark = pytest.mark.skip(reason='Requires system to support plotting')

ENVIRONMENT_HOOKS = ['PYVISTA_PLOT_SKIP', 'PYVISTA_PLOT_SKIP_OPTIONAL']


@flaky_test(exceptions=(AssertionError,))
@pytest.mark.skip_windows('path issues on Azure Windows CI')
@pytest.mark.parametrize('ename', ENVIRONMENT_HOOKS)
@pytest.mark.parametrize('evalue', [False, True])
def test_tinypages(tmp_path, ename, evalue):
    # sanitise the environment namespace
    for hook in ENVIRONMENT_HOOKS:
        os.environ.pop(hook, None)

    # configure the plot-directive environment variable hook for conf.py
    os.environ[ename] = str(evalue)

    skip = False if ename != 'PLOT_SKIP' else evalue
    skip_optional = False if ename != 'PLOT_SKIP_OPTIONAL' else evalue
    expected = not skip
    expected_optional = False if skip else not skip_optional

    html_dir = tmp_path / 'html'
    doctree_dir = tmp_path / 'doctrees'
    # Build the pages with warnings turned into errors
    cmd = [
        sys.executable,
        '-msphinx',
        '-W',
        '-b',
        'html',
        '-d',
        str(doctree_dir),
        str(Path(__file__).parent / 'tinypages'),
        str(html_dir),
    ]
    proc = Popen(
        cmd,
        stdout=PIPE,
        stderr=PIPE,
        universal_newlines=True,
        env={**os.environ, 'MPLBACKEND': ''},
        encoding='utf8',
    )
    out, err = proc.communicate()

    assert proc.returncode == 0, f'sphinx build failed with stdout:\n{out}\nstderr:\n{err}\n'

    if err:
        if err.strip() != 'vtkDebugLeaks has found no leaks.':
            pytest.fail(f'sphinx build emitted the following warnings:\n{err}')

    assert html_dir.is_dir()

    def plot_file(plt, num, subnum, extension='vtksz'):
        return html_dir / f'some_plots-{plt}_{num:02d}_{subnum:02d}.{extension}'

    # verify directives generating a figure generated figures
    assert plot_file(1, 0, 0).exists() == expected
    assert plot_file(2, 0, 0).exists() == expected
    assert plot_file(4, 0, 0).exists() == expected
    assert plot_file(21, 0, 0).exists() == expected
    assert plot_file(22, 0, 0).exists() == expected
    assert plot_file(24, 0, 0).exists() == expected
    assert plot_file(25, 0, 0).exists() == expected
    assert plot_file(8, 0, 0, 'png').exists() == expected
    assert plot_file(9, 0, 0, 'png').exists() == expected
    assert plot_file(9, 1, 0, 'png').exists() == expected
    assert plot_file(23, 0, 0, 'gif').exists() == expected

    # verify a figure is *not* generated when show isn't called
    assert not plot_file(20, 0, 0).exists()

    # test skip directive
    assert not plot_file(10, 0, 0).exists()

    # verify external file generated figure
    cone_file = html_dir / 'plot_cone_00_00.png'
    assert cone_file.exists() == expected

    html_contents = (html_dir / 'some_plots.html').read_bytes()
    assert b'# Only a comment' in html_contents

    # check if figure caption made it into html file
    assert (b'This is the caption for plot 8.' in html_contents) == expected

    # check if figure caption using :caption: made it into html file
    assert (b'Plot 8 uses the caption option.' in html_contents) == expected

    # check that the multi-image caption is applied twice
    assert (html_contents.count(b'This caption applies to both plots.') == 2) == expected

    assert b'you should not be reading this right now' not in html_contents
    assert b'should be printed: include-source with no args' in html_contents

    # check that caption with tabs works
    assert (
        html_contents.count(b'Plot 15 uses the caption option with tabbed UI.') == 1
    ) == expected

    # check that no skip always exists
    assert b'Plot 16 will never be skipped' in html_contents
    assert plot_file(16, 0, 0).exists()

    # check that enforced skip caption doesn't exist
    assert b'This plot will always be skipped with no caption' not in html_contents

    # check conditional execution
    assert (b'This plot may be skipped with no caption' in html_contents) == expected_optional
    assert plot_file(18, 0, 0).exists() == expected_optional

    # check matplotlib plot exists
    # we're using the same counter, but mpl doesn't add num and subnum
    plt_num = 19
    mpl_figure_file = html_dir / f'some_plots-{plt_num}.png'
    assert mpl_figure_file.exists
    assert b'This is a matplotlib plot.' in html_contents


def test_parallel(tmp_path: Path) -> None:
    """Ensure that labeling image serial fails."""
    html_dir = tmp_path / 'html'
    doctree_dir = tmp_path / 'doctrees'
    cmd = [
        sys.executable,
        '-msphinx',
        '-W',
        '-b',
        'html',
        '-j2',
        '-d',
        str(doctree_dir),
        str(Path(__file__).parent / 'tinypages'),
        str(html_dir),
    ]
    proc = Popen(
        cmd,
        stdout=PIPE,
        stderr=PIPE,
        universal_newlines=True,
        env={**os.environ, 'MPLBACKEND': ''},
        encoding='utf8',
    )
    out, err = proc.communicate()

    assert 'pyvista_plot_use_counter' in err
    assert 'cannot be enabled for parallel builds' in err

    proc = Popen(
        cmd,
        stdout=PIPE,
        stderr=PIPE,
        universal_newlines=True,
        env={**os.environ, 'MPLBACKEND': '', 'PYVISTA_PLOT_USE_COUNTER': 'false'},
        encoding='utf8',
    )
    out, err = proc.communicate()
    assert not err

    assert len(list(html_dir.glob('**/*.png'))) == 27


def test_hash_plot_code_consistency():
    code = 'import matplotlib.pyplot as plt\nplt.plot([1, 2, 3])'
    options = {}

    hash1 = hash_plot_code(code, options)
    hash2 = hash_plot_code(code, options)
    assert hash1 == hash2
    assert len(hash1) == 16

    different_code = 'plt.plot([4, 5, 6])'
    hash3 = hash_plot_code(different_code, options)
    assert hash1 != hash3


def test_hash_plot_code_normalization():
    code_with_noise = (
        'import matplotlib.pyplot as plt  # plotting lib\n\nplt.plot([1, 2, 3])  # make plot\n\n'
    )
    code_clean = 'import matplotlib.pyplot as plt\nplt.plot([1, 2, 3])'
    doctest_code = '>>> import matplotlib.pyplot as plt\n>>> plt.plot([1, 2, 3])'
    options = {}

    hash1 = hash_plot_code(code_with_noise, options)
    hash2 = hash_plot_code(code_clean, options)
    hash3 = hash_plot_code(doctest_code, options)
    assert hash1 == hash2 == hash3


def test_hash_plot_code_context_option():
    code = 'plt.plot([1, 2, 3])'

    hash_no_context = hash_plot_code(code, {})
    hash_with_context = hash_plot_code(code, {'context': True})
    hash_other_option = hash_plot_code(code, {'other': True})

    assert hash_no_context != hash_with_context
    assert hash_no_context == hash_other_option
