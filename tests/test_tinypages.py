"""Tests for tinypages build using sphinx extensions."""

from __future__ import annotations

import os
from pathlib import Path
from subprocess import PIPE
from subprocess import Popen
import sys

import pytest

from pyvista.plotting import system_supports_plotting

pytest.importorskip('sphinx')

# skip all tests if unable to render
if not system_supports_plotting():
    pytestmark = pytest.mark.skip(reason='Requires system to support plotting')

ENVIRONMENT_HOOKS = ['PLOT_SKIP', 'PLOT_SKIP_OPTIONAL']


@pytest.mark.skipif(os.name == 'nt', reason='path issues on Azure Windows CI')
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

    tmp_dir = tmp_path / f'{ename}_{evalue}'
    tmp_dir.mkdir()
    html_dir = tmp_dir / 'html'
    doctree_dir = tmp_dir / 'doctrees'
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
    assert plot_file(8, 0, 0, 'png').exists() == expected
    assert plot_file(9, 0, 0, 'png').exists() == expected
    assert plot_file(9, 1, 0, 'png').exists() == expected

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
