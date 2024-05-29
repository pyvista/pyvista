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


@pytest.mark.skipif(os.name == 'nt', reason='path issues on Azure Windows CI')
def test_tinypages(tmpdir):
    tmp_path = Path(tmpdir)
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
        env={**os.environ, "MPLBACKEND": ""},
        encoding="utf8",
    )
    out, err = proc.communicate()

    assert proc.returncode == 0, f"sphinx build failed with stdout:\n{out}\nstderr:\n{err}\n"

    if err:
        if err.strip() != 'vtkDebugLeaks has found no leaks.':
            pytest.fail(f"sphinx build emitted the following warnings:\n{err}")

    assert html_dir.is_dir()

    def plot_file(plt, num, subnum, extension="vtksz"):
        return html_dir / f'some_plots-{plt}_{num:02d}_{subnum:02d}.{extension}'

    # verify directives generating a figure generated figures
    assert plot_file(1, 0, 0).exists()
    assert plot_file(2, 0, 0).exists()
    assert plot_file(4, 0, 0).exists()
    assert plot_file(8, 0, 0, "png").exists()
    assert plot_file(9, 0, 0, "png").exists()
    assert plot_file(9, 1, 0, "png").exists()

    # test skip directive
    assert not plot_file(10, 0, 0).exists()

    # verify external file generated figure
    cone_file = html_dir / 'plot_cone_00_00.png'
    assert cone_file.exists()

    html_contents = (html_dir / 'some_plots.html').read_bytes()
    assert b'# Only a comment' in html_contents

    # check if figure caption made it into html file
    assert b'This is the caption for plot 8.' in html_contents

    # check if figure caption using :caption: made it into html file
    assert b'Plot 10 uses the caption option.' in html_contents

    # check that the multi-image caption is applied twice
    assert html_contents.count(b'This caption applies to both plots.') == 2

    assert b'you should not be reading this right now' not in html_contents
    assert b'should be printed: include-source with no args' in html_contents

    # check that caption with tabs works
    assert html_contents.count(b'Plot __ uses the caption option with tabbed UI.') == 1
