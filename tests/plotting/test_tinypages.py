"""Tests for tinypages build using pyvista's plot_directive extension."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from subprocess import PIPE
from subprocess import Popen
import sys

import pytest

from pyvista.plotting import system_supports_plotting
from tests.conftest import flaky_test

pytest.importorskip('sphinx')

# skip all tests if unable to render
if not system_supports_plotting():
    pytestmark = pytest.mark.skip(reason='Requires system to support plotting')


ENVIRONMENT_HOOKS = ('PYVISTA_PLOT_SKIP', 'PYVISTA_PLOT_SKIP_OPTIONAL')
MATPLOTLIB_PLOT_DIRECTIVE_OUTPUT = frozenset()
PYVISTA_PLOT_DIRECTIVE_OUTPUT = frozenset(
    {
        'plot_cone.py',
        'plot_cone_00_00.png',
        'plot_polygon.py',
        'plot_polygon_00_00.png',
        'plot_polygon_00_00.vtksz',
        'some_autodocs-1.py',
        'some_autodocs-1_00_00.png',
        'some_autodocs-1_00_00.vtksz',
        'some_autodocs-2.py',
        'some_autodocs-2_00_00.png',
        'some_autodocs-2_00_00.vtksz',
        'some_autodocs-3.py',
        'some_autodocs-4.py',
        'some_plots-1.py',
        'some_plots-1_00_00.png',
        'some_plots-1_00_00.vtksz',
        'some_plots-2.py',
        'some_plots-2_00_00.png',
        'some_plots-2_00_00.vtksz',
        'some_plots-3.py',
        'some_plots-4.py',
        'some_plots-4_00_00.png',
        'some_plots-4_00_00.vtksz',
        'some_plots-5.py',
        'some_plots-6.py',
        'some_plots-7.py',
        'some_plots-8.py',
        'some_plots-8_00_00.png',
        'some_plots-9.py',
        'some_plots-9_00_00.png',
        'some_plots-9_01_00.png',
        'some_plots-10.py',
        'some_plots-11.py',
        'some_plots-12.py',
        'some_plots-13.py',
        'some_plots-13_00_00.png',
        'some_plots-13_00_00.vtksz',
        'some_plots-13_01_00.png',
        'some_plots-13_01_00.vtksz',
        'some_plots-14.py',
        'some_plots-14_00_00.png',
        'some_plots-14_01_00.png',
        'some_plots-15.py',
        'some_plots-15_00_00.png',
        'some_plots-15_00_00.vtksz',
        'some_plots-16.py',
        'some_plots-16_00_00.png',
        'some_plots-16_00_00.vtksz',
        'some_plots-17.py',
        'some_plots-18.py',
        'some_plots-18_00_00.png',
        'some_plots-18_00_00.vtksz',
        'some_plots-20.py',
        'some_plots-21.py',
        'some_plots-21_00_00.png',
        'some_plots-21_00_00.vtksz',
        'some_plots-22.py',
        'some_plots-22_00_00.png',
        'some_plots-22_00_00.vtksz',
        'some_plots-23.py',
        'some_plots-23_00_00.gif',
        'some_plots-24.py',
        'some_plots-24_00_00.png',
        'some_plots-24_00_00.vtksz',
        'some_plots-25.py',
        'some_plots-25_00_00.png',
        'some_plots-25_00_00.vtksz',
        'some_plots-26.py',
        'some_plots-26_00_01.png',
        'some_plots-26_00_01.vtksz',
    }
)
PLOTS_NEVER_SKIPPED = frozenset(
    {
        'some_plots-16_00_00.png',
        'some_plots-16_00_00.vtksz',
    }
)

PLOTS_OPTIONAL = frozenset(
    {
        'some_plots-18_00_00.png',
        'some_plots-18_00_00.vtksz',
    }
)

MATPLOTLIB_FILES = frozenset({'some_plots-19.png'})

PARALLEL_IMAGES = {
    'plot_cone_00_00.png',
    'plot_polygon_00_00.png',
    'plot_polygon_00_00.vtksz',
    'some_autodocs-e52f03c2c2ffaa1c_00_00.png',
    'some_autodocs-e52f03c2c2ffaa1c_00_00.vtksz',
    'some_plots-1.png',
    'some_plots-14c4044831c60ffd_00_00.png',
    'some_plots-14c4044831c60ffd_00_00.vtksz',
    'some_plots-174974120ecedd6c_00_00.png',
    'some_plots-174974120ecedd6c_00_00.vtksz',
    'some_plots-20cfc06bd259c838_00_00.png',
    'some_plots-20cfc06bd259c838_00_00.vtksz',
    'some_plots-3afe6bf6426295e1_00_00.png',
    'some_plots-3afe6bf6426295e1_00_00.vtksz',
    'some_plots-7ad3e6918a43db3f_00_00.png',
    'some_plots-7ad3e6918a43db3f_00_00.vtksz',
    'some_plots-7fc6e5a3627f84fa_00_00.png',
    'some_plots-7fc6e5a3627f84fa_00_00.vtksz',
    'some_plots-8dc162c98f3a4e7e_00_00.png',
    'some_plots-8dc162c98f3a4e7e_00_00.vtksz',
    'some_plots-8dc162c98f3a4e7e_01_00.png',
    'some_plots-8dc162c98f3a4e7e_01_00.vtksz',
    'some_plots-a0da58d327144b49_00_00.png',
    'some_plots-a0da58d327144b49_01_00.png',
    'some_plots-a2221791fdc1a8f6_00_00.gif',
    'some_plots-a9e7a9568ee7353a_00_00.png',
    'some_plots-a9e7a9568ee7353a_00_00.vtksz',
    'some_plots-b5a0cc1b085bd550_00_00.png',
    'some_plots-b5a0cc1b085bd550_00_00.vtksz',
    'some_plots-d9da419279a7f8bd_00_01.png',
    'some_plots-d9da419279a7f8bd_00_01.vtksz',
}

PY_FILES = frozenset(
    filename for filename in PYVISTA_PLOT_DIRECTIVE_OUTPUT if filename.endswith('.py')
)
PNG_FILES = frozenset(
    filename for filename in PYVISTA_PLOT_DIRECTIVE_OUTPUT if filename.endswith('.png')
)
VTKSZ_FILES = frozenset(
    filename for filename in PYVISTA_PLOT_DIRECTIVE_OUTPUT if filename.endswith('.vtksz')
)


@dataclass(frozen=True)
class TinyPagesCase:
    id: str
    env: dict[str, str]
    expected_files: frozenset[str]


CASES = (
    TinyPagesCase(
        id='default',
        env={},
        expected_files=PYVISTA_PLOT_DIRECTIVE_OUTPUT,
    ),
    TinyPagesCase(
        id='plot_skip_false',
        env={'PYVISTA_PLOT_SKIP': 'false'},
        expected_files=PYVISTA_PLOT_DIRECTIVE_OUTPUT,
    ),
    TinyPagesCase(
        id='plot_skip_true',
        env={'PYVISTA_PLOT_SKIP': 'true'},
        expected_files=PLOTS_NEVER_SKIPPED | PY_FILES,
    ),
    TinyPagesCase(
        id='plot_skip_optional_true',
        env={'PYVISTA_PLOT_SKIP_OPTIONAL': 'true'},
        expected_files=frozenset(PYVISTA_PLOT_DIRECTIVE_OUTPUT - PLOTS_OPTIONAL),
    ),
)


@flaky_test(exceptions=(AssertionError,))
@pytest.mark.parametrize('case', CASES, ids=lambda case: case.id)
def test_tinypages(tmp_path: Path, case: TinyPagesCase):
    """Test tinypages build using pyvista's plot_directive extension."""
    for hook in ENVIRONMENT_HOOKS:
        os.environ.pop(hook, None)
    os.environ.update(case.env)

    skip = case.env.get('PYVISTA_PLOT_SKIP', 'false').lower() == 'true'
    skip_optional = case.env.get('PYVISTA_PLOT_SKIP_OPTIONAL', 'false').lower() == 'true'
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

    assert html_dir.is_dir()

    # All files are saved to the pyvista plot dir
    pyvista_plot_directive_dir = html_dir.parent / 'pyvista_plot_directive'
    pyvista_plot_directive_files = {path.name for path in pyvista_plot_directive_dir.iterdir()}
    assert pyvista_plot_directive_files == case.expected_files

    # Ensure all generated Python files can be compiled
    non_compilable_files = set()
    for file in case.expected_files:
        path = pyvista_plot_directive_dir / file
        if path.suffix == '.py':
            try:
                compile(path.read_text(encoding='utf-8'), str(path), 'exec')
            except SyntaxError:
                non_compilable_files.add(file)
    non_compilable = '\n'.join(sorted(non_compilable_files))
    assert not non_compilable_files, f'Non-compilable Python files: \n{non_compilable}'

    # Ensure no copies are saved to the html dir itself
    unexpected_files = [
        path for ext in ('py', 'rst', 'png', 'vtksz') for path in html_dir.glob(f'*.{ext}')
    ]
    assert not unexpected_files, f'Unexpected files in html dir: {unexpected_files}'

    # Sphinx auto-copies to `_images`. Expect matplotlib's directive output to exist as well
    expected_html_images = (case.expected_files - PY_FILES) | MATPLOTLIB_FILES
    actual_html_images = {p.name for p in (html_dir / '_images').rglob('*') if p.is_file()}
    assert actual_html_images == expected_html_images

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

    # check that enforced skip caption doesn't exist
    assert b'This plot will always be skipped with no caption' not in html_contents

    # check conditional execution
    assert (b'This plot may be skipped with no caption' in html_contents) == expected_optional

    # check matplotlib plot exists
    assert b'This is a matplotlib plot.' in html_contents


@flaky_test(exceptions=(AssertionError,))
@pytest.mark.skip_windows('path issues, e.g. image file not readable')
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
    assert proc.returncode == 0, f'sphinx build failed with stdout:\n{out}\nstderr:\n{err}\n'

    actual_images = {p.name for p in (html_dir / '_images').rglob('*') if p.is_file()}
    assert actual_images == PARALLEL_IMAGES


@pytest.mark.needs_playwright
def test_interactive_plot_moves(tmp_path: Path):
    from http.server import SimpleHTTPRequestHandler
    from http.server import ThreadingHTTPServer
    import subprocess
    from threading import Thread

    from playwright.sync_api import sync_playwright

    source_dir = Path(__file__).parent / 'tinypages'
    html_dir = tmp_path / '_build'

    result = subprocess.run(
        [
            'sphinx-build',
            '-b',
            'html',
            str(source_dir),
            str(html_dir),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr

    old_cwd = Path.cwd()
    os.chdir(html_dir)

    server = ThreadingHTTPServer(('127.0.0.1', 0), SimpleHTTPRequestHandler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        with sync_playwright() as p:
            # Open docs in browser
            browser = p.chromium.launch()
            page = browser.new_page()

            host, port = server.server_address
            page.goto(f'http://{host}:{port}/some_plots.html')
            page.wait_for_timeout(1000)

            # Navigate to interactive scene tab
            page.get_by_text('Interactive Scene', exact=True).first.click()
            page.wait_for_timeout(1000)

            frame = page.frame_locator('iframe').first
            canvas = frame.locator('canvas')

            canvas.wait_for(timeout=10000)

            # Simulate interacting with the scene, taking a screenshot before and after
            before = canvas.screenshot()

            box = canvas.bounding_box()
            assert box is not None

            x = box['x'] + box['width'] / 2
            y = box['y'] + box['height'] / 2

            page.mouse.move(x, y)
            page.mouse.down()
            page.mouse.move(x + 200, y + 100)
            page.mouse.up()

            page.wait_for_timeout(500)

            after = canvas.screenshot()

            # Interaction is successful if screenshot differs
            assert before != after

    finally:
        server.shutdown()
        server.server_close()
        os.chdir(old_cwd)
