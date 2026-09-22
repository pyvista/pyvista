from __future__ import annotations

from pathlib import Path
import shlex
import subprocess
import sys
from typing import TYPE_CHECKING

from pytest_cases import parametrize

import pyvista as pv
from pyvista.__main__ import main

if TYPE_CHECKING:
    import pytest

TOKENS_KWARGS = [
    ('--color=red', dict(color='red')),
    ('--color=red --opacity=0.1', dict(color='red', opacity=0.1)),
    ('--color=blue --culling="front"', dict(color='blue', culling='front')),
    ('--background="blue" --color red', dict(background='blue', color='red')),
]


def _plot_argv(outfile: str) -> str:
    """Return the `plot` invocation writing a screenshot of the ant example."""
    infile = Path(pv.examples.antfile).as_posix()
    return f'plot {infile} --off-screen --screenshot={outfile}'


def _plotter_difference(outfile: str, kwargs: dict) -> float:
    """Return the image difference against the same scene built through the API."""
    pl = pv.Plotter()
    if (b := 'background') in kwargs:
        pl.set_background(kwargs[b])
        kwargs = {
            k: v for k, v in kwargs.items() if k != b
        }  # no del since mutable and shared between tests
    pl.add_mesh(Path(pv.examples.antfile).as_posix(), **kwargs)
    return pv.compare_images(outfile, pl)


@parametrize(tokens_kwargs=TOKENS_KWARGS, idgen=lambda **a: a['tokens_kwargs'][0])
def test_plot(
    tmp_path: Path,
    tokens_kwargs: tuple[str, dict],
    monkeypatch: pytest.MonkeyPatch,
):
    """Each set of `plot` arguments renders what the same arguments render through the API."""
    monkeypatch.setenv('PYVISTA_PLOT_THEME', 'testing')

    tokens, kwargs = tokens_kwargs
    outfile = (tmp_path / 'out.png').as_posix()

    main(f'{_plot_argv(outfile)} {tokens}')

    assert _plotter_difference(outfile, kwargs) < 200


@parametrize(as_script=[True, False])
def test_plot_entry_point(tmp_path: Path, as_script: bool, monkeypatch: pytest.MonkeyPatch):
    """Both `pyvista` and `python -m pyvista` render through the real entry point."""
    monkeypatch.setenv('PYVISTA_PLOT_THEME', 'testing')

    tokens, kwargs = TOKENS_KWARGS[0]
    outfile = (tmp_path / 'out.png').as_posix()
    args = ['pyvista'] if as_script else [sys.executable, '-m', 'pyvista']

    subprocess.run(
        [*args, *shlex.split(f'{_plot_argv(outfile)} {tokens}')],
        check=True,
        capture_output=True,
        encoding='utf-8',
    )

    assert _plotter_difference(outfile, kwargs) < 200
