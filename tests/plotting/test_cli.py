from __future__ import annotations

from pathlib import Path
import shlex
import subprocess
import sys

import pytest
from pytest_cases import parametrize

import pyvista as pv
from pyvista.__main__ import main


@parametrize(
    tokens_kwargs=[
        ('--color=red', dict(color='red')),
        ('--color=red --opacity=0.1', dict(color='red', opacity=0.1)),
        ('--color=blue --culling="front"', dict(color='blue', culling='front')),
        ('--background="blue" --color red', dict(background='blue', color='red')),
    ],
    idgen=lambda **a: a['tokens_kwargs'][0],
)
@parametrize(as_script=[True, False])
@parametrize(with_main=[True, False])
@pytest.mark.no_default_theme
def test_plot(tmp_path: Path, tokens_kwargs: tuple[str, dict], as_script: bool, with_main: bool):
    """
    Test a real call to `pv.plot` using CLI and compare images to a Plotter output.
    """
    tokens, kwargs = tokens_kwargs

    infile = Path(pv.examples.antfile).as_posix()
    outfile = (tmp_path / 'out.png').as_posix()

    argv = f'plot {infile} --off-screen --screenshot={outfile} {tokens}'

    if not with_main:
        args = [sys.executable, '-m', 'pyvista'] if not as_script else ['pyvista']
        args += [*shlex.split(argv)]

        subprocess.run(
            args,
            check=True,
            capture_output=True,
            encoding='utf-8',
        )
    else:
        main(f'plot {infile} --off-screen --screenshot={outfile} {tokens}')

    pl = pv.Plotter()
    if (b := 'background') in kwargs:
        pl.set_background(kwargs[b])
        kwargs = {
            k: v for k, v in kwargs.items() if k != b
        }  # no del since mutable and shared between tests
    pl.add_mesh(infile, **kwargs)

    assert pv.compare_images(outfile, pl) < 200
