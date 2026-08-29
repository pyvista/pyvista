#!/usr/bin/env python
"""Benchmark: one representative pyvista session, run on two checkouts.

`maint/optimize` replaces a batch of function-local ``import`` statements
(marked ``# noqa: PLC0415`` on main) with module-level imports or
precomputed constants, on functions that sit in genuinely hot paths:

  * ``_NoNewAttrMixin.__setattr__``            -> runs on every attribute SET
  * ``DisableVtkSnakeCase.__getattribute__``   -> runs on every attribute GET
  * ``DataObject.__getattribute__`` / ``__dir__``   -> accessor-plugin resolution
  * ``BasePlotter.__getattr__`` / ``__dir__``       -> component resolution
  * ``CompositeAttributes.__len__``                 -> MultiBlock counting
  * ``Property`` attribute access (anisotropy, repr)
  * ``DataSetFilters.gaussian_splatting``           -> _validation import per call

None of these are import-*time* hot paths in the "cold `import pyvista`"
sense -- they're the import *machinery* being invoked repeatedly at
runtime. So rather than a big table of isolated micro-benchmarks, this
times ONE realistic session end-to-end (build meshes, assemble a composite
scene, hammer attribute get/set the way real code does, drive an
off-screen Plotter) many times and reports the wall-clock distribution.
Cold `import pyvista` is measured separately, for completeness.

Usage
-----
::

    git checkout main
    .venv/bin/python bench_hotpaths.py --out main.json

    git checkout maint/optimize
    .venv/bin/python bench_hotpaths.py --out branch.json

    .venv/bin/python bench_hotpaths.py --compare main.json branch.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault('PYVISTA_OFF_SCREEN', 'true')

REPO = Path.cwd()


def _git(*args: str) -> str:
    try:
        out = subprocess.run(
            ['git', *args], capture_output=True, text=True, check=True, cwd=str(REPO)
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return 'unknown'
    return out.stdout.strip()


def measure_cold_import(repeats: int) -> list[float]:
    """Time `import pyvista` in a fresh interpreter, `repeats` times (ms)."""
    snippet = (
        "import os, time\n"
        "os.environ.setdefault('PYVISTA_OFF_SCREEN', 'true')\n"
        "start = time.perf_counter()\n"
        "import pyvista  # noqa: F401\n"
        "print(time.perf_counter() - start)\n"
    )
    samples = []
    for i in range(repeats + 1):
        proc = subprocess.run(
            [sys.executable, '-c', snippet], capture_output=True, text=True, check=True
        )
        if i == 0:
            continue  # discard warmup (may compile .pyc files)
        samples.append(float(proc.stdout.strip()) * 1e3)
    return samples


def _build_meshes(pv):
    return [pv.Sphere(), pv.Cube(), pv.Cylinder(), pv.Cone(), pv.Plane(), pv.Icosahedron()]


def run_scenario(pv, attr_rounds: int) -> None:
    """One realistic session, touching most of the hot paths above."""
    meshes = _build_meshes(pv)  # heavy on __init__/__setattr__

    # Attribute get/set hot loop: snake_case getattr, CamelCase getattr
    # (both go through DisableVtkSnakeCase.__getattribute__), public and
    # private setattr (_NoNewAttrMixin.__setattr__), and an attribute-miss
    # hasattr (DataObject.__getattribute__ -> accessor_registry lookup).
    for _ in range(attr_rounds):
        for m in meshes:
            _ = m.n_points
            _ = m.points
            _ = m.bounds
            _ = m.GetNumberOfPoints()
            m._scratch = 1
            m.active_scalars_name = m.active_scalars_name
            hasattr(m, 'not_a_real_attribute')
    for m in meshes:
        dir(m)

    # Composite scene: CompositeAttributes.__len__ used to do
    # `from pyvista import MultiBlock` on every call.
    mb = pv.MultiBlock(meshes)
    mapper = pv.CompositePolyDataMapper()
    mapper.dataset = mb
    block_attr = mapper.block_attr
    for _ in range(attr_rounds):
        len(block_attr)

    # Property objects: color/opacity/anisotropy access and repr(), whose
    # anisotropy path used to do a lazy VTKVersionError import.
    for _ in range(max(attr_rounds // 4, 1)):
        prop = pv.Property()
        _ = prop.color
        _ = prop.opacity
        _ = prop.anisotropy
        repr(prop)

    # Filter whose body used to do `from pyvista.core import _validation`
    # on every call. Kept small (few sample points) so it stays cheap.
    for m in meshes[:2]:
        m.gaussian_splatting(radius=0.05, dimensions=(8, 8, 8))

    # Plotter session: BasePlotter.__getattr__/__dir__ (component-registry
    # resolution), off-screen so nothing pops up on screen.
    pl = pv.Plotter(off_screen=True, window_size=(100, 100))
    for m in meshes:
        pl.add_mesh(m)
    dir(pl)
    hasattr(pl, 'not_a_real_component')
    pl.show(auto_close=False)
    pl.close()


def measure_scenario(repeats: int, attr_rounds: int) -> list[float]:
    import pyvista as pv

    pv.OFF_SCREEN = True

    # Warmup: pays for lazy plugin/module imports once so they don't leak
    # into the per-iteration timings.
    run_scenario(pv, attr_rounds=1)

    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        run_scenario(pv, attr_rounds=attr_rounds)
        samples.append((time.perf_counter() - start) * 1e3)
    return samples


def _stats(samples: list[float]) -> dict:
    return {
        'median': statistics.median(samples),
        'mean': statistics.mean(samples),
        'stdev': statistics.stdev(samples) if len(samples) > 1 else 0.0,
        'min': min(samples),
        'max': max(samples),
        'n': len(samples),
        'unit': 'ms',
    }


def run(repeats: int, attr_rounds: int) -> dict:
    print(f'python     : {sys.executable}')
    print(f'branch     : {_git("rev-parse", "--abbrev-ref", "HEAD")}')
    print(f'commit     : {_git("rev-parse", "--short", "HEAD")}')
    print(f'repeats    : {repeats}, attr_rounds={attr_rounds}\n')

    print('cold `import pyvista` ...')
    import_samples = measure_cold_import(repeats)

    print('scenario (warmup + timed repeats) ...')
    scenario_samples = measure_scenario(repeats, attr_rounds)

    results = {
        'import pyvista (cold)': _stats(import_samples),
        'scenario (end-to-end)': _stats(scenario_samples),
    }
    return {
        'branch': _git('rev-parse', '--abbrev-ref', 'HEAD'),
        'commit': _git('rev-parse', '--short', 'HEAD'),
        'python': sys.executable,
        'repeats': repeats,
        'attr_rounds': attr_rounds,
        'results': results,
    }


def show(data: dict) -> None:
    print(f'\n{"benchmark":<26} {"median":>10} {"mean":>10} {"stdev":>9} {"min":>10}')
    print('-' * 68)
    for label, s in data['results'].items():
        print(
            f'{label:<26} {s["median"]:>8.2f}ms {s["mean"]:>8.2f}ms '
            f'{s["stdev"]:>7.2f}ms {s["min"]:>8.2f}ms'
        )


def compare(before_path: str, after_path: str) -> None:
    before = json.loads(Path(before_path).read_text())
    after = json.loads(Path(after_path).read_text())

    print(f'before : {before["branch"]} @ {before["commit"]}')
    print(f'after  : {after["branch"]} @ {after["commit"]}')
    print(f'\n{"benchmark":<26} {"before":>10} {"after":>10} {"change":>9}')
    print('-' * 60)
    for label, b in before['results'].items():
        a = after['results'].get(label)
        if a is None:
            print(f'{label:<26} {"-":>10} {"n/a":>10}')
            continue
        delta = (a['median'] - b['median']) / b['median'] * 100
        mark = '' if abs(delta) < 3 else ('  <-- faster' if delta < 0 else '  <-- SLOWER')
        print(f'{label:<26} {b["median"]:>8.2f}ms {a["median"]:>8.2f}ms {delta:>+8.1f}%{mark}')
    print('\nChanges under ~3% are within noise on a laptop; re-run to confirm.')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', metavar='FILE', help='write results to FILE as JSON')
    parser.add_argument(
        '--compare', nargs=2, metavar=('BEFORE', 'AFTER'), help='compare two JSON files'
    )
    parser.add_argument('--repeats', type=int, default=25, help='timed scenario repeats')
    parser.add_argument(
        '--attr-rounds', type=int, default=300, help='attribute get/set rounds per scenario'
    )
    args = parser.parse_args()

    if args.compare:
        compare(*args.compare)
        return

    data = run(args.repeats, args.attr_rounds)
    show(data)
    if args.out:
        Path(args.out).write_text(json.dumps(data, indent=2))
        print(f'\nwrote {args.out}')


if __name__ == '__main__':
    main()
