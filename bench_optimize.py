"""Benchmark the hot paths touched by the maint/optimize branch.

Measures import time in a subprocess and a set of attribute-access paths
in-process, then writes the results to JSON so two checkouts can be compared.

Usage
-----
Record a run on each checkout, then compare::

    git checkout main
    python bench_optimize.py --out main.json

    git checkout maint/optimize
    python bench_optimize.py --out branch.json

    python bench_optimize.py --compare main.json branch.json

Run it with the interpreter that has pyvista installed (an editable install
picks up whichever branch is checked out), e.g. ``./venv/bin/python``.

``--quick`` cuts the iteration counts for a faster, noisier answer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys
import textwrap
import timeit

# Each entry: (label, setup, statement). Kept as source strings so timeit runs
# them in a clean namespace, and so anything missing on one branch can be
# skipped rather than crashing the run.
BENCHMARKS: list[tuple[str, str, str]] = [
    (
        'Sphere() construction',
        'import pyvista as pv',
        'pv.Sphere()',
    ),
    (
        'getattr: mesh.n_points',
        'import pyvista as pv; m = pv.Sphere()',
        'm.n_points',
    ),
    (
        'getattr: mesh.points',
        'import pyvista as pv; m = pv.Sphere()',
        'm.points',
    ),
    (
        'getattr: CamelCase (mesh.GetBounds)',
        'import pyvista as pv; m = pv.Sphere()',
        'm.GetBounds',
    ),
    (
        'setattr: private (mesh._x = 1)',
        'import pyvista as pv; m = pv.Sphere()',
        "setattr(m, '_x', 1)",
    ),
    (
        'attribute miss: hasattr(mesh, ...)',
        'import pyvista as pv; m = pv.Sphere()',
        "hasattr(m, 'not_a_real_attribute')",
    ),
    (
        'check_attribute: snake_case name',
        'import pyvista as pv\n'
        'from pyvista.core._vtk_utilities import DisableVtkSnakeCase as D\n'
        'm = pv.Sphere()',
        "D.check_attribute(m, 'points')",
    ),
    (
        'check_attribute: CamelCase name',
        'import pyvista as pv\n'
        'from pyvista.core._vtk_utilities import DisableVtkSnakeCase as D\n'
        'm = pv.Sphere()',
        "D.check_attribute(m, 'GetBounds')",
    ),
    (
        'vtk_version_info < (9, 4)',
        'from pyvista.core._vtk_utilities import vtk_version_info as v',
        'v < (9, 4)',
    ),
    (
        'dir(mesh)',
        'import pyvista as pv; m = pv.Sphere()',
        'dir(m)',
    ),
    (
        'len(composite block_attr)',
        'import pyvista as pv\n'
        'mb = pv.MultiBlock([pv.Sphere(), pv.Cube()])\n'
        'mp = pv.CompositePolyDataMapper()\n'
        'mp.dataset = mb\n'
        'ba = mp.block_attr',
        'len(ba)',
    ),
]

# Statement used to time a cold `import pyvista` in a fresh interpreter.
_IMPORT_SNIPPET = textwrap.dedent("""
    import time

    start = time.perf_counter()
    import pyvista  # noqa: F401

    print(time.perf_counter() - start)
""")


def _git(*args: str) -> str:
    try:
        out = subprocess.run(
            ['git', *args],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).parent,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return 'unknown'
    return out.stdout.strip()


def measure_import(repeats: int) -> dict[str, float]:
    """Time `import pyvista` in a fresh interpreter, `repeats` times."""
    samples = []
    for i in range(repeats + 1):
        proc = subprocess.run(
            [sys.executable, '-c', _IMPORT_SNIPPET],
            capture_output=True,
            text=True,
            check=True,
        )
        if i == 0:
            continue  # discard warmup, which may compile .pyc files
        samples.append(float(proc.stdout.strip()) * 1e3)  # ms
    return {'median': statistics.median(samples), 'min': min(samples), 'unit': 'ms'}


def measure_one(setup: str, stmt: str, repeats: int) -> dict[str, float] | None:
    """Time a single statement, or return None if it does not run here.

    The iteration count is calibrated per benchmark: these range from ~20 ns
    (a tuple comparison) to ~1 ms (building a Sphere), so a single fixed count
    would either be noise-dominated at one end or take hours at the other.
    """
    try:
        timeit.timeit(stmt, setup=setup, number=1)
    except Exception as exc:  # noqa: BLE001 - a branch may lack this API
        print(f'    skipped: {type(exc).__name__}: {exc}', file=sys.stderr)
        return None
    timer = timeit.Timer(stmt, setup=setup)
    number, _ = timer.autorange()  # smallest 1/2/5*10**i giving >= 0.2 s
    samples = [t / number * 1e9 for t in timer.repeat(repeat=repeats, number=number)]  # ns
    return {
        'median': statistics.median(samples),
        'min': min(samples),
        'unit': 'ns',
        'number': number,
    }


def run(quick: bool) -> dict:
    repeats = 3 if quick else 7
    import_repeats = 3 if quick else 7

    results: dict[str, dict] = {}

    print(f'python   : {sys.executable}')
    print(f'branch   : {_git("rev-parse", "--abbrev-ref", "HEAD")}')
    print(f'commit   : {_git("rev-parse", "--short", "HEAD")}')
    print(f'settings : repeats={repeats}, iterations auto-calibrated per benchmark\n')

    print('import pyvista (fresh interpreter) ...')
    results['import pyvista'] = measure_import(import_repeats)

    for label, setup, stmt in BENCHMARKS:
        print(f'{label} ...')
        got = measure_one(setup, stmt, repeats)
        if got is not None:
            results[label] = got

    return {
        'branch': _git('rev-parse', '--abbrev-ref', 'HEAD'),
        'commit': _git('rev-parse', '--short', 'HEAD'),
        'python': sys.executable,
        'repeats': repeats,
        'results': results,
    }


def _fmt(value: float, unit: str) -> str:
    if unit == 'ms':
        return f'{value:.1f} ms'
    if value >= 1000:
        return f'{value / 1000:.2f} us'
    return f'{value:.1f} ns'


def show(data: dict) -> None:
    print(f'\n{"benchmark":<38} {"median":>12} {"min":>12}')
    print('-' * 64)
    for label, got in data['results'].items():
        print(
            f'{label:<38} {_fmt(got["median"], got["unit"]):>12} {_fmt(got["min"], got["unit"]):>12}'
        )


def compare(before_path: str, after_path: str) -> None:
    before = json.loads(Path(before_path).read_text())
    after = json.loads(Path(after_path).read_text())

    print(f'before : {before["branch"]} @ {before["commit"]}')
    print(f'after  : {after["branch"]} @ {after["commit"]}')
    print(f'\n{"benchmark":<38} {"before":>11} {"after":>11} {"change":>9}')
    print('-' * 72)

    for label, b in before['results'].items():
        a = after['results'].get(label)
        if a is None:
            print(f'{label:<38} {"-":>11} {"n/a":>11} {"":>9}')
            continue
        # Compare medians; min is reported by `show` for reference.
        delta = (a['median'] - b['median']) / b['median'] * 100
        mark = '' if abs(delta) < 3 else ('  <-- faster' if delta < 0 else '  <-- SLOWER')
        print(
            f'{label:<38} {_fmt(b["median"], b["unit"]):>11} '
            f'{_fmt(a["median"], a["unit"]):>11} {delta:>+8.1f}%{mark}'
        )
    print('\nChanges under 3% are within noise on most machines; re-run to confirm.')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', metavar='FILE', help='write results to FILE as JSON')
    parser.add_argument(
        '--compare',
        nargs=2,
        metavar=('BEFORE', 'AFTER'),
        help='compare two JSON files instead of measuring',
    )
    parser.add_argument('--quick', action='store_true', help='fewer iterations, noisier')
    args = parser.parse_args()

    if args.compare:
        compare(*args.compare)
        return

    data = run(args.quick)
    show(data)
    if args.out:
        Path(args.out).write_text(json.dumps(data, indent=2))
        print(f'\nwrote {args.out}')


if __name__ == '__main__':
    main()
