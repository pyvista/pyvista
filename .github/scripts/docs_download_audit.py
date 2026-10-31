"""Run every example download of a built documentation tree and report the failures."""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

SUFFIXES = {'marimo': '.marimo.py', 'jupyter': '.ipynb', 'python': '.py'}
FORMATS = ('python', 'jupyter', 'marimo')
OUTPUT_TAIL = 4000
ENV = {
    'PYVISTA_OFF_SCREEN': 'true',
    'PYVISTA_JUPYTER_BACKEND': 'static',
    'MPLBACKEND': 'Agg',
    'PYDEVD_DISABLE_FILE_VALIDATION': '1',
}


def stem(path: Path) -> str:
    """Return the example name shared by all formats of one download."""
    name = path.name
    for suffix in SUFFIXES.values():
        if name.endswith(suffix):
            return name.removesuffix(suffix)
    return path.stem


def collect(downloads: Path, fmt: str) -> list[Path]:
    """Return the downloads of one format, sorted."""
    files = downloads.rglob(f'*{SUFFIXES[fmt]}')
    if fmt == 'python':
        files = (f for f in files if not f.name.endswith(SUFFIXES['marimo']))
    return sorted(files)


def commands(path: Path, fmt: str, tmp: str) -> list[tuple[str, list[str]]]:
    """Return the named stages that check and execute one download."""
    if fmt == 'python':
        return [('run', [sys.executable, str(path)])]
    if fmt == 'jupyter':
        execute = [
            *(sys.executable, '-m', 'jupyter', 'nbconvert', '--to', 'notebook', '--execute'),
            *('--ExecutePreprocessor.timeout=-1', '--output-dir', tmp, '--output', 'executed'),
        ]
        return [('run', [*execute, str(path)])]
    return [
        ('check', [sys.executable, '-m', 'marimo', 'check', str(path)]),
        ('run', [sys.executable, str(path)]),
    ]


def execute(path: Path, fmt: str, timeout: float) -> dict:
    """Run one download in its own temporary directory and return its result."""
    env = {**os.environ, **ENV}
    start = time.monotonic()
    result = {'file': str(path), 'name': stem(path), 'format': fmt, 'ok': True}
    with tempfile.TemporaryDirectory(prefix='docs-download-audit-') as tmp:
        for stage, cmd in commands(path, fmt, tmp):
            try:
                proc = subprocess.run(
                    cmd,
                    cwd=tmp,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    check=False,
                )
                code, output = proc.returncode, proc.stdout + proc.stderr
            except subprocess.TimeoutExpired as exc:
                code, output = None, f'Timed out after {timeout:g} s\n{exc.stdout or ""}'
            if code != 0:
                # `marimo check` leads with the diagnostic; a traceback ends with the error.
                output = output[:OUTPUT_TAIL] if stage == 'check' else output[-OUTPUT_TAIL:]
                result.update(ok=False, stage=stage, code=code, output=output)
                break
    result['seconds'] = round(time.monotonic() - start, 1)
    return result


def run(args: argparse.Namespace) -> None:
    """Execute every download of one format and write the results as JSON lines."""
    args.downloads = args.downloads.resolve()
    files = collect(args.downloads, args.format)[: args.limit]
    print(f'{len(files)} {args.format} downloads in {args.downloads}', flush=True)
    failed = 0
    with ProcessPoolExecutor(args.workers) as pool, args.output.open('w') as out:
        futures = [pool.submit(execute, f, args.format, args.timeout) for f in files]
        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            result['file'] = str(Path(result['file']).relative_to(args.downloads))
            out.write(json.dumps(result) + '\n')
            out.flush()
            if not result['ok']:
                failed += 1
                print(f'FAIL {result["file"]} ({result["stage"]})', flush=True)
            if i % 100 == 0:
                print(f'{i}/{len(files)} done, {failed} failed', flush=True)
    print(f'{len(files)} run, {failed} failed', flush=True)


def error_line(result: dict) -> str:
    """Return the line of a failure's output that names the error, safe for a table cell."""
    lines = [line.strip() for line in result.get('output', '').splitlines() if line.strip()]
    if not lines:
        text = f'exit code {result.get("code")}'
    else:
        text = lines[0] if result['stage'] == 'check' else lines[-1]
    text = text.replace('|', '\\|').replace('`', "'")
    return text if len(text) <= 160 else text[:157] + '...'


def table(rows: list[dict], *, also: dict[str, list[str]] | None = None) -> list[str]:
    """Return a Markdown table of failures, one row per distinct example and error."""
    if also is None:
        lines = ['| Example | Format | Stage | Error |', '| --- | --- | --- | --- |']
    else:
        lines = ['| Example | Also fails as | Error |', '| --- | --- | --- |']
    for r in sorted(rows, key=lambda r: (r['name'], r['format'])):
        name = Path(r['file']).name
        if also is None:
            row = f'| `{name}` | {r["format"]} | {r["stage"]} | `{error_line(r)}` |'
        else:
            row = f'| `{name}` | {", ".join(also[r["name"]]) or "-"} | `{error_line(r)}` |'
        if row not in lines:
            lines.append(row)
    return lines


def report(args: argparse.Namespace) -> None:
    """Classify the failures of every result file and write a Markdown report."""
    results = [
        json.loads(line)
        for path in args.results
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    counts = {fmt: [0, 0] for fmt in FORMATS}
    python_ok: dict[str, bool] = {}
    for r in results:
        counts[r['format']][0] += 1
        counts[r['format']][1] += not r['ok']
        if r['format'] == 'python':
            python_ok[r['name']] = python_ok.get(r['name'], True) and r['ok']

    failures = [r for r in results if not r['ok']]
    python_failures = [r for r in failures if r['format'] == 'python']
    format_failures = [
        r for r in failures if r['format'] != 'python' and python_ok.get(r['name'], True)
    ]
    also: dict[str, list[str]] = defaultdict(list)
    for r in failures:
        if r['format'] != 'python' and not python_ok.get(r['name'], True):
            also[r['name']].append(r['format'])

    lines = ['| Format | Downloads | Failed |', '| --- | --- | --- |']
    lines += [f'| {fmt} | {n} | {bad} |' for fmt, (n, bad) in counts.items()]
    lines += ['', f'#### Format-specific failures ({len(format_failures)})', '']
    if format_failures:
        lines += [
            'These fail as a notebook while the plain Python script of the same example runs.',
            '',
            *table(format_failures),
        ]
    else:
        lines.append('None.')
    n_python = len({r['name'] for r in python_failures})
    lines += ['', f'#### Plain Python failures ({n_python})', '']
    if python_failures:
        lines += [
            '<details>',
            '<summary>Examples that fail as a plain Python script</summary>',
            '',
            *table(python_failures, also=also),
            '',
            '</details>',
        ]
    else:
        lines.append('None.')
    text = '\n'.join(lines) + '\n'
    args.output.write_text(text)
    print(text)
    if github_output := os.environ.get('GITHUB_OUTPUT'):
        with Path(github_output).open('a') as out:
            out.write(f'format-failures={len(format_failures)}\n')
            out.write(f'python-failures={n_python}\n')


def main() -> None:
    """Parse the command line and run the chosen subcommand."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(required=True)

    run_parser = sub.add_parser('run', help='execute the downloads of one format')
    run_parser.add_argument('downloads', type=Path, help='the _downloads directory')
    run_parser.add_argument('--format', choices=FORMATS, required=True)
    run_parser.add_argument('--output', type=Path, required=True, help='JSON lines file')
    run_parser.add_argument('--workers', type=int, default=os.cpu_count())
    run_parser.add_argument('--timeout', type=float, default=300, help='seconds per stage')
    run_parser.add_argument('--limit', type=int, help='run only the first N files')
    run_parser.set_defaults(func=run)

    report_parser = sub.add_parser('report', help='summarize result files as Markdown')
    report_parser.add_argument('results', type=Path, nargs='*', help='JSON lines files')
    report_parser.add_argument('--output', type=Path, required=True, help='Markdown file')
    report_parser.set_defaults(func=report)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
