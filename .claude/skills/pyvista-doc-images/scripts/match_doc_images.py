"""Plan and verify updates to ``tests/doc/doc_image_cache``."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys

import pyvista as pv

ERROR_THRESHOLD = 500.0
WARNING_THRESHOLD = 200.0
MAX_PLAN_FILES = 80


def digest(path: Path) -> str:
    """Return the sha256 of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def score(cached: Path, generated: Path) -> float:
    """Return pyvista's image comparison value, or ``inf`` for mismatched shapes."""
    try:
        return float(pv.compare_images(str(cached), str(generated)))
    except Exception:  # noqa: BLE001 - differing dimensions raise
        return float('inf')


def verdict(value: float) -> str:
    """Classify a comparison value against the pytest-pyvista thresholds."""
    if value > ERROR_THRESHOLD:
        return 'ERROR'
    return 'warn' if value > WARNING_THRESHOLD else 'ok'


def classify(
    gen: Path, cache_files: dict[str, Path], cache_digests: dict[str, str]
) -> tuple[str, str]:
    """Return the action for one generated image, with a sentence explaining it."""
    gen_digest = digest(gen)
    identical = [name for name, d in cache_digests.items() if d == gen_digest]
    if identical:
        if gen.name in identical:
            return 'unchanged', 'byte-identical to its cache slot'
        return 'rename', f'byte-identical to {identical[0]}'

    scores = sorted((score(path, gen), name) for name, path in cache_files.items())
    if not scores:
        return 'new', 'no cached files to compare against'

    best_value, best_name = scores[0]
    if best_value <= WARNING_THRESHOLD:
        same_slot = best_name == gen.name
        detail = 're-encoded, same render' if same_slot else f'same render as {best_name}'
        return 'unchanged' if same_slot else 'rename', f'{detail} (value {best_value:.2f})'
    if gen.name in cache_files:
        own = score(cache_files[gen.name], gen)
        return 'replace', f'differs from its slot (value {own:.2f})'
    return 'new', f'no cache slot; closest is {best_name} (value {best_value:.2f})'


def plan(cache: Path, generated: Path, pattern: str) -> int:
    """Report the action each generated image needs."""
    gen_files = sorted(generated.glob(pattern))
    if not gen_files:
        print(f'no generated images match {pattern!r} in {generated}')
        return 1
    if len(gen_files) > MAX_PLAN_FILES:
        print(f'{len(gen_files)} files match {pattern!r}; narrow it to one example first')
        return 1

    cache_files = {p.name: p for p in cache.glob(pattern)}
    cache_digests = {name: digest(path) for name, path in cache_files.items()}

    actions = dict.fromkeys(('unchanged', 'rename', 'replace', 'new'), 0)
    for gen in gen_files:
        action, detail = classify(gen, cache_files, cache_digests)
        actions[action] += 1
        print(f'{gen.name:52} {action.upper():9} {detail}')

    orphans = sorted(set(cache_files) - {p.name for p in gen_files})
    print('\n' + ', '.join(f'{k}={v}' for k, v in actions.items()))
    print(f'cached files the run did not generate: {orphans or "none"}')
    return 0


def verify(cache: Path, generated: Path, pattern: str) -> int:
    """Score every generated image against the file now sitting in its cache slot."""
    gen_files = sorted(generated.glob(pattern))
    if not gen_files:
        print(f'no generated images match {pattern!r} in {generated}')
        return 1

    failures = 0
    for gen in gen_files:
        cached = cache / gen.name
        if not cached.exists():
            print(f'{gen.name:52} MISSING from the cache')
            failures += 1
            continue
        value = score(cached, gen)
        state = verdict(value)
        failures += state == 'ERROR'
        print(f'{gen.name:52} {value:10.2f}  {state}')

    orphans = sorted({p.name for p in cache.glob(pattern)} - {p.name for p in gen_files})
    print(f'\ncached files the run did not generate: {orphans or "none"}')
    print(f'slots above the {ERROR_THRESHOLD:.0f} error threshold: {failures}')
    return 1 if failures else 0


def main() -> int:
    """Parse arguments and dispatch."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['plan', 'verify'])
    parser.add_argument('generated', type=Path, help='extracted doc-generated-test-images')
    parser.add_argument('--cache', type=Path, default=Path('tests/doc/doc_image_cache'))
    parser.add_argument('--pattern', default='*.jpg', help="e.g. 'sphx_glr_axes_objects_*.jpg'")
    args = parser.parse_args()

    if not args.cache.is_dir():
        print(f'cache directory not found: {args.cache}')
        return 1
    runner = plan if args.mode == 'plan' else verify
    return runner(args.cache, args.generated, args.pattern)


if __name__ == '__main__':
    sys.exit(main())
