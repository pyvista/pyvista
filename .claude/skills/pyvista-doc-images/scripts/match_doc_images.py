"""Plan and verify updates to ``tests/doc/doc_image_cache``."""

from __future__ import annotations

import argparse
import fnmatch
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


def slots(cache: Path, pattern: str) -> dict[str, list[Path]]:
    """Map each cached slot name to its baselines, a directory holding several."""
    found = {p.name: [p] for p in cache.glob(pattern) if p.is_file()}
    for d in cache.iterdir():
        if d.is_dir() and fnmatch.fnmatch(f'{d.name}.jpg', pattern):
            found[f'{d.name}.jpg'] = sorted(d.rglob('*.jpg'))
    return found


def best_match(variants: list[Path], generated: Path) -> tuple[float, Path | None]:
    """Return the closest baseline for one generated image, as pytest-pyvista grades it."""
    best: tuple[float, Path | None] = (float('inf'), None)
    gen_digest = digest(generated)
    for path in variants:
        if digest(path) == gen_digest:
            return 0.0, path
        value = score(path, generated)
        if value < best[0]:
            best = (value, path)
    return best


def describe(name: str, variants: list[Path], chosen: Path | None) -> str:
    """Name the matched baseline, mentioning the variant only for a directory slot."""
    if chosen is None or len(variants) == 1:
        return name
    return f'{name} [{chosen.name}]'


def classify(gen: Path, cache_slots: dict[str, list[Path]]) -> tuple[str, str]:
    """Return the action for one generated image, with a sentence explaining it."""
    gen_digest = digest(gen)
    identical = [
        name for name, paths in cache_slots.items() if any(digest(p) == gen_digest for p in paths)
    ]
    if identical:
        return (
            ('unchanged', 'byte-identical to its cache slot')
            if gen.name in identical
            else ('rename', f'byte-identical to {identical[0]}')
        )

    matches = {name: best_match(paths, gen) for name, paths in cache_slots.items()}
    scored = sorted((value, name, path) for name, (value, path) in matches.items())
    if not scored:
        return 'new', 'no cached files to compare against'

    best_value, best_name, best_path = scored[0]
    if best_value <= WARNING_THRESHOLD:
        if best_name == gen.name:
            return 'unchanged', 're-encoded, same render'
        target = describe(best_name, cache_slots[best_name], best_path)
        return 'rename', f'same render as {target} (value {best_value:.2f})'
    if gen.name in cache_slots:
        variants = cache_slots[gen.name]
        own_value, own_path = matches[gen.name]
        if len(variants) > 1:
            detail = (
                f'differs from every variant (value {own_value:.2f}); '
                f'refresh {own_path.name} only'  # type: ignore[union-attr]
            )
        else:
            detail = f'differs from its slot (value {own_value:.2f})'
        return 'replace', detail
    target = describe(best_name, cache_slots[best_name], best_path)
    return 'new', f'no cache slot; closest is {target} (value {best_value:.2f})'


def plan(cache: Path, generated: Path, pattern: str) -> int:
    """Report the action each generated image needs."""
    gen_files = sorted(generated.glob(pattern))
    if not gen_files:
        print(f'no generated images match {pattern!r} in {generated}')
        return 1
    if len(gen_files) > MAX_PLAN_FILES:
        print(f'{len(gen_files)} files match {pattern!r}; narrow it to one example first')
        return 1

    cache_slots = slots(cache, pattern)

    actions = dict.fromkeys(('unchanged', 'rename', 'replace', 'new'), 0)
    for gen in gen_files:
        action, detail = classify(gen, cache_slots)
        actions[action] += 1
        print(f'{gen.name:52} {action.upper():9} {detail}')

    orphans = sorted(set(cache_slots) - {p.name for p in gen_files})
    print('\n' + ', '.join(f'{k}={v}' for k, v in actions.items()))
    print(f'cached files the run did not generate: {orphans or "none"}')
    return 0


def verify(cache: Path, generated: Path, pattern: str) -> int:
    """Score every generated image against the file now sitting in its cache slot."""
    gen_files = sorted(generated.glob(pattern))
    if not gen_files:
        print(f'no generated images match {pattern!r} in {generated}')
        return 1

    cache_slots = slots(cache, pattern)
    failures = 0
    for gen in gen_files:
        variants = cache_slots.get(gen.name, [])
        if not variants:
            print(f'{gen.name:52} MISSING from the cache')
            failures += 1
            continue
        value, path = best_match(variants, gen)
        state = verdict(value)
        failures += state == 'ERROR'
        variant = f'  [{path.name}]' if len(variants) > 1 and path is not None else ''
        print(f'{gen.name:52} {value:10.2f}  {state}{variant}')

    orphans = sorted(set(cache_slots) - {p.name for p in gen_files})
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
