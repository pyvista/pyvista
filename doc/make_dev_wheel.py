"""Build a development wheel and write a PEP 503 simple index into the docs.

Produces a ``wheels/`` tree under the Sphinx HTML output that is bundled into
the same Netlify deploy as the rest of the docs. Once published to
``dev.pyvista.org``, users can install the latest ``main``-branch build with::

    pip install --pre --index-url https://dev.pyvista.org/wheels/simple/ pyvista

The wheel version is ``<base>+g<shortsha>`` (PEP 440 local version segment),
so every commit produces a unique wheel while still sorting under the same
base version. Pip's resolver treats the rolling URL as authoritative: a fresh
``pip install --pre -U pyvista`` always pulls the newest commit's wheel.

These are unsupported development builds rebuilt on every push to ``main`` —
not nightlies, not releases.
"""

from __future__ import annotations

import argparse
import ast
from html import escape
import logging
from pathlib import Path
import re
import shutil
import subprocess
import sys

logger = logging.getLogger('make_dev_wheel')

REPO_ROOT = Path(__file__).resolve().parent.parent
VERSION_FILE = REPO_ROOT / 'pyvista' / '_version.py'


def short_sha() -> str:
    """Return the short git SHA of HEAD."""
    return subprocess.check_output(
        ['git', '-C', str(REPO_ROOT), 'rev-parse', '--short=8', 'HEAD'],
        text=True,
    ).strip()


def read_base_version() -> str:
    """Read the version tuple from pyvista/_version.py without importing it."""
    tree = ast.parse(VERSION_FILE.read_text())
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == 'version_info'
            and isinstance(node.value, ast.Tuple)
        ):
            parts = [str(ast.literal_eval(elt)) for elt in node.value.elts]
            return '.'.join(parts)
    msg = f'Could not find version_info assignment in {VERSION_FILE}'
    raise RuntimeError(msg)


def patch_version(local: str) -> str:
    """Append ``+<local>`` to ``__version__`` and return the original text."""
    original = VERSION_FILE.read_text()
    base = read_base_version()
    new_version = f'{base}+{local}'
    patched = re.sub(
        r'^__version__\s*=.*$',
        f"__version__ = '{new_version}'",
        original,
        count=1,
        flags=re.MULTILINE,
    )
    VERSION_FILE.write_text(patched)
    return original


def build_wheel(out_dir: Path) -> Path:
    """Build a wheel into ``out_dir`` and return the wheel path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(
        [sys.executable, '-m', 'build', '--wheel', '--outdir', str(out_dir)],
        cwd=REPO_ROOT,
    )
    wheels = sorted(out_dir.glob('pyvista-*.whl'))
    if not wheels:
        msg = f'No wheel produced in {out_dir}'
        raise RuntimeError(msg)
    return wheels[-1]


SIMPLE_PROJECT_TMPL = """<!DOCTYPE html>
<html>
  <head>
    <meta name="pypi:repository-version" content="1.0">
    <title>Links for {project}</title>
  </head>
  <body>
    <h1>Links for {project}</h1>
{links}
  </body>
</html>
"""

SIMPLE_ROOT_TMPL = """<!DOCTYPE html>
<html>
  <head>
    <meta name="pypi:repository-version" content="1.0">
    <title>PyVista development wheels</title>
  </head>
  <body>
    <a href="pyvista/">pyvista</a>
  </body>
</html>
"""

LANDING_TMPL = """<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <title>PyVista development wheels</title>
    <style>
      body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
             max-width: 720px; margin: 3em auto; padding: 0 1em; line-height: 1.5; }}
      code, pre {{ background: #f4f4f4; border-radius: 4px; }}
      code {{ padding: 0.1em 0.3em; }}
      pre {{ padding: 0.8em 1em; overflow-x: auto; }}
      .meta {{ color: #666; font-size: 0.9em; }}
    </style>
  </head>
  <body>
    <h1>PyVista development wheels</h1>
    <p>Install the latest <code>main</code>-branch build of PyVista:</p>
    <pre>pip install --pre --index-url https://dev.pyvista.org/wheels/simple/ pyvista</pre>
    <p>To keep PyPI as the source for dependencies and only pull
    <code>pyvista</code> itself from here:</p>
    <pre>pip install --pre --extra-index-url https://dev.pyvista.org/wheels/simple/ pyvista</pre>
    <p class="meta">
      Current build: <code>{wheel}</code><br>
      Commit: <code>{sha}</code><br>
      These wheels are unsupported development builds rebuilt on every push to
      <code>main</code>. For stable releases, install from PyPI as usual.
    </p>
  </body>
</html>
"""


def write_simple_index(wheels_root: Path, wheel: Path) -> None:
    """Write a PEP 503 simple index for the given wheel."""
    project_dir = wheels_root / 'simple' / 'pyvista'
    project_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(wheel, project_dir / wheel.name)
    link = f'    <a href="{escape(wheel.name)}">{escape(wheel.name)}</a><br>'
    (project_dir / 'index.html').write_text(
        SIMPLE_PROJECT_TMPL.format(project='pyvista', links=link),
    )
    (wheels_root / 'simple' / 'index.html').write_text(SIMPLE_ROOT_TMPL)


def main() -> int:
    """Build the dev wheel and write the simple index into the docs HTML dir."""
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--html-dir',
        type=Path,
        default=REPO_ROOT / 'doc' / '_build' / 'html',
        help='Sphinx HTML output directory (the wheels/ tree is written here).',
    )
    parser.add_argument(
        '--sha',
        default=None,
        help='Short commit SHA to embed in the version. Defaults to HEAD.',
    )
    args = parser.parse_args()

    sha = args.sha or short_sha()
    local = f'g{sha}'

    wheels_root = args.html_dir / 'wheels'
    build_dir = wheels_root / '_build'
    if build_dir.exists():
        shutil.rmtree(build_dir)

    original_version = patch_version(local)
    try:
        wheel = build_wheel(build_dir)
    finally:
        VERSION_FILE.write_text(original_version)

    write_simple_index(wheels_root, wheel)
    (wheels_root / 'index.html').write_text(
        LANDING_TMPL.format(wheel=escape(wheel.name), sha=escape(sha)),
    )
    shutil.rmtree(build_dir, ignore_errors=True)

    logger.info('Dev wheel: %s', wheel.name)
    logger.info('Index written to: %s', wheels_root / 'simple')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
