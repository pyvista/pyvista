"""Print the source URL of every downloadable example, one per line, for lychee."""

# ruff: noqa: INP001
from __future__ import annotations

import sys
from typing import get_args

from pyvista.examples._dataset_loader import _DOWNLOADABLE_TYPES
from pyvista.examples._get_example import ExampleName
from pyvista.examples._get_example import _get_dataset_loader

urls: set[str] = set()
for name in get_args(ExampleName):
    loader, _, _ = _get_dataset_loader(name)
    if isinstance(loader, _DOWNLOADABLE_TYPES):
        urls.update(loader.web_urls)

if not urls:
    sys.exit('No example dataset URLs found.')
sys.stdout.write('\n'.join(sorted(urls)) + '\n')
