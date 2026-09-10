"""Print the web URL of every downloadable example, one per line, for lychee."""

from __future__ import annotations

import sys

from pyvista.examples._dataset_loader import _DOWNLOADABLE_TYPES
from pyvista.examples._get_example import _supported_modules


def main() -> None:
    """Collect the URLs of every downloadable loader and print them sorted."""
    urls: set[str] = set()
    for module in _supported_modules():
        for loader in vars(module).values():
            if isinstance(loader, _DOWNLOADABLE_TYPES):
                urls.update(loader.web_urls)
    if not urls:
        sys.exit('No example dataset URLs found.')
    print('\n'.join(sorted(urls)))


if __name__ == '__main__':
    main()
