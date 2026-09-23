"""Write the web URLs of every downloadable example, one file per loader, for lychee."""

from __future__ import annotations

from pathlib import Path
import sys

from pyvista.examples._dataset_loader import _DOWNLOADABLE_TYPES
from pyvista.examples._get_example import _supported_modules


def main(directory: Path) -> None:
    """Write each loader's URLs to ``<module>.<loader>.txt`` under ``directory``."""
    directory.mkdir(parents=True, exist_ok=True)
    written = 0
    for module in _supported_modules():
        prefix = module.__name__.rpartition('.')[2]
        for name, loader in vars(module).items():
            if isinstance(loader, _DOWNLOADABLE_TYPES) and loader.web_urls:
                path = directory / f'{prefix}.{name}.txt'
                path.write_text('\n'.join(loader.web_urls) + '\n')
                written += 1
    if not written:
        sys.exit('No example dataset loaders found.')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit(f'usage: {sys.argv[0]} DIRECTORY')
    main(Path(sys.argv[1]))
