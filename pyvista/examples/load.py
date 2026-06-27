"""Load any example dataset."""

from __future__ import annotations

from pathlib import Path
from types import FunctionType

from pyvista.examples._dataset_loader import DatasetObject
from pyvista.examples._dataset_loader import _Downloadable
from pyvista.examples._dataset_loader import _get_dataset_loader


def load(
    name: str | FunctionType, *, return_paths: bool = False, allow_download: bool = True
) -> tuple[DatasetObject, Path | tuple[Path] | None]:
    """Load an example dataset.

    Parameters
    ----------
    name : str | FunctionType
        Name of the example dataset to load.

    return_paths : bool, default: False
        If ``True``, return the file path(s) for the example.

    allow_download : bool, default: True
        If ``False``, a ``ValueError`` is raised if the example must
        first be downloaded. The error is raised even if the file
        was previously downloaded.

    Returns
    -------
    DataSet | MultiBlock | Texture | ndarray
        Loaded dataset and file paths depending on ``return_paths``.

    """
    dataset_loader = _get_dataset_loader(name)

    # Download if necessary
    if isinstance(dataset_loader, _Downloadable):
        if not allow_download:
            name_str = name.__name__ if isinstance(name, FunctionType) else name
            msg = f'Example {name_str!r} requires download.'
            raise ValueError(msg)
        dataset_loader.download()

    dataset = dataset_loader.load()

    # Process paths
    if return_paths:
        path = getattr(dataset_loader, 'path', None)
        if path:
            # Convert to Path objects
            path_out = Path(path) if isinstance(path, str) else tuple(Path(p) for p in path)
        else:
            path_out = None
        return dataset, path_out
    return dataset
