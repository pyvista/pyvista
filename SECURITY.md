# Security Policy

## Supported Versions

Security vulnerability reports will be accepted and acted upon for all released versions.

## Reporting a Vulnerability

Allow the community to report potential security vulnerabilities to maintainers and repository owners privately.
[Learn more about private vulnerability reporting](https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing/privately-reporting-a-security-vulnerability).

## Trust model

PyVista wraps VTK and meshio. Mesh parsers inherit upstream parser CVEs. **Do not load mesh files from untrusted sources.** PyVista is a viewer/processor for trusted scientific data; it is not a sandbox.

## Pickle is not a file format

PyVista **refuses** `.pkl` / `.pickle` extensions in `pyvista.read()` and `DataObject.save()`. The top-level `pyvista.read_pickle` / `pyvista.save_pickle` API shims remain importable for backwards compatibility but always raise `ValueError`.

**Why.** Pickle is a Python serialization protocol, not a mesh file format. Unpickling untrusted data is arbitrary code execution (CWE-502). A `pyvista.read(path)` call with an attacker-influenced `path` (downloaded data, shared notebook, copy-pasted command) would be a one-shot RCE if pickle dispatch were enabled.

**What still works.** Python's pickle protocol via `DataObject.__getstate__` / `__setstate__` is unchanged — `multiprocessing`, `dask`, and `joblib` continue to work. `pyvista.set_pickle_format()` tunes that in-memory protocol.

**Migration.**

- For mesh files: use a real mesh format (`.vtu`, `.vtp`, `.vtm`, `.vtk`, `.ply`, `.stl`, ...).
- For fast single-blob round-trips: install [`pyvista-zstd`](https://github.com/pyvista/pyvista-zstd) (`pip install pyvista[io]`) and use the `.pv` format.
