<p align="center">
  <img src="https://github.com/pyvista/pyvista/raw/main/doc/source/_static/pyvista_logo.svg" alt="PyVista" width="400" />
</p>

<p align="center">
  <strong>3D visualization and mesh analysis for science and engineering</strong>
</p>

<p align="center">
  <a href="https://docs.pyvista.org/examples/index.html">
    <img src="https://github.com/pyvista/pyvista/raw/main/doc/source/_static/pyvista_banner_small.png" alt="PyVista examples gallery" width="100%" />
  </a>
</p>

<p align="center">
  <a href="https://pypi.org/project/pyvista/"><img src="https://img.shields.io/pypi/v/pyvista.svg?logo=python&logoColor=white" alt="PyPI" /></a>
  <a href="https://anaconda.org/conda-forge/pyvista"><img src="https://img.shields.io/conda/vn/conda-forge/pyvista.svg?logo=conda-forge&logoColor=white" alt="Conda" /></a>
  <a href="https://numfocus.org/sponsored-projects/affiliated-projects"><img src="https://img.shields.io/badge/affiliated-NumFOCUS-orange.svg?style=flat&colorA=E1523D&colorB=007D8A" alt="NumFOCUS Affiliated" /></a>
  <a href="https://doi.org/10.21105/joss.01450"><img src="http://joss.theoj.org/papers/10.21105/joss.01450/status.svg" alt="JOSS paper" /></a>
  <a href="https://opensource.org/license/mit/"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="MIT License" /></a>
</p>

<p align="center">
  <em>
    PyVista is an open source, community-owned project, MIT licensed and
    <a href="https://numfocus.org/sponsored-projects/affiliated-projects">NumFOCUS Affiliated</a>.
  </em>
</p>

<p align="center">
  <em>
    <a href="https://codimensional.com"><strong>CoDimensional PBC</strong></a>,
    founded by PyVista maintainers, is the project's commercial steward.
  </em>
</p>

PyVista provides:

- a NumPy-native API for 3D visualization and mesh analysis
- dataset structures and filters for points, surfaces, and volumes
- one plotting framework for notebooks, scripts, CI, and apps
- a streamlined 3D interface for newcomers and graphics experts alike

![PyVista IPython demo](https://github.com/pyvista/pyvista/raw/main/assets/pyvista_ipython_demo.gif)

## Why PyVista

PyVista is the foundational Python library for 3D visualization and mesh
analysis in scientific computing and engineering. It plays the same role for
3D data that pandas plays for tabular data and xarray plays for labeled
n-dimensional arrays: NumPy-native datasets for point clouds, surfaces, and
volumetric meshes; a filter API covering clipping, slicing, thresholding,
smoothing, and dozens of other operations; and a unified plotting framework
that runs interactively in Jupyter notebooks, headlessly in CI, and as
embedded views inside larger web and desktop applications.

### Built for production

PyVista is the reliable layer between user code and the underlying graphics
stack. The library is image-regression tested on every commit across all
Python versions still in their lifecycle and [VTK](https://vtk.org) releases,
holds its public API stable through a deliberate deprecation lifecycle, and
locks rendering behavior under visual regression baselines. The C++ toolkit
underneath provides few of these assurances and doesn't share our enthusiasm
for testing and reliability, which is why downstream science and engineering
teams build on PyVista.

### Built to extend

Your downstream code can build on PyVista through a small, lazily evaluated
extension API. Third-party packages attach domain-specific filters and
plotter components via registered accessors, with no subclassing, no
monkey-patching, and no vendoring of upstream algorithms. See
[Extending PyVista](https://docs.pyvista.org/extras/extending_pyvista) for
the contract.

## Quickstart

PyVista runs on Python 3.10+:

```bash
pip install pyvista
```

Or via conda-forge:

```bash
conda install -c conda-forge pyvista
```

Try PyVista in your browser without installing anything, on
[MyBinder](https://mybinder.org/v2/gh/pyvista/pyvista-examples/master).

## Command line interface

PyVista also installs a `pyvista` CLI for quick plotting, format conversion,
and mesh validation, without writing any Python:

```bash
# Plot a mesh file in an interactive window
pyvista plot bunny.stl

# Convert a mesh file to another format
pyvista convert bunny.stl .vtp

# Validate a mesh's data, points, and cells
pyvista validate bunny.stl
```

See the [CLI reference](https://docs.pyvista.org/api/cli) for the full
set of commands and options.

## Documentation

- [Getting started](https://docs.pyvista.org/getting-started/)
- [User guide](https://docs.pyvista.org/user-guide/)
- [Examples gallery](http://docs.pyvista.org/examples/index.html)
- [API reference](https://docs.pyvista.org/api/)
- [Installation](http://docs.pyvista.org/getting-started/installation.html#install-ref.)
  (including optional dependencies)

For general questions, ideas, or to share what you've built with PyVista,
start a thread in
[GitHub Discussions](https://github.com/pyvista/pyvista/discussions) or join
the [Slack community](https://communityinviter.com/apps/pyvista/pyvista).

## Connections

PyVista is used across science and engineering disciplines to visualize 3D
data and models, generate publication-quality figures, automate analysis
workflows, and build custom applications on top of PyVista's 3D capabilities.

- [awesome-pyvista](https://github.com/pyvista/awesome-pyvista): a
  continuously updated list of domain-specific tooling that interoperates
  with or is built on PyVista.
- [Connections page](https://docs.pyvista.org/getting-started/connections.html):
  selected highlights and context on how PyVista is used across the community.

## Contributing

[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-3.0-4baaaa.svg)](CODE_OF_CONDUCT.md)
[![Code Triage](https://www.codetriage.com/pyvista/pyvista/badges/users.svg)](https://www.codetriage.com/pyvista/pyvista)
[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/pyvista/pyvista)

PyVista is mostly maintained on a volunteer basis and we welcome contributions
of every shape. Bug reports, documentation fixes, new examples, filter ideas;
all of it helps. Start with the
[Contributing Guide](https://github.com/pyvista/pyvista/blob/main/CONTRIBUTING.rst)
and our
[Code of Conduct](https://github.com/pyvista/pyvista/blob/main/CODE_OF_CONDUCT.md).

## Authors

[![contrib.rocks](https://contrib.rocks/image?repo=pyvista/pyvista)](https://github.com/pyvista/pyvista/graphs/contributors)

PyVista is built by a global community. See the
[contributors page](https://github.com/pyvista/pyvista/graphs/contributors/)
and the active
[list of authors](https://docs.pyvista.org/getting-started/authors.html#authors).
Made with [contrib rocks](https://contrib.rocks).

## Professional support

Many users and organizations rely on PyVista in production workflows,
research pipelines, and custom visualization systems. For expert guidance,
development help, or guaranteed support, there are several ways to engage
with the people who build and maintain PyVista.

For general inquiries, reach out to <info@pyvista.org> and we can help
connect you with the right community experts for your 3D visualization or
analysis needs.

For professional services such as consulting, custom development, feature
design, integration support, or training, consider sponsoring PyVista's core
developers through the "Sponsor this project" section on GitHub. Sponsorship
provides direct access to experts and helps sustain the maintenance and
feature work that keeps PyVista reliable. More details in the discussion
post: <https://github.com/pyvista/pyvista/discussions/4033>.

## Citing PyVista

If you use PyVista in scientific research, please cite the
[JOSS paper](https://doi.org/10.21105/joss.01450).

> Sullivan and Kaszynski (2019). PyVista: 3D plotting and mesh analysis
> through a streamlined interface for the Visualization Toolkit (VTK).
> _Journal of Open Source Software_, 4(37), 1450.
> <https://doi.org/10.21105/joss.01450>

```bibtex
@article{sullivan2019pyvista,
  doi = {10.21105/joss.01450},
  url = {https://doi.org/10.21105/joss.01450},
  year = {2019},
  month = {May},
  publisher = {The Open Journal},
  volume = {4},
  number = {37},
  pages = {1450},
  author = {Bane Sullivan and Alexander Kaszynski},
  title = {{PyVista}: {3D} plotting and mesh analysis through a streamlined interface for the {Visualization Toolkit} ({VTK})},
  journal = {Journal of Open Source Software}
}
```

## Status

**Deployment:**
[![PyPI](https://img.shields.io/pypi/v/pyvista.svg?logo=python&logoColor=white)](https://pypi.org/project/pyvista/)
[![Python versions](https://img.shields.io/pypi/pyversions/pyvista.svg?color=orange&logo=python&label=python&logoColor=white)](https://pypi.org/project/pyvista)
[![Conda](https://img.shields.io/conda/vn/conda-forge/pyvista.svg?logo=conda-forge&logoColor=white)](https://anaconda.org/conda-forge/pyvista)
[![nix](https://img.shields.io/badge/nix-unstable-blue.svg?logo=nixos&logoColor=white)](https://search.nixos.org/packages?channel=unstable&show=python3Packages.pyvista&query=pyvista)
[![Packaging status](https://repology.org/badge/tiny-repos/python:pyvista.svg)](https://repology.org/project/python:pyvista/versions)

**Build:**
[![CI](https://github.com/pyvista/pyvista/actions/workflows/testing-and-deployment.yml/badge.svg)](https://github.com/pyvista/pyvista/actions/workflows/testing-and-deployment.yml)
[![python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/pyvista/pyvista/main.svg)](https://results.pre-commit.ci/latest/github/pyvista/pyvista/main)

**Quality:**
[![codacy](https://app.codacy.com/project/badge/Grade/779ac6aed37548839384acfc0c1aab44)](https://app.codacy.com/gh/pyvista/pyvista/dashboard)
[![codecov](https://codecov.io/gh/pyvista/pyvista/branch/main/graph/badge.svg)](https://app.codecov.io/gh/pyvista/pyvista)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![code style: prettier](https://img.shields.io/badge/code_style-prettier-ff69b4.svg?style=flat)](https://github.com/prettier/prettier)

**Activity:**
[![PyPI downloads](https://img.shields.io/pypi/dm/pyvista.svg?label=PyPI%20downloads)](https://pypi.org/project/pyvista/)
[![Conda downloads](https://img.shields.io/conda/dn/conda-forge/pyvista.svg?label=Conda%20downloads)](https://anaconda.org/conda-forge/pyvista)
[![GitHub Repo stars](https://img.shields.io/github/stars/pyvista/pyvista)](https://github.com/pyvista/pyvista/stargazers)
[![Good first issue](https://img.shields.io/github/issues/pyvista/pyvista/good%20first%20issue)](https://github.com/pyvista/pyvista/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)

**Citation:**
[![JOSS](http://joss.theoj.org/papers/10.21105/joss.01450/status.svg)](https://doi.org/10.21105/joss.01450)
[![Zenodo](https://zenodo.org/badge/DOI/10.5281/zenodo.8415866.svg)](https://zenodo.org/records/8415866)

**Community:**
[![Slack](https://img.shields.io/badge/Slack-pyvista-green.svg?logo=slack)](https://communityinviter.com/apps/pyvista/pyvista)
[![Discussions](https://img.shields.io/badge/GitHub-Discussions-green?logo=github)](https://github.com/pyvista/pyvista/discussions)

**Affiliations & mentions:**
[![NumFOCUS Affiliated](https://img.shields.io/badge/affiliated-NumFOCUS-orange.svg?style=flat&colorA=E1523D&colorB=007D8A)](https://numfocus.org/sponsored-projects/affiliated-projects)
[![Awesome Scientific Computing](https://awesome.re/mentioned-badge.svg)](https://github.com/nschloe/awesome-scientific-computing)
