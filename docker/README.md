## PyVista Docker Images

Prebuilt images are published to the GitHub Container Registry for every
release and every push to `main`:

| Tag                                   | Base                                        | Purpose                                     |
| ------------------------------------- | ------------------------------------------- | ------------------------------------------- |
| `ghcr.io/pyvista/pyvista:latest`      | `quay.io/jupyter/base-notebook:python-3.13` | PyVista + JupyterLab + Trame (interactive). |
| `ghcr.io/pyvista/pyvista:latest-slim` | `python:3.13-slim`                          | PyVista only, off-screen rendering.         |

Pull and run JupyterLab:

```bash
docker pull ghcr.io/pyvista/pyvista:latest
docker run --rm -p 8888:8888 ghcr.io/pyvista/pyvista:latest
```

Open the URL printed in the terminal to start using PyVista in JupyterLab.

See [Working with the GitHub Container registry](https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry)
if you need to authenticate.

## Building locally

Both images are produced from a single multi-stage
[`Dockerfile`](./Dockerfile). The wheel is built inside Docker, so there are
no pre-build steps — just clone and run `docker build` from the repository
root:

```bash
git clone https://github.com/pyvista/pyvista
cd pyvista

# JupyterLab image
docker build -f docker/Dockerfile --target jupyter -t pyvista:jupyter .

# Slim (off-screen) image
docker build -f docker/Dockerfile --target slim -t pyvista:slim .
```

Override the Python version via build arg (must match a supported
[VTK wheel](https://pypi.org/project/vtk/)):

```bash
docker build --build-arg PY_VERSION=3.12 \
  -f docker/Dockerfile --target jupyter -t pyvista:jupyter .
```

The `jupyter` target pulls extras (`jupyter`, `colormaps`, `io`) directly
from `pyproject.toml`, so package versions always track the project's pins.

## Gitpod

[`dev-gitpod.Dockerfile`](./dev-gitpod.Dockerfile) is a separate development
image used by the [Gitpod](https://www.gitpod.io/) workspace configuration.
It is unrelated to the published runtime images above.
