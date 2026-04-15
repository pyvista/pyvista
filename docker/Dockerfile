# syntax=docker/dockerfile:1.7
#
# Multi-stage Dockerfile for PyVista.
#
# Targets:
#   - slim    : Minimal Python image with PyVista installed (off-screen).
#   - jupyter : JupyterLab image with PyVista + Trame for interactive
#               in-notebook rendering.
#
# Build from the repository root:
#
#   docker build -f docker/Dockerfile --target slim    -t pyvista:slim    .
#   docker build -f docker/Dockerfile --target jupyter -t pyvista:jupyter .
#
# Override the Python version (must match a supported VTK wheel):
#
#   docker build --build-arg PY_VERSION=3.12 \
#     -f docker/Dockerfile --target jupyter -t pyvista:jupyter .

ARG PY_VERSION=3.13


# ---------------------------------------------------------------------------
# Stage 1: build the PyVista wheel from the source tree in the build context.
# Produces /build/dist/pyvista-*.whl, which the runtime stages install. This
# keeps build tooling out of the final images and avoids any pre-build step
# outside of Docker.
# ---------------------------------------------------------------------------
FROM python:${PY_VERSION}-slim AS builder

WORKDIR /build

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip build

COPY . /build

RUN python -m build --wheel --outdir /build/dist


# ---------------------------------------------------------------------------
# Stage 2: slim runtime. PyVista only, no Jupyter, off-screen by default.
# ---------------------------------------------------------------------------
FROM python:${PY_VERSION}-slim AS slim

LABEL org.opencontainers.image.title="PyVista (slim)"
LABEL org.opencontainers.image.description="Minimal PyVista runtime image for off-screen rendering."
LABEL org.opencontainers.image.source="https://github.com/pyvista/pyvista"
LABEL org.opencontainers.image.licenses="MIT"

ENV PYVISTA_OFF_SCREEN=true \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=graphics,utility,compute

# Runtime libraries VTK dlopens for off-screen rendering. libegl1 enables
# EGL so the image uses GPU acceleration when run with the NVIDIA container
# runtime, and falls back to Mesa llvmpipe software rendering on CPU-only
# hosts via the libgl1/GLX path.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      libopengl0 \
      libgl1 \
      libegl1 \
      libxrender1 \
 && rm -rf /var/lib/apt/lists/*

COPY --from=builder /build/dist/pyvista-*.whl /tmp/

RUN pip install /tmp/pyvista-*.whl \
 && rm /tmp/pyvista-*.whl

WORKDIR /work


# ---------------------------------------------------------------------------
# Stage 3: jupyter runtime. PyVista with Jupyter/Trame extras, built on the
# upstream jupyter/base-notebook image. The optional dependency groups come
# directly from pyproject.toml so there is a single source of truth.
#
# The jupyter base image only publishes a single current Python tag at
# quay.io/jupyter/base-notebook, so this stage is pinned to python-3.13
# regardless of the PY_VERSION build arg (which still applies to the
# builder and slim stages).
# ---------------------------------------------------------------------------
FROM quay.io/jupyter/base-notebook:python-3.13 AS jupyter

LABEL org.opencontainers.image.title="PyVista (jupyter)"
LABEL org.opencontainers.image.description="PyVista with JupyterLab and Trame for interactive rendering."
LABEL org.opencontainers.image.source="https://github.com/pyvista/pyvista"
LABEL org.opencontainers.image.licenses="MIT"

ENV JUPYTER_ENABLE_LAB=yes \
    PYVISTA_TRAME_SERVER_PROXY_PREFIX=/proxy/ \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=graphics,utility,compute

USER root

# Runtime libraries VTK dlopens for off-screen rendering. libegl1 enables
# EGL so the image uses GPU acceleration when run with the NVIDIA container
# runtime, and falls back to Mesa llvmpipe software rendering on CPU-only
# hosts via the libgl1/GLX path. The upstream jupyter image does not ship
# these.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      libopengl0 \
      libgl1 \
      libegl1 \
      libxrender1 \
 && rm -rf /var/lib/apt/lists/*

COPY --from=builder /build/dist/pyvista-*.whl /tmp/

RUN pip install "$(ls /tmp/pyvista-*.whl)[jupyter,colormaps,io]" \
 && rm /tmp/pyvista-*.whl

USER ${NB_UID}
WORKDIR ${HOME}

EXPOSE 8888
