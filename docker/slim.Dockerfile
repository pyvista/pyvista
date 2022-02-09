FROM python:3.9-slim
LABEL maintainer="PyVista Developers"
LABEL repo="https://github.com/pyvista/pyvista"

RUN apt-get update \
 && apt-get install  -yq --no-install-recommends \
    libosmesa6 \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY dist/*.tar.gz /build-context/
COPY README.rst /build-context/
COPY LICENSE /build-context/
COPY docker/requirements.txt /build-context/requirements.txt
WORKDIR /build-context/

# Install our custom vtk wheel
RUN pip install --no-cache-dir https://github.com/pyvista/pyvista-wheels/raw/main/vtk-osmesa-9.1.0-cp39-cp39-linux_x86_64.whl

RUN pip install --no-cache-dir pyvista*.tar.gz

WORKDIR $HOME
ENV PYVISTA_OFF_SCREEN=true
