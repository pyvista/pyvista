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

RUN pip install --no-cache-dir pyvista*.tar.gz
# Install our custom vtk wheel
RUN pip uninstall vtk -y
RUN pip install --no-cache-dir https://github.com/pyvista/pyvista-wheels/raw/main/vtk_osmesa-9.2.5-cp39-cp39-linux_x86_64.whl

WORKDIR $HOME
ENV PYVISTA_OFF_SCREEN=true
