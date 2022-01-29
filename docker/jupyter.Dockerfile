FROM jupyter/base-notebook:python-3.9.7
LABEL maintainer="PyVista Developers"
LABEL repo="https://github.com/pyvista/pyvista"

USER root
RUN apt-get update \
 && apt-get install  -yq --no-install-recommends \
    libfontconfig1 \
    libxrender1 \
    libosmesa6 \
 && apt-get clean && rm -rf /var/lib/apt/lists/*
USER jovyan

COPY dist/*.tar.gz /build-context/
COPY README.rst /build-context/
COPY LICENSE /build-context/
COPY docker/requirements.txt /build-context/requirements.txt
WORKDIR /build-context/

# Install our custom vtk wheel
RUN pip install https://github.com/pyvista/pyvista-wheels/raw/main/vtk-osmesa-9.1.0-cp39-cp39-linux_x86_64.whl

RUN pip install pyvista*.tar.gz
RUN pip install -r requirements.txt

WORKDIR $HOME

# allow jupyterlab for ipyvtk
ENV JUPYTER_ENABLE_LAB=yes
ENV PYVISTA_USE_IPYVTK=true
