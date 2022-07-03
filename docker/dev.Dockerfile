FROM gitpod/workspace-full:2022-05-08-14-31-53
LABEL maintainer="PyVista Developers"
LABEL repo="https://github.com/pyvista/pyvista"

RUN apt-get install  -yq --no-install-recommends \
    libosmesa6

RUN pip install --no-cache-dir https://github.com/pyvista/pyvista-wheels/raw/main/vtk-osmesa-9.1.0-cp39-cp39-linux_x86_64.whl

WORKDIR $HOME
ENV PYVISTA_OFF_SCREEN=true
