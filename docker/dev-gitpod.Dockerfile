FROM gitpod/workspace-full
LABEL maintainer="PyVista Developers"
LABEL repo="https://github.com/pyvista/pyvista"

RUN sudo apt-get install  -yq --no-install-recommends \
    libosmesa6

ENV PYTHONUSERBASE=/workspace/.pip-modules
ENV PATH=$PYTHONUSERBASE/bin:$PATH

WORKDIR $HOME
ENV PYVISTA_OFF_SCREEN=true
