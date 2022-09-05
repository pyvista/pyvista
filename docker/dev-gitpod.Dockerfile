FROM gitpod/workspace-full
LABEL maintainer="PyVista Developers"
LABEL repo="https://github.com/pyvista/pyvista"

RUN sudo apt-get install  -yq --no-install-recommends \
    libosmesa6

RUN echo "[ ! -d /workspace/venv ] && python -m venv /workspace/venv" > $HOME/.bashrc.d/999-pyvista
RUN echo "source /workspace/venv/bin/activate" >> $HOME/.bashrc.d/999-pyvista

WORKDIR $HOME
ENV PYVISTA_OFF_SCREEN=true
ENV PRE_COMMIT_HOME=/workspace/.precommit
