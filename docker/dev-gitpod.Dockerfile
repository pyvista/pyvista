FROM gitpod/workspace-full-vnc
LABEL maintainer="PyVista Developers"
LABEL repo="https://github.com/pyvista/pyvista"

RUN sudo apt-get update \
  && sudo apt-get install  -yq --no-install-recommends libxrender1

RUN echo "[ ! -d /workspace/pv-venv ] && python -m venv /workspace/pv-venv" > $HOME/.bashrc.d/999-pyvista
RUN echo "source /workspace/pv-venv/bin/activate" >> $HOME/.bashrc.d/999-pyvista

ENV PYVISTA_OFF_SCREEN=false
ENV PRE_COMMIT_HOME=/workspace/.precommit
