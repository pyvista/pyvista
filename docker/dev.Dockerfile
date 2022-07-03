FROM python:3.9-slim
LABEL maintainer="PyVista Developers"
LABEL repo="https://github.com/pyvista/pyvista"

RUN apt-get update \
 && apt-get install  -yq --no-install-recommends \
    libgl1-mesa-glx xvfb

WORKDIR $HOME
ENV PYVISTA_OFF_SCREEN=true
