FROM python:3.11-slim
LABEL maintainer="PyVista Developers"
LABEL repo="https://github.com/pyvista/pyvista"

COPY dist/*.tar.gz /build-context/
COPY README.rst /build-context/
COPY LICENSE /build-context/
WORKDIR /build-context/

RUN pip install --no-cache-dir pyvista*.tar.gz

WORKDIR $HOME
ENV PYVISTA_OFF_SCREEN=true
