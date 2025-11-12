FROM jupyter/base-notebook:python-3.11
LABEL maintainer="PyVista Developers"
LABEL repo="https://github.com/pyvista/pyvista"

COPY dist/*.tar.gz /build-context/
COPY README.rst /build-context/
COPY LICENSE /build-context/
COPY docker/requirements.txt /build-context/requirements.txt
WORKDIR /build-context/

RUN pip install --no-cache-dir pyvista*.tar.gz
RUN pip install -r requirements.txt

WORKDIR $HOME

# allow jupyterlab for ipyvtk
ENV JUPYTER_ENABLE_LAB=yes
ENV PYVISTA_TRAME_SERVER_PROXY_PREFIX='/proxy/'
