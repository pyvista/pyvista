FROM gitpod/workspace-full
LABEL maintainer="PyVista Developers"
LABEL repo="https://github.com/pyvista/pyvista"

RUN sudo apt-get install  -yq --no-install-recommends \
    libosmesa6

RUN pip install --no-cache-dir https://github.com/pyvista/pyvista-wheels/raw/main/vtk-osmesa-9.1.0-cp38-cp38-linux_x86_64.whl

COPY  requirements_test.txt requirements_test.txt
COPY requirements_docs.txt requirements_docs.txt

RUN pip install --no-cache-dir -r requirements_test.txt
RUN pip install --no-cache-dir -r requirements_docs.txt
RUN pip install --no-cache-dir pre-commit

RUN pre-commit install --install-hooks

WORKDIR $HOME
ENV PYVISTA_OFF_SCREEN=true
