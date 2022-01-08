FROM jupyter/base-notebook:python-3.8.6
LABEL maintainer="PyVista Developers"

USER root
RUN apt-get update \
 && apt-get install  -yq --no-install-recommends \
    libgl1-mesa-glx \
    libglu1-mesa \
    libfontconfig1 \
    libxrender1 \
    xvfb \
 && apt-get clean && rm -rf /var/lib/apt/lists/*
USER jovyan

COPY pyvista/ /build-context/pyvista/
COPY setup.py /build-context/
COPY README.rst /build-context/
COPY LICENSE /build-context/
COPY requirements*.txt /build-context/
WORKDIR /build-context/
RUN pip install -r requirements_docs.txt -r requirements_test.txt
RUN pip install .

WORKDIR $HOME

# allow jupyterlab for ipyvtk
ENV DISPLAY=:99.0
ENV JUPYTER_ENABLE_LAB=yes
ENV PYVISTA_USE_IPYVTK=true

# modify the CMD to start a background Xvfb process first
CMD /bin/bash -c "Xvfb :99 -screen 0 1024x768x24 > /dev/null 2>&1 &" && start-notebook.sh
