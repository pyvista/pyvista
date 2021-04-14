FROM jupyter/base-notebook:python-3.8.6
LABEL maintainer="pyvista Developers"

USER root
RUN apt-get update \
 && apt-get install  -yq --no-install-recommends \
    libgl1-mesa-glx \
    libglu1-mesa \
    xvfb \
 && apt-get clean && rm -rf /var/lib/apt/lists/*
USER jovyan

# setup the itkwidgets conda environment
COPY environment.yml labextensions.txt /tmp/
RUN conda env update --name base --file /tmp/environment.yml
RUN conda run conda install -y nodejs
RUN conda run jupyter labextension install $(cat /tmp/labextensions.txt)

RUN pip install ipyvtklink==0.2.1

# allow jupyterlab for ipyvtk
ENV DISPLAY=:99.0
ENV JUPYTER_ENABLE_LAB=yes
ENV PYVISTA_USE_IPYVTK=true

# modify the CMD and start a background server first
CMD /bin/bash -c "Xvfb :99 -screen 0 1024x768x24 > /dev/null 2>&1 &" && start-notebook.sh
