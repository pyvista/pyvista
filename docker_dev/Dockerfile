FROM jupyter/base-notebook:lab-2.2.8
LABEL maintainer="pyvista Developers"

USER root
RUN apt-get update \
 && apt-get install  -yq --no-install-recommends \
    libgl1-mesa-glx \
    libglu1-mesa \
    libsm6 \
    libopengl0 \
    libegl1 \
 && apt-get clean && rm -rf /var/lib/apt/lists/*
USER jovyan

# setup the itkwidgets conda environment
COPY environment.yml labextensions.txt /tmp/
RUN conda env update --name base --file /tmp/environment.yml
RUN conda run conda install -y nodejs
RUN conda run jupyter labextension install $(cat /tmp/labextensions.txt)

RUN pip install ipyvtklink==0.2.1

# COPY vtk-9.0.20201105-cp38-cp38-linux_x86_64.whl /tmp/
# RUN pip install /tmp/vtk-9.0.20201105-cp38-cp38-linux_x86_64.whl

COPY pyvista-0.27.dev1-py3-none-any.whl /tmp/
RUN pip install /tmp/pyvista-0.27.dev1-py3-none-any.whl

USER root
RUN sudo apt update && sudo apt-get install xvfb -y

COPY examples examples
RUN chown jovyan -R examples
RUN rm -rf /tmp/* && rm -rf work
USER jovyan

ENV DISPLAY=:99.0
ENV JUPYTER_ENABLE_LAB=yes
ENV PYVISTA_USE_IPYVTK=true

# modify the CMD and start a background server first
CMD /bin/bash -c "Xvfb :99 -screen 0 1024x768x24 > /dev/null 2>&1 &" && start-notebook.sh
