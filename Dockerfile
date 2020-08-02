FROM ubuntu:latest
MAINTAINER info@pyvista.org.

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG C.UTF-8
ENV TERM xterm
ENV TAG_NAME=master

USER root
RUN apt-get install python-qt4 libgl1-mesa-glx
RUN apt-get install -y xvfb
ENV PYVISTA_VIRTUAL_DISPLAY True
ENV PYVISTA_OFF_SCREEN=True
RUN apt-get install -y --no-install-recommends python3-pip
RUN pip install --no-cache --upgrade pip
RUN pip install git+https://github.com/pyvista/pyvista
