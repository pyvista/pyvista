ARG BASE_IMAGE=ghcr.io/pyvista/pyvista
FROM $BASE_IMAGE

COPY requirements_docs.txt /build-context/requirements_docs.txt

RUN pip install -r /build-context/requirements_docs.txt

# TODO: convert to Jupyter Notebooks
COPY examples/ $HOME/examples/
