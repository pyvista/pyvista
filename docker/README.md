### PyVista within a Docker Container

You can use PyVista from within a docker container with Jupyter Lab.

To create a local docker image install ``docker`` and be sure you've logged into docker by following the directions at [Configuring Docker for use with GitHub Packages](https://docs.github.com/en/free-pro-team@latest/packages/using-github-packages-with-your-projects-ecosystem/configuring-docker-for-use-with-github-packages#authenticating-with-a-personal-access-token)

Next, pull, and run the image with:

```bash
docker pull ghcr.io/pyvista/pyvista:latest
docker run -p 8888:8888 ghcr.io/pyvista/pyvista:latest
```

Finally, open the link that shows up and start playing around with
PyVista in Jupyter Lab.


### Build PyVista Docker Image Locally

Clone PyVista and run the following at the top level of the project:

```bash
pip install build
python -m build --sdist
docker build -t my-pyvista-jupyterlab -f docker/jupyter.Dockerfile .
```
