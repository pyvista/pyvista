### Pyvista within a Docker Container
You can use ``pyvista`` from within a docker container with jupyterlab.
To create a local docker image install ``docker`` and be sure you've logged into docker by following the directions at [Configuring Docker for use with GitHub Packages](https://docs.github.com/en/free-pro-team@latest/packages/using-github-packages-with-your-projects-ecosystem/configuring-docker-for-use-with-github-packages#authenticating-with-a-personal-access-token)

Next, pull, and run the image with:

```bash
docker pull docker.pkg.github.com/pyvista/pyvista/pyvista-jupyterlab:v0.27.0
docker run -p 8888:8888 docker.pkg.github.com/pyvista/pyvista/pyvista-jupyterlab:v0.27.0
```

Finally, open the link that shows up and start playing around with
pyvista in jupyterlab!  For example:

```
http://127.0.0.1:8888/?token=XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
```

### Create your own Docker Container with `pyvista`

Clone pyvista and cd into this directory to create your own customized docker image.

```bash
IMAGE=my-pyvista-jupyterlab:v0.1.0
docker build -t $IMAGE .
docker push $IMAGE
```
