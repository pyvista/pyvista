## Developer Docker Build

Build file and Dockerfile for creating the docker images hosted at

```
docker.pkg.github.com/pyvista/pyvista/pyvista-jupyterlab:v0.27.0
```

Since this image was pushed before ``0.27.0`` was out, the
``Dockerfile`` used ``0.27.dev1``.  Additionally, there are some
features left over from building with EGL GPU offscreen support, but I
dropped it as it seems that EGL has issues with software rendering.
