"""Simple flask app to display dynamic threejs generated from pyvista.

Expected paths:
dynamic_ex/
└── app.py
    templates/
    └── index.html

"""
import os

from flask import Flask, render_template, request
import numpy as np

import pyvista
from pyvista import examples

static_image_path = os.path.join('static', 'images')
if not os.path.isdir(static_image_path):
    os.makedirs(static_image_path)


app = Flask(__name__)


@app.route("/")
def index():
    return render_template('index.html')


@app.route("/getimage")
def get_img():
    """Generate a screenshot of a simple pyvista mesh.

    Returns
    -------
    str
        Local path within the static directory of the image.

    """
    # get the user selected mesh option
    meshtype = request.args.get('meshtype')
    color = request.args.get('color')
    style = request.args.get('style')
    print(f"'{style}'")

    # bool types
    show_edges = request.args.get('show_edges') == 'true'
    lighting = request.args.get('lighting') == 'true'
    pbr = request.args.get('pbr') == 'true'
    anti_aliasing = request.args.get('anti_aliasing') == 'true'

    if meshtype == 'Sphere':
        mesh = pyvista.Sphere()
    elif meshtype == 'Cube':
        mesh = pyvista.Cube()
    elif meshtype == 'Bohemian Dome':
        mesh = pyvista.ParametricBohemianDome()
    elif meshtype == 'Cylinder':
        mesh = pyvista.Cylinder()
    elif meshtype == 'Vectors':
        n_points = 20
        points = np.random.random((n_points, 3))
        poly = pyvista.PolyData(points)
        poly['direction'] = np.random.random((n_points, 3))
        poly['direction'] -= poly['direction'].mean(axis=0)  # normalize
        mesh = poly.glyph(geom=pyvista.Arrow(), orient=True, scale=True, factor=0.2)
        # reset color as we will want to see the colors of the arrows
        color = None

    elif meshtype == 'Queen Nefertiti':
        mesh = examples.download_nefertiti()
    elif meshtype == 'Lidar':
        mesh = examples.download_lidar()
    else:
        # invalid entry
        raise ValueError('Invalid Option')

    # generate screenshot
    filename = 'mesh.html'
    filepath = os.path.join(static_image_path, filename)

    # create a plotter and add the mesh to it
    pl = pyvista.Plotter(window_size=(600, 600))
    pl.add_mesh(
        mesh,
        style=style,
        lighting=lighting,
        pbr=pbr,
        metallic=0.8,
        split_sharp_edges=True,
        show_edges=show_edges,
        color=color,
    )
    if anti_aliasing:
        pl.enable_anti_aliasing()
    pl.background_color = 'white'
    pl.export_html(filepath)
    return os.path.join('images', filename)


if __name__ == '__main__':
    app.run()
