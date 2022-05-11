"""Simple flask app to display static images generated from pyvista.

Expected paths:
static_ex/
└── app.py
    templates/
    └── index.html

"""
import os

from flask import Flask, render_template, request

import pyvista

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
    if meshtype == 'Sphere':
        mesh = pyvista.Sphere()
    elif meshtype == 'Cube':
        mesh = pyvista.Cube()
    elif meshtype == 'Bohemian Dome':
        mesh = pyvista.ParametricBohemianDome()
    elif meshtype == 'Cylinder':
        mesh = pyvista.Cylinder()
    else:
        # invalid entry
        raise ValueError('Invalid Option')

    # generate screenshot
    filename = f'{meshtype}.png'
    filepath = os.path.join(static_image_path, filename)
    mesh.plot(off_screen=True, window_size=(300, 300), screenshot=filepath)
    return os.path.join('images', filename)


if __name__ == '__main__':
    app.run()
