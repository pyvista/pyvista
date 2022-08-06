"""A helper script to generate the external examples gallery."""
from io import StringIO
import os


def format_icon(title, description, link, image):
    body = r"""

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="{}">

.. only:: html

    .. figure:: ../images/external-examples/{}
       :target: {}

       {}




.. raw:: html

    </div>


.. toctree::
   :hidden:

   {} <{}>



"""
    content = body.format(description, image, link, title, title, link)
    return content


class Example:
    def __init__(self, title, description, link, image):
        self.title = title
        self.description = description
        self.link = link
        self.image = image

    def format(self):
        return format_icon(self.title, self.description, self.link, self.image)


###############################################################################

articles = dict(
    omf=Example(
        title="3D visualization for the Open Mining Format (omf)",
        description="3D visualization for the Open Mining Format (omf)",
        link="https://opengeovis.github.io/omfvista/examples/index.html",
        image="omfvista.png",
    ),
    discretize=Example(
        title="3D Rendering with Discretize",
        description="3D Rendering with Discretize",
        link="http://discretize.simpeg.xyz/en/main/examples/plot_pyvista_laguna.html",
        image="discretize.png",
    ),
    open_foam=Example(
        title="OpenFOAM Rendering",
        description="OpenFOAM Rendering",
        link="https://pswpswpsw.github.io/posts/2018/09/blog-post-modify-vtk-openfoam/",
        image="open-foam.png",
    ),
    aero_sandbox=Example(
        title="AeroSandbox",
        description="AeroSandbox",
        link="https://peterdsharpe.github.io/AeroSandbox/",
        image="AeroSandbox.png",
    ),
    forge=Example(
        title="FORGE Geothermal Project",
        description="FORGE Geothermal Project",
        link="https://forge.pvgeo.org/project/index.html",
        image="forge.png",
    ),
    pvgeo=Example(
        title="PVGeo's example gallery",
        description="PVGeo's example gallery",
        link="https://pvgeo.org/examples/index.html",
        image="pvgeo.png",
    ),
    tetgen=Example(
        title="TetGen's example gallery",
        description="TetGen's example gallery",
        link="http://tetgen.pyvista.org/examples/index.html",
        image="tetgen.png",
    ),
    mesh_fix=Example(
        title="PyMeshFix's example gallery",
        description="PyMeshFix's example gallery",
        link="http://pymeshfix.pyvista.org/examples/index.html",
        image="pymeshfix.png",
    ),
    orvisu=Example(
        title="Orvisu Demo Application",
        description="Orvisu Demo Application",
        link="https://github.com/BartheG/Orvisu",
        image="orvisu.gif",
    ),
    flem=Example(
        title="FLEM: A diffusive landscape evolution model",
        description="FLEM: A diffusive landscape evolution model",
        link="https://github.com/johnjarmitage/flem",
        image="flem.png",
    ),
    optimization=Example(
        title="Optimization visualization with PyVista",
        description="Optimization visualization with PyVista",
        link="https://gist.github.com/hichamjanati/6668d91848283c31ac18d801552fb582",
        image="optimization.gif",
    ),
    anvil_cirrus_plumes=Example(
        title="Anvil Cirrus Plumes",
        description="Dr. Morgan O'Neill at Stanford University is researching Above Anvil Cirrus Plumes and its dynamics as a hydraulic jump.",
        link="https://www.youtube.com/watch?v=cCPjnF_vHxw&feature=youtu.be",
        image="anvil_cirrus_plumes.png",
    ),
    damavand=Example(
        title="Damavand Volcano",
        description="Visualize 3D models of Damavand Volcano, Alborz, Iran.",
        link="https://nbviewer.jupyter.org/github/banesullivan/damavand-volcano/blob/master/Damavand_Volcano.ipynb",
        image="damavand_volcano.gif",
    ),
    atmos_conv=Example(
        title="Atmospheric Convection",
        description="Atmospheric convection plays a key role in the climate of tidally-locked terrestrial exoplanets: insights from high-resolution simulations",
        link="https://dennissergeev.github.io/exoconvection-apj-2020/",
        image="atmospheric_convection.jpeg",
    ),
    vessel_vio=Example(
        title="VesselVio",
        description="An open-source application for vasculature dataset analysis and visualization",
        link="https://jacobbumgarner.github.io/VesselVio/",
        image="vessel_vio.png",
    ),
    pyvista_artworks=Example(
        title="Stéphane Laurent's artwork",
        description="Stéphane Laurent's blog showing a sample of the animations they realized with PyVista.",
        link="https://laustep.github.io/stlahblog/posts/MyPyVistaArtworks.html",
        image="Duoprism_3-30.gif",
    ),
    ptera_software=Example(
        title="PteraSoftware",
        description="Ptera Software is a fast, easy-to-use, and open-source software package for analyzing flapping-wing flight.",
        link="https://github.com/camurban/pterasoftware",
        image="ptera_software.gif",
    ),
    geemap=Example(
        title="geemap",
        description="A Python package for interactive mapping with Google Earth Engine, ipyleaflet, and ipywidgets.",
        link="https://geemap.org/",
        image="geemap.gif",
    ),
    gmshmodel=Example(
        title="GmshModel",
        description="A mesh modeling interface to the Gmsh-Python-API",
        link="https://gmshmodel.readthedocs.io/en/latest/",
        image="gmsh_model.png",
    ),
    grad_descent_visualizer=Example(
        title="Gradient Descent Visualizer",
        description="A Python package used to visualize the gradient descent of function landscapes.",
        link="https://github.com/JacobBumgarner/grad-descent-visualizer",
        image="grad_descent_visualizer.gif",
    ),
    # entry=Example(title="",
    #     description="",
    #     link="",
    #     image=""),
)


###############################################################################


def make_example_gallery():
    """Make the example gallery."""
    path = "./getting-started/external_examples.rst"

    with StringIO() as new_fid:
        new_fid.write(
            """
External Examples
=================

Here are a list of longer, more technical examples of what PyVista can do!

.. caution::

    Please note that these examples link to external websites.
    If any of these links are broken, please raise an issue on the repository.


Do you have a technical processing workflow or visualization routine you
would like to share?
If so, please consider sharing your work here submitting a PR at
https://github.com/pyvista and we would be glad to add it!



"""
        )
        # Reverse to put the latest items at the top
        for example in list(articles.values())[::-1]:
            new_fid.write(example.format())

        new_fid.write(
            """

.. raw:: html

    <div class="sphx-glr-clear"></div>


"""
        )
        new_fid.seek(0)
        new_text = new_fid.read()

    # check if it's necessary to overwrite the table
    existing = ""
    if os.path.exists(path):
        with open(path) as existing_fid:
            existing = existing_fid.read()

    # write if different or does not exist
    if new_text != existing:
        with open(path, "w") as fid:
            fid.write(new_text)

    return
