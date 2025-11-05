"""A helper script to generate the external examples gallery."""

from __future__ import annotations

from io import StringIO
from pathlib import Path


def format_icon(title, link, image):  # noqa: D103
    body = r"""
   .. grid-item-card:: {}
      :link: {}
      :text-align: center
      :class-title: pyvista-card-title

      .. image:: ../images/external-examples/{}
"""
    return body.format(title, link, image)


class Example:  # noqa: D101
    def __init__(self, title, link, image):
        self.title = title
        self.link = link
        self.image = image

    def format(self):  # noqa: D102
        return format_icon(self.title, self.link, self.image)


###############################################################################

articles = dict(
    omf=Example(
        title='3D visualization for the Open Mining Format (omf)',
        link='https://opengeovis.github.io/omfvista/examples/index.html',
        image='omfvista.png',
    ),
    discretize=Example(
        title='3D Rendering with Discretize',
        link='http://discretize.simpeg.xyz/en/main/examples/plot_pyvista_laguna.html',
        image='discretize.png',
    ),
    aero_sandbox=Example(
        title='AeroSandbox',
        link='https://peterdsharpe.github.io/AeroSandbox/',
        image='AeroSandbox.png',
    ),
    forge=Example(
        title='FORGE Geothermal Project',
        link='https://forge.pvgeo.org/project/index.html',
        image='forge.png',
    ),
    pvgeo=Example(
        title="PVGeo's example gallery",
        link='https://pvgeo.org/examples/index.html',
        image='pvgeo.png',
    ),
    tetgen=Example(
        title="TetGen's example gallery",
        link='http://tetgen.pyvista.org/examples/index.html',
        image='tetgen.png',
    ),
    mesh_fix=Example(
        title="PyMeshFix's example gallery",
        link='http://pymeshfix.pyvista.org/examples/index.html',
        image='pymeshfix.png',
    ),
    orvisu=Example(
        title='Orvisu Demo Application',
        link='https://github.com/BartheG/Orvisu',
        image='orvisu.gif',
    ),
    flem=Example(
        title='FLEM: A diffusive landscape evolution model',
        link='https://github.com/johnjarmitage/flem',
        image='flem.png',
    ),
    optimization=Example(
        title='Optimization visualization with PyVista',
        link='https://gist.github.com/hichamjanati/6668d91848283c31ac18d801552fb582',
        image='optimization.gif',
    ),
    anvil_cirrus_plumes=Example(
        title='Anvil Cirrus Plumes',
        link='https://www.youtube.com/watch?v=cCPjnF_vHxw&feature=youtu.be',
        image='anvil_cirrus_plumes.png',
    ),
    damavand=Example(
        title='Damavand Volcano',
        link='https://nbviewer.jupyter.org/github/banesullivan/damavand-volcano/blob/master/Damavand_Volcano.ipynb',
        image='damavand_volcano.gif',
    ),
    atmos_conv=Example(
        title='Atmospheric Convection',
        link='https://dennissergeev.github.io/exoconvection-apj-2020/',
        image='atmospheric_convection.jpeg',
    ),
    vessel_vio=Example(
        title='VesselVio',
        link='https://jacobbumgarner.github.io/VesselVio/',
        image='vessel_vio.png',
    ),
    pyvista_artworks=Example(
        title="Stéphane Laurent's artwork",
        link='https://laustep.github.io/stlahblog/posts/MyPyVistaArtworks.html',
        image='Duoprism_3-30.gif',
    ),
    ptera_software=Example(
        title='PteraSoftware',
        link='https://github.com/camurban/pterasoftware',
        image='ptera_software.gif',
    ),
    geemap=Example(
        title='geemap',
        link='https://geemap.org/',
        image='geemap.gif',
    ),
    geovista=Example(
        title='GeoVista',
        link='https://github.com/bjlittle/geovista',
        image='geovista_earth.png',
    ),
    gmshmodel=Example(
        title='GmshModel',
        link='https://gmshmodel.readthedocs.io/en/latest/',
        image='gmsh_model.png',
    ),
    grad_descent_visualizer=Example(
        title='Gradient Descent Visualizer',
        link='https://github.com/JacobBumgarner/grad-descent-visualizer',
        image='grad_descent_visualizer.gif',
    ),
    nikolov1=Example(
        title='Ivan Nikolov on Visualization Libraries',
        link='https://medium.com/towards-data-science/python-libraries-for-mesh-and-point-cloud-visualization-part-1-daa2af36de30',
        image='nikolov1.gif',
    ),
    nikolov2=Example(
        title='Ivan Nikolov on Voxelization',
        link='https://medium.com/towards-data-science/how-to-voxelize-meshes-and-point-clouds-in-python-ca94d403f81d',
        image='nikolov2.gif',
    ),
    nikolov3=Example(
        title='Ivan Nikolov on Neighbourhood Analysis',
        link='https://medium.com/towards-data-science/neighborhood-analysis-kd-trees-and-octrees-for-meshes-and-point-clouds-in-python-19fa96527b77',
        image='nikolov3.gif',
    ),
    magpylib=Example(
        title='Coil Field Lines example in Magpylib',
        link='https://magpylib.readthedocs.io/en/latest/_pages/user_guide/examples/examples_vis_pv_streamlines.html',
        image='coil_field_lines.png',
    ),
    pyfbs=Example(
        title='pyFBS: Frequency Based Substructuring in Python',
        link='https://pyfbs.readthedocs.io/en/master/examples/examples.html',
        image='pyfbs.webp',
    ),
    topogenesis=Example(
        title='topoGenesis',
        link='https://topogenesis.readthedocs.io/notebooks/boolean_marching_cubes/',
        image='boolean_marching_cubes.png',
    ),
    entry=Example(
        title='PyHyperbolic3D',
        link='https://github.com/stla/PyHyperbolic3D/tree/main',
        image='griddip.gif',
    ),
    sunkit=Example(
        title='sunkit-pyvista',
        link='https://docs.sunpy.org/projects/sunkit-pyvista/en/latest/',
        image='sunkit-pyvista.png',
    ),
    gemgis=Example(
        title='GemGIS',
        link='https://gemgis.readthedocs.io/en/latest',
        image='gemgis.png',
    ),
    air_racing_optimization=Example(
        title='Air Racing Trajectory Optimization',
        link='https://github.com/peterdsharpe/air-racing-optimization',
        image='air_racing_optimization.png',
    ),
    felupe=Example(
        title='FElupe',
        link='https://felupe.readthedocs.io/en/latest/',
        image='felupe.png',
    ),
    drilldown=Example(
        title='DrillDown',
        link='https://github.com/cardinalgeo/drilldown',
        image='drilldown.jpg',
    ),
    stpyvista=Example(
        title='stpyvista',
        link='https://github.com/edsaac/stpyvista',
        image='stpyvista_intro_crop.gif',
    ),
    visualpic=Example(
        title='VisualPIC',
        link='https://github.com/AngelFP/VisualPIC',
        image='visualpic.png',
    ),
    pyelastica=Example(
        title='PyElastica',
        link='https://github.com/GazzolaLab/PyElastica',
        image='pyelastica.gif',
    ),
    comet_fenicsx=Example(
        title='Numerical Tours of Computational Mechanics with FEniCSx',
        link='https://bleyerj.github.io/comet-fenicsx',
        image='comet_fenicsx.png',
    ),
    # entry=Example(title="",
    #     link="",
    #     image=""),
)


###############################################################################


def make_example_gallery():
    """Make the example gallery."""
    path = './getting-started/external_examples.rst'

    with StringIO() as new_fid:
        new_fid.write(
            """.. _external_examples:

External Examples
=================

Here are a list of longer, more technical examples of what PyVista can do.

.. caution::

    Please note that these examples link to external websites. If any of these
    links are broken, please raise an `issue
    <https://github.com/pyvista/pyvista/issues>`_.


Do you have a technical processing workflow or visualization routine you would
like to share? If so, please consider sharing your work here submitting a PR
at `pyvista/pyvista <https://github.com/pyvista/pyvista/>`_ and we would be
glad to add it.


.. grid:: 3
   :gutter: 1

""",
        )
        # Reverse to put the latest items at the top
        for example in list(articles.values())[::-1]:
            new_fid.write(example.format())

        new_fid.write(
            """

.. raw:: html

    <div class="sphx-glr-clear"></div>


""",
        )
        new_fid.seek(0)
        new_text = new_fid.read()

    # check if it's necessary to overwrite the table
    existing = ''
    if Path(path).exists():
        with Path(path).open() as existing_fid:
            existing = existing_fid.read()

    # write if different or does not exist
    if new_text != existing:
        with Path(path).open('w', encoding='utf-8') as fid:
            fid.write(new_text)
        print(f'Wrote external example gallery to {path}')  # noqa: T201


if __name__ == '__main__':
    make_example_gallery()
