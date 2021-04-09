"""
Multiple Slider Widgets
~~~~~~~~~~~~~~~~~~~~~~~

Use a class based callback to track multiple slider widgets for updating a
single mesh.

In this example we simply change a few parameters for the
:func:`pyvista.Sphere` method, but this could easily be applied to any
mesh-generating/altering code.

"""
import pyvista as pv


class MyCustomRoutine():
    def __init__(self, mesh):
        self.output = mesh # Expected PyVista mesh type
        # default parameters
        self.kwargs = {
            'radius': 0.5,
            'theta_resolution': 30,
            'phi_resolution': 30,
        }

    def __call__(self, param, value):
        self.kwargs[param] = value
        self.update()

    def update(self):
        # This is where you call your simulation
        result = pv.Sphere(**self.kwargs)
        self.output.overwrite(result)
        return

###############################################################################

starting_mesh = pv.Sphere()
engine = MyCustomRoutine(starting_mesh)

###############################################################################

p = pv.Plotter()
p.add_mesh(starting_mesh, show_edges=True)
p.add_slider_widget(
    callback=lambda value: engine('phi_resolution', int(value)),
    rng=[3, 60],
    value=30,
    title="Phi Resolution",
    pointa=(.025, .1), pointb=(.31, .1),
    style='modern',
)
p.add_slider_widget(
    callback=lambda value: engine('theta_resolution', int(value)),
    rng=[3, 60],
    value=30,
    title="Theta Resolution",
    pointa=(.35, .1), pointb=(.64, .1),
    style='modern',
)
p.add_slider_widget(
    callback=lambda value: engine('radius', value),
    rng=[0.1, 1.5],
    value=0.5,
    title="Radius",
    pointa=(.67, .1), pointb=(.98, .1),
    style='modern',
)
p.show()
