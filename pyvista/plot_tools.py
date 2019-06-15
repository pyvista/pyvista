import pyvista



class OrthographicSlicer(object):
    """Creates an interactive plotting window to orthographically slice through
    a volumetric dataset
    """
    def __init__(self, dataset, outline=None, clean=True, border=None,
                 notebook=False, border_color='k', window_size=None,
                 generate_triangles=False, contour=False, show_bounds=False,
                 background=False, **kwargs):
        if not pyvista.is_pyvista_obj(dataset):
            dataset = pyvista.wrap(dataset)

        # Keep track of the input
        self.input_dataset = dataset

        # Keep track of output
        self.slices = [None, None, None]

        # Start the intersection point at the center
        self._location = self.input_dataset.center

        scalars = kwargs.get('scalars', self.input_dataset.active_scalar_name)
        preference = kwargs.get('preference', 'cell')
        if scalars is not None:
            self.input_dataset.set_active_scalar(scalars, preference)

        if clean and self.input_dataset.active_scalar is not None:
            # This will clean out the nan values
            self.input_dataset = self.input_dataset.threshold()

        # Hold all other kwargs for plotting
        self.show_scalar_bar = kwargs.pop('show_scalar_bar', True)
        _ = kwargs.pop('name', None)
        self.kwargs = kwargs
        self.generate_triangles = generate_triangles
        self.contour = contour
        self.show_bounds = show_bounds

        plotter = pyvista.Plotter
        if background:
            plotter = pyvista.BackgroundPlotter

        self.plotter = plotter(shape=(2, 2), border=border,
                        notebook=notebook, border_color=border_color,
                        window_size=window_size)

        self.update()

        self.plotter.subplot(1,1)
        self.plotter.isometric_view()

        self.plotter.hide_axes()
        # And show the plotter
        self.plotter.show()

    @property
    def location(self):
        return self._location

    @location.setter
    def location(self, location):
        if not pyvista.is_inside_bounds(location, self.input_dataset.bounds):
            raise ValueError('Point outside of data bounds.')
        self._location = location
        self.update()

    def update_slices(self):
        """Re runs the slicing filter"""
        axes = ['z', 'y', 'x']
        for ax in [0, 1, 2]:
            normal = axes[ax]
            slc = self.input_dataset.slice(normal=normal, origin=self.location,
                        generate_triangles=self.generate_triangles,
                        contour=self.contour)
            self.slices[ax] = slc
        return

    def _update_bounds(self):
        if self.show_bounds:
            self.plotter.show_grid()
        else:
            self.plotter.remove_bounds_axes()
        return


    def update_3d_view(self):
        self.plotter.subplot(1,1)
        self.plotter.add_mesh(self.slices[0], show_scalar_bar=self.show_scalar_bar, name='top', **self.kwargs)
        self.plotter.add_mesh(self.slices[1], show_scalar_bar=self.show_scalar_bar, name='right', **self.kwargs)
        self.plotter.add_mesh(self.slices[2], show_scalar_bar=self.show_scalar_bar, name='front', **self.kwargs)
        self._update_bounds()
        self.plotter.enable()

    def update_top_view(self):
        self.plotter.subplot(0,0)
        self.plotter.enable()
        self.plotter.add_mesh(self.slices[0], show_scalar_bar=False, name='top', **self.kwargs)
        self._update_bounds()
        self.plotter.view_xy()
        self.plotter.disable()
        return

    def update_right_view(self):
        self.plotter.subplot(0,1)
        self.plotter.enable()
        self.plotter.add_mesh(self.slices[1], show_scalar_bar=False, name='right', **self.kwargs)
        self._update_bounds()
        self.plotter.view_xz()
        self.plotter.disable()
        return

    def update_front_view(self):
        self.plotter.subplot(1,0)
        self.plotter.enable()
        self.plotter.add_mesh(self.slices[2], show_scalar_bar=False, name='front', **self.kwargs)
        self._update_bounds()
        self.plotter.view_yz()
        self.plotter.disable()
        return

    def update(self):
        self.update_slices()
        self.update_top_view()
        self.update_right_view()
        self.update_front_view()
        # Update 3D view last so its renderer is set as active
        self.update_3d_view()
