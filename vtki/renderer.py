"""
Module containing vtki implementation of vtkRenderer
"""
import collections
import vtk
from weakref import proxy

import numpy as np
from vtk import vtkRenderer

from vtki.plotting import rcParams


class Renderer(vtkRenderer):
    def __init__(self, parent):
        super(Renderer, self).__init__()
        self._actors = {}
        self.parent = parent
        self.camera_set = False
        self.bounding_box_actor = None

    def add_actor(self, uinput, reset_camera=False, name=None, loc=None):
        """
        Adds an actor to render window.  Creates an actor if input is
        a mapper.

        Parameters
        ----------
        uinput : vtk.vtkMapper or vtk.vtkActor
            vtk mapper or vtk actor to be added.

        reset_camera : bool, optional
            Resets the camera when true.

        loc : int, tuple, or list
            Index of the renderer to add the actor to.  For example, 
            ``loc=2`` or ``loc=(1, 1)``.

        Returns
        -------
        actor : vtk.vtkActor
            The actor.

        actor_properties : vtk.Properties
            Actor properties.

        """
        # Remove actor by that name if present
        rv = self.remove_actor(name, reset_camera=False)

        if isinstance(uinput, vtk.vtkMapper):
            actor = vtk.vtkActor()
            actor.SetMapper(uinput)
        else:
            actor = uinput

        self.AddActor(actor)
        actor.renderer = proxy(self)

        if name is None:
            name = str(hex(id(actor)))

        self._actors[name] = actor

        if reset_camera:
            self.reset_camera()
        elif not self.camera_set and reset_camera is None and not rv:
            self.reset_camera()
        else:
            self.parent._render()

        self.update_bounds_axes()

        try:
            actor.GetProperty().FrontfaceCullingOn()
            actor.GetProperty().BackfaceCullingOn()
        except AttributeError:
            pass

        return actor, actor.GetProperty()

    @property
    def camera_position(self):
        """ Returns camera position of active render window """
        return [self.camera.GetPosition(),
                self.camera.GetFocalPoint(),
                self.camera.GetViewUp()]

    @camera_position.setter
    def camera_position(self, camera_location):
        """ Set camera position of all active render windows """
        if camera_location is None:
            return

        # everything is set explicitly
        self.camera.SetPosition(camera_location[0])
        self.camera.SetFocalPoint(camera_location[1])
        self.camera.SetViewUp(camera_location[2])

        # reset clipping range
        self.ResetCameraClippingRange()
        self.camera_set = True

    @property
    def camera(self):
        """The active camera for the rendering scene"""
        return self.GetActiveCamera()

    def remove_actor(self, actor, reset_camera=False):
        """
        Removes an actor from the Renderer.

        Parameters
        ----------
        actor : vtk.vtkActor
            Actor that has previously added to the Renderer.

        reset_camera : bool, optional
            Resets camera so all actors can be seen.

        Returns
        -------
        success : bool
            True when actor removed.  False when actor has not been
            removed.
        """
        name = None
        if isinstance(actor, str):
            name = actor
            keys = list(self._actors.keys())
            names = []
            for k in keys:
                if k.startswith('{}-'.format(name)):
                    names.append(k)
            if len(names) > 0:
                self.remove_actor(names, reset_camera=reset_camera)
            try:
                actor = self._actors[name]
            except KeyError:
                # If actor of that name is not present then return success
                return False
        if isinstance(actor, collections.Iterable):
            success = False
            for a in actor:
                rv = self.remove_actor(a, reset_camera=reset_camera)
                if rv or success:
                    success = True
            return success
        if actor is None:
            return False

        # First remove this actor's mapper from _scalar_bar_mappers
        _remove_mapper_from_plotter(self.parent, actor, False)
        self.RemoveActor(actor)

        if name is None:
            for k, v in self._actors.items():
                if v == actor:
                    name = k
        self._actors.pop(name, None)
        self.update_bounds_axes()
        if reset_camera:
            self.reset_camera()
        elif not self.camera_set and reset_camera is None:
            self.reset_camera()
        else:
            self.parent._render()
        return True

    @property
    def bounds(self):
        """ Bounds of all actors present in the rendering window """
        the_bounds = [np.inf, -np.inf, np.inf, -np.inf, np.inf, -np.inf]

        def _update_bounds(bounds):
            def update_axis(ax):
                if bounds[ax*2] < the_bounds[ax*2]:
                    the_bounds[ax*2] = bounds[ax*2]
                if bounds[ax*2+1] > the_bounds[ax*2+1]:
                    the_bounds[ax*2+1] = bounds[ax*2+1]
            for ax in range(3):
                update_axis(ax)
            return

        for actor in self._actors.values():
            if isinstance(actor, vtk.vtkCubeAxesActor):
                continue
            if ( hasattr(actor, 'GetBounds') and actor.GetBounds() is not None
                 and id(actor) != id(self.bounding_box_actor)):
                _update_bounds(actor.GetBounds())

        return the_bounds

    @property
    def center(self):
        """Center of the bounding box around all data present in the scene"""
        bounds = self.bounds
        x = (bounds[1] + bounds[0])/2
        y = (bounds[3] + bounds[2])/2
        z = (bounds[5] + bounds[4])/2
        return [x, y, z]

    def get_default_cam_pos(self):
        """
        Returns the default focal points and viewup. Uses ResetCamera to
        make a useful view.
        """
        focal_pt = self.center
        return [np.array(rcParams['camera']['position']) + np.array(focal_pt),
                focal_pt, rcParams['camera']['viewup']]

    def update_bounds_axes(self):
        """Update the bounds axes of the render window """
        if (hasattr(self, '_box_object') and self._box_object is not None
                and self.bounding_box_actor is not None):
            if not np.allclose(self._box_object.bounds, self.bounds):
                color = self.bounding_box_actor.GetProperty().GetColor()
                self.remove_bounding_box()
                self.add_bounding_box(color=color)
        if hasattr(self, 'cube_axes_actor'):
            self.cube_axes_actor.SetBounds(self.bounds)
            if not np.allclose(self.scale, [1.0, 1.0, 1.0]):
                self.cube_axes_actor.SetUse2DMode(True)
            else:
                self.cube_axes_actor.SetUse2DMode(False)

    def reset_camera(self):
        """
        Reset camera so it slides along the vector defined from camera
        position to focal point until all of the actors can be seen.
        """
        self.ResetCamera()
        self.parent._render()

    def isometric_view(self):
        """
        Resets the camera to a default isometric view showing all the
        actors in the scene.
        """
        self.camera_position = self.get_default_cam_pos()
        self.camera_set = False
        return self.reset_camera()


def _remove_mapper_from_plotter(plotter, actor, reset_camera):
    """removes this actor's mapper from the given plotter's _scalar_bar_mappers"""
    try:
        mapper = actor.GetMapper()
    except AttributeError:
        return
    for name in list(plotter._scalar_bar_mappers.keys()):
        try:
            plotter._scalar_bar_mappers[name].remove(mapper)
        except ValueError:
            pass
        if len(plotter._scalar_bar_mappers[name]) < 1:
            plotter._scalar_bar_mappers.pop(name)
            plotter._scalar_bar_ranges.pop(name)
            plotter.remove_actor(plotter._scalar_bar_actors.pop(name), reset_camera=reset_camera)
            plotter._scalar_bar_slots.add(plotter._scalar_bar_slot_lookup.pop(name))
    return
