import os

import numpy as np
import pytest

import pyvista
import vtk
from pyvista import QtInteractor, MainWindow, Renderer
from pyvista.plotting import system_supports_plotting
from pyvista.plotting.qt_plotting import (QVTKRenderWindowInteractor, QTimer,
                                          _create_menu_bar)


NO_PLOTTING = not system_supports_plotting()


try:
    from PyQt5.Qt import (QMainWindow, QFrame, QVBoxLayout, QMenuBar, QAction, QToolBar)
    has_pyqt5 = True
except:
    has_pyqt5 = False
    class QMainWindow(object):
        pass


class TstWindow(MainWindow):
    def __init__(self, parent=None, show=True, off_screen=True):
        MainWindow.__init__(self, parent)

        self.frame = QFrame()
        vlayout = QVBoxLayout()
        self.vtk_widget = QtInteractor(
            parent=self.frame,
            off_screen=off_screen
        )
        vlayout.addWidget(self.vtk_widget.interactor)

        self.frame.setLayout(vlayout)
        self.setCentralWidget(self.frame)

        mainMenu = _create_menu_bar(parent=self)

        fileMenu = mainMenu.addMenu('File')
        self.exit_action = QAction('Exit', self)
        self.exit_action.setShortcut('Ctrl+Q')
        self.exit_action.triggered.connect(self.close)
        fileMenu.addAction(self.exit_action)

        meshMenu = mainMenu.addMenu('Mesh')
        self.add_sphere_action = QAction('Add Sphere', self)
        self.exit_action.setShortcut('Ctrl+A')
        self.add_sphere_action.triggered.connect(self.add_sphere)
        meshMenu.addAction(self.add_sphere_action)

        self.signal_close.connect(self.vtk_widget.close)

        if show:
            self.show()

    def add_sphere(self):
        sphere = pyvista.Sphere(
            phi_resolution=6,
            theta_resolution=6
        )
        self.vtk_widget.add_mesh(sphere)
        self.vtk_widget.reset_camera()


@pytest.mark.skipif(NO_PLOTTING, reason="Requires system to support plotting")
@pytest.mark.skipif(not has_pyqt5, reason="requires pyqt5")
def test_qt_interactor(qtbot):
    from pyvista.plotting.plotting import close_all, _ALL_PLOTTERS
    close_all()  # this is necessary to test _ALL_PLOTTERS
    assert len(_ALL_PLOTTERS) == 0

    window = TstWindow(show=False, off_screen=False)
    qtbot.addWidget(window)  # register the main widget

    # check that TstWindow.__init__() is called
    assert _hasattr(window, "vtk_widget", QtInteractor)

    vtk_widget = window.vtk_widget  # QtInteractor

    # check that QtInteractor.__init__() is called
    assert _hasattr(vtk_widget, "iren", vtk.vtkRenderWindowInteractor)
    assert _hasattr(vtk_widget, "render_timer", QTimer)
    # check that BasePlotter.__init__() is called
    assert _hasattr(vtk_widget, "_style", vtk.vtkInteractorStyle)
    assert _hasattr(vtk_widget, "_closed", bool)
    # check that QVTKRenderWindowInteractorAdapter.__init__() is called
    assert _hasattr(vtk_widget, "interactor", QVTKRenderWindowInteractor)

    interactor = vtk_widget.interactor  # QVTKRenderWindowInteractor
    render_timer = vtk_widget.render_timer  # QTimer

    # ensure that self.render is called by the timer
    render_blocker = qtbot.wait_signals([render_timer.timeout], timeout=500)
    render_blocker.wait()

    window.add_sphere()
    assert np.any(window.vtk_widget.mesh.points)

    with qtbot.wait_exposed(window, timeout=5000):
        window.show()
    with qtbot.wait_exposed(interactor, timeout=5000):
        interactor.show()

    assert window.isVisible()
    assert interactor.isVisible()
    assert render_timer.isActive()
    assert not vtk_widget._closed

    window.close()

    assert not window.isVisible()
    assert not interactor.isVisible()
    assert not render_timer.isActive()

    # check that BasePlotter.close() is called
    assert not hasattr(vtk_widget, "_style")
    assert not hasattr(vtk_widget, "iren")
    assert vtk_widget._closed

    # check that BasePlotter.__init__() is called only once
    assert len(_ALL_PLOTTERS) == 1


@pytest.mark.skipif(NO_PLOTTING, reason="Requires system to support plotting")
@pytest.mark.skipif(not has_pyqt5, reason="requires pyqt5")
@pytest.mark.parametrize('show_plotter', [
    True,
    False,
    ])
def test_background_plotting_axes_scale(qtbot, show_plotter):
    plotter = pyvista.BackgroundPlotter(
        show=show_plotter,
        off_screen=False,
        title='Testing Window'
    )
    assert _hasattr(plotter, "app_window", MainWindow)
    window = plotter.app_window  # MainWindow
    qtbot.addWidget(window)  # register the window

    # show the window
    if not show_plotter:
        assert not window.isVisible()
        with qtbot.wait_exposed(window, timeout=1000):
            window.show()
    assert window.isVisible()

    plotter.add_mesh(pyvista.Sphere())
    assert _hasattr(plotter, "renderer", Renderer)
    renderer = plotter.renderer
    assert len(renderer._actors) == 1
    assert np.any(plotter.mesh.points)

    dlg = plotter.scale_axes_dialog(show=False)  # ScaleAxesDialog
    qtbot.addWidget(dlg)  # register the dialog

    # show the dialog
    assert not dlg.isVisible()
    with qtbot.wait_exposed(dlg, timeout=500):
        dlg.show()
    assert dlg.isVisible()

    value = 2.0
    dlg.x_slider_group.value = value
    assert plotter.scale[0] == value
    dlg.x_slider_group.spinbox.setValue(-1)
    assert dlg.x_slider_group.value == 0
    dlg.x_slider_group.spinbox.setValue(1000.0)
    assert dlg.x_slider_group.value < 100

    plotter._last_update_time = 0.0
    plotter.update()
    plotter.update_app_icon()
    plotter.close()
    assert not window.isVisible()
    assert not dlg.isVisible()


@pytest.mark.skipif(NO_PLOTTING, reason="Requires system to support plotting")
@pytest.mark.skipif(not has_pyqt5, reason="requires pyqt5")
def test_background_plotting_camera(qtbot):
    plotter = pyvista.BackgroundPlotter(off_screen=False, title='Testing Window')
    plotter.add_mesh(pyvista.Sphere())

    cpos = [(0.0, 0.0, 1.0), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
    plotter.camera_position = cpos
    plotter.save_camera_position()
    plotter.camera_position = [(0.0, 0.0, 3.0), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)]

    # load existing position
    # NOTE: 2 because first two (0 and 1) buttons save and clear positions
    plotter.saved_cameras_tool_bar.actions()[2].trigger()
    assert plotter.camera_position == cpos

    plotter.clear_camera_positions()
    # 2 because the first two buttons are save and clear
    assert len(plotter.saved_cameras_tool_bar.actions()) == 2
    plotter.close()


@pytest.mark.skipif(NO_PLOTTING, reason="Requires system to support plotting")
@pytest.mark.skipif(not has_pyqt5, reason="requires pyqt5")
@pytest.mark.parametrize('show_plotter', [
    True,
    False,
    ])
def test_background_plotter_export_files(qtbot, tmpdir, show_plotter):
    # setup filesystem
    output_dir = str(tmpdir.mkdir("tmpdir"))
    assert os.path.isdir(output_dir)

    plotter = pyvista.BackgroundPlotter(
        show=show_plotter,
        off_screen=False,
        title='Testing Window'
    )
    assert _hasattr(plotter, "app_window", MainWindow)
    window = plotter.app_window  # MainWindow
    qtbot.addWidget(window)  # register the window

    # show the window
    if not show_plotter:
        assert not window.isVisible()
        with qtbot.wait_exposed(window, timeout=1000):
            window.show()
    assert window.isVisible()

    plotter.add_mesh(pyvista.Sphere())
    assert _hasattr(plotter, "renderer", Renderer)
    renderer = plotter.renderer
    assert len(renderer._actors) == 1
    assert np.any(plotter.mesh.points)

    dlg = plotter._qt_screenshot(show=False)  # FileDialog
    qtbot.addWidget(dlg)  # register the dialog

    filename = str(os.path.join(output_dir, "tmp.png"))
    dlg.selectFile(filename)

    # show the dialog
    assert not dlg.isVisible()
    with qtbot.wait_exposed(dlg, timeout=500):
        dlg.show()
    assert dlg.isVisible()

    # synchronise signal and callback
    with qtbot.wait_signals([dlg.dlg_accepted], timeout=1000):
        dlg.accept()
    assert not dlg.isVisible()  # dialog is closed after accept()

    plotter.close()
    assert not window.isVisible()
    assert os.path.isfile(filename)


@pytest.mark.skipif(NO_PLOTTING, reason="Requires system to support plotting")
@pytest.mark.skipif(not has_pyqt5, reason="requires pyqt5")
@pytest.mark.parametrize('show_plotter', [
    True,
    False,
    ])
def test_background_plotter_export_vtkjs(qtbot, tmpdir, show_plotter):
    # setup filesystem
    output_dir = str(tmpdir.mkdir("tmpdir"))
    assert os.path.isdir(output_dir)

    plotter = pyvista.BackgroundPlotter(
        show=show_plotter,
        off_screen=False,
        title='Testing Window'
    )
    assert _hasattr(plotter, "app_window", MainWindow)
    window = plotter.app_window  # MainWindow
    qtbot.addWidget(window)  # register the window

    # show the window
    if not show_plotter:
        assert not window.isVisible()
        with qtbot.wait_exposed(window, timeout=1000):
            window.show()
    assert window.isVisible()

    plotter.add_mesh(pyvista.Sphere())
    assert _hasattr(plotter, "renderer", Renderer)
    renderer = plotter.renderer
    assert len(renderer._actors) == 1
    assert np.any(plotter.mesh.points)

    dlg = plotter._qt_export_vtkjs(show=False)  # FileDialog
    qtbot.addWidget(dlg)  # register the dialog

    filename = str(os.path.join(output_dir, "tmp"))
    dlg.selectFile(filename)

    # show the dialog
    assert not dlg.isVisible()
    with qtbot.wait_exposed(dlg, timeout=500):
        dlg.show()
    assert dlg.isVisible()

    # synchronise signal and callback
    with qtbot.wait_signals([dlg.dlg_accepted], timeout=1000):
        dlg.accept()
    assert not dlg.isVisible()  # dialog is closed after accept()

    plotter.close()
    assert not window.isVisible()
    assert os.path.isfile(filename + '.vtkjs')


@pytest.mark.skipif(NO_PLOTTING, reason="Requires system to support plotting")
@pytest.mark.skipif(not has_pyqt5, reason="requires pyqt5")
def test_background_plotting_orbit(qtbot):
    plotter = pyvista.BackgroundPlotter(off_screen=False, title='Testing Window')
    plotter.add_mesh(pyvista.Sphere())
    # perform the orbit:
    plotter.orbit_on_path(bkg=False, step=0.0)
    plotter.close()

@pytest.mark.skipif(NO_PLOTTING, reason="Requires system to support plotting")
@pytest.mark.skipif(not has_pyqt5, reason="requires pyqt5")
def test_background_plotting_toolbar(qtbot):
    with pytest.raises(TypeError, match='toolbar'):
        pyvista.BackgroundPlotter(off_screen=False, toolbar="foo")

    plotter = pyvista.BackgroundPlotter(off_screen=False, toolbar=False)
    assert not hasattr(plotter, "default_camera_tool_bar")
    assert not hasattr(plotter, "saved_cameras_tool_bar")
    plotter.close()

    plotter = pyvista.BackgroundPlotter(off_screen=False)

    assert _hasattr(plotter, "app_window", MainWindow)
    assert _hasattr(plotter, "default_camera_tool_bar", QToolBar)
    assert _hasattr(plotter, "saved_cameras_tool_bar", QToolBar)

    window = plotter.app_window
    default_camera_tool_bar = plotter.default_camera_tool_bar
    saved_cameras_tool_bar = plotter.saved_cameras_tool_bar

    with qtbot.wait_exposed(window, timeout=500):
        window.show()

    assert default_camera_tool_bar.isVisible()
    assert saved_cameras_tool_bar.isVisible()

    plotter.close()


@pytest.mark.skipif(NO_PLOTTING, reason="Requires system to support plotting")
@pytest.mark.skipif(not has_pyqt5, reason="requires pyqt5")
def test_background_plotting_add_callback(qtbot):
    class CallBack(object):
        def __init__(self, sphere):
            self.sphere = sphere

        def __call__(self):
            self.sphere.points *= 0.5

    plotter = pyvista.BackgroundPlotter(
            show=False,
            off_screen=False,
            title='Testing Window'
    )
    sphere = pyvista.Sphere()
    mycallback = CallBack(sphere)
    plotter.add_mesh(sphere)
    plotter.add_callback(mycallback, interval=200, count=3)

    # check that timers are set properly in add_callback()
    assert _hasattr(plotter, "app_window", MainWindow)
    assert _hasattr(plotter, "_callback_timer", QTimer)
    assert _hasattr(plotter, "counters", list)

    window = plotter.app_window  # MainWindow
    callback_timer = plotter._callback_timer  # QTimer
    counter = plotter.counters[-1]  # Counter

    # ensure that the window is showed
    assert not window.isVisible()
    with qtbot.wait_exposed(window, timeout=500):
        window.show()
    assert window.isVisible()
    # ensure that self.callback_timer send a signal
    callback_blocker = qtbot.wait_signals([callback_timer.timeout], timeout=300)
    callback_blocker.wait()
    # ensure that self.counters send a signal
    counter_blocker = qtbot.wait_signals([counter.signal_finished], timeout=700)
    counter_blocker.wait()
    assert not callback_timer.isActive()  # counter stops the callback

    plotter.add_callback(mycallback, interval=200)
    callback_timer = plotter._callback_timer  # QTimer

    # ensure that self.callback_timer send a signal
    callback_blocker = qtbot.wait_signals([callback_timer.timeout], timeout=300)
    callback_blocker.wait()

    assert callback_timer.isActive()
    plotter.close()
    assert not callback_timer.isActive()  # window stops the callback


@pytest.mark.skipif(NO_PLOTTING, reason="Requires system to support plotting")
@pytest.mark.skipif(not has_pyqt5, reason="requires pyqt5")
@pytest.mark.parametrize('close_event', [
    "plotter_close",
    "window_close",
    "q_key_press",
    "menu_exit",
    "del_finalizer",
    ])
@pytest.mark.parametrize('empty_scene', [
    True,
    False,
    ])
def test_background_plotting_close(qtbot, close_event, empty_scene):
    from pyvista.plotting.plotting import close_all, _ALL_PLOTTERS
    close_all()  # this is necessary to test _ALL_PLOTTERS
    assert len(_ALL_PLOTTERS) == 0

    plotter = _create_testing_scene(empty_scene)

    # check that BackgroundPlotter.__init__() is called
    assert _hasattr(plotter, "app_window", MainWindow)
    assert _hasattr(plotter, "main_menu", QMenuBar)
    # check that QtInteractor.__init__() is called
    assert _hasattr(plotter, "iren", vtk.vtkRenderWindowInteractor)
    assert _hasattr(plotter, "render_timer", QTimer)
    # check that BasePlotter.__init__() is called
    assert _hasattr(plotter, "_style", vtk.vtkInteractorStyle)
    assert _hasattr(plotter, "_closed", bool)
    # check that QVTKRenderWindowInteractorAdapter._init__() is called
    assert _hasattr(plotter, "interactor", QVTKRenderWindowInteractor)

    window = plotter.app_window  # MainWindow
    main_menu = plotter.main_menu  # QMenuBar
    assert not main_menu.isNativeMenuBar()
    interactor = plotter.interactor  # QVTKRenderWindowInteractor
    render_timer = plotter.render_timer  # QTimer

    qtbot.addWidget(window)  # register the main widget

    # ensure that self.render is called by the timer
    render_blocker = qtbot.wait_signals([render_timer.timeout], timeout=500)
    render_blocker.wait()

    # a full scene may take a while to setup, especially on macOS
    show_timeout = 500 if empty_scene else 10000

    # ensure that the widgets are showed
    with qtbot.wait_exposed(window, timeout=show_timeout):
        window.show()
    with qtbot.wait_exposed(interactor, timeout=show_timeout):
        interactor.show()

    # check that the widgets are showed properly
    assert window.isVisible()
    assert interactor.isVisible()
    assert main_menu.isVisible()
    assert render_timer.isActive()
    assert not plotter._closed

    with qtbot.wait_signals([window.signal_close], timeout=500):
        if close_event == "plotter_close":
            plotter.close()
        elif close_event == "window_close":
            window.close()
        elif close_event == "q_key_press":
            qtbot.keyClick(interactor, "q")
        elif close_event == "menu_exit":
            plotter._menu_close_action.trigger()
        elif close_event == "del_finalizer":
            plotter.__del__()

    # check that the widgets are closed
    assert not window.isVisible()
    assert not interactor.isVisible()
    assert not main_menu.isVisible()
    assert not render_timer.isActive()

    # check that BasePlotter.close() is called
    assert not hasattr(plotter, "_style")
    assert not hasattr(plotter, "iren")
    assert plotter._closed

    # check that BasePlotter.__init__() is called only once
    assert len(_ALL_PLOTTERS) == 1


def _create_testing_scene(empty_scene, show=False, off_screen=False):
    if empty_scene:
        plotter = pyvista.BackgroundPlotter(
            show=show,
            off_screen=off_screen
        )
    else:
        plotter = pyvista.BackgroundPlotter(
            shape=(2, 2),
            border=True,
            border_width=10,
            border_color='grey',
            show=show,
            off_screen=off_screen
        )
        plotter.set_background('black', top='blue')
        plotter.subplot(0, 0)
        cone = pyvista.Cone(resolution=4)
        actor = plotter.add_mesh(cone)
        plotter.remove_actor(actor)
        plotter.add_text('Actor is removed')
        plotter.subplot(0, 1)
        plotter.add_mesh(pyvista.Box(), color='green', opacity=0.8)
        plotter.subplot(1, 0)
        cylinder = pyvista.Cylinder(resolution=6)
        plotter.add_mesh(cylinder, smooth_shading=True)
        plotter.show_bounds()
        plotter.subplot(1, 1)
        sphere = pyvista.Sphere(
            phi_resolution=6,
            theta_resolution=6
        )
        plotter.add_mesh(sphere)
        plotter.enable_cell_picking()
    return plotter


def _hasattr(variable, attribute_name, variable_type):
    if not hasattr(variable, attribute_name):
        return False
    return isinstance(getattr(variable, attribute_name), variable_type)
