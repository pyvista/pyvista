from pyvista.plotting.render_window import RenderWindow


def test_render_window_init():
    ren_win = RenderWindow()
    ren_win.attach_render_window()
    ren_win.off_screen = True
    ren_win.show()
