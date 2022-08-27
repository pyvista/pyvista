from pyvista.plotting.render_window import RenderWindow


def test_render_window_init():
    ren_win = RenderWindow()
    ren_win.attach_render_window()
    ren_win.off_screen = True
    assert ren_win.rendered is False
    ren_win.show()
    assert ren_win.rendered is True

    # ensure rendered state is cached
    ren_win.finalize()
    assert ren_win.rendered is True
