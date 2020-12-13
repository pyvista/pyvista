import pyvista
import pytest
import vtk


def test_interactive_update():
    p = pyvista.Plotter()
    p.show(interactive_update=True)
    assert isinstance(p.iren, vtk.vtkRenderWindowInteractor)
    p.close()

    p = pyvista.Plotter()
    with pytest.warns(UserWarning):
        p.show(auto_close=True, interactive_update=True)