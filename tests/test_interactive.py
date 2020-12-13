import pytest
import pyvista
import vtk
from pyvista.plotting import system_supports_plotting


skip_no_plotting = pytest.mark.skipif(not system_supports_plotting(),
                                      reason="Test requires system to support plotting")


@skip_no_plotting
def test_interactive_update():
    # Regression test for #1053
    p = pyvista.Plotter()
    p.show(interactive_update=True)
    assert isinstance(p.iren, vtk.vtkRenderWindowInteractor)
    p.close()

    p = pyvista.Plotter()
    with pytest.warns(UserWarning):
        p.show(auto_close=True, interactive_update=True)