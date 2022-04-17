"""VRML examples."""

def download_teapot():
    """Download the a 2-manifold solid version of the famous teapot example.

    Files hosted at https://github.com/lorensen/VTKExamples/blob/master/src/Testing/Data

    Returns
    -------
    str
        Filename of the VRML file.

    Examples
    --------
    >>> import pyvista
    >>> from pyvista import examples
    >>> vrml_file = examples.vrml.download_teapot()
    >>> pl = pyvista.Plotter()
    >>> pl.import_vrml(vrml_file)
    >>> pl.show()

    """
    return "https://raw.githubusercontent.com/lorensen/VTKExamples/master/src/Testing/Data/teapot.wrl"
