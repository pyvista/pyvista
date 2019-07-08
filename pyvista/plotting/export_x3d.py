import os

import vtk

import pyvista


X3D_JAVASCRIPT = '''
<?xml version="1.0" encoding ="UTF-8"?>
<!DOCTYPE X3D PUBLIC "ISO//Web3D//DTD X3D 3.3//EN"
"http://www.web3d.org/specifications/x3d-3.3.dtd">
<head>
<script type='text/javascript' src='http://www.x3dom.org/x3dom/release/x3dom.js'> </script>
<link rel='stylesheet' type='text/css' href='http://www.x3dom.org/x3dom/release/x3dom.css'></link>
</head>
<body>
{}
</body>
'''

def export_x3d(plotter, filename=None, binary=True, speed=4.0):
    """Exports the scene to an X3D (XML-based format) for
    representation 3D scenes (similar to VRML). Check out
    http://www.web3d.org/x3d/ for more details.

    If no filename given, the raw HTML for this scene with be returned
    """
    if not hasattr(plotter, 'ren_win'):
        raise RuntimeError('Export must be called before showing/closing the scene.')
    exporter = vtk.vtkX3DExporter()
    exporter.SetInput(plotter.ren_win)
    exporter.FastestOff()
    if speed:
        exporter.SetSpeed(speed)
    if filename is not None:
        if isinstance(pyvista.FIGURE_PATH, str) and not os.path.isabs(filename):
            filename = os.path.join(pyvista.FIGURE_PATH, filename)
        else:
            filename = os.path.abspath(os.path.expanduser(filename))
        exporter.SetFileName(filename)
        exporter.SetBinary(binary)
    else:
        exporter.SetWriteToOutputString(True)
    exporter.Update()
    exporter.Write()
    if filename is None:
        return exporter.GetOutputString()
    return
