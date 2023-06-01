"""Module managing errors."""
from collections import namedtuple
import re
import subprocess
import sys
import warnings

import scooby
from vtkmodules.vtkCommonCore import vtkVersion


def VTKVersionInfo():
    """Return the vtk version as a namedtuple."""
    version_info = namedtuple('VTKVersionInfo', ['major', 'minor', 'micro'])

    try:
        ver = vtkVersion()
        major = ver.GetVTKMajorVersion()
        minor = ver.GetVTKMinorVersion()
        micro = ver.GetVTKBuildVersion()
    except AttributeError:  # pragma: no cover
        warnings.warn("Unable to detect VTK version. Defaulting to v4.0.0")
        major, minor, micro = (4, 0, 0)

    return version_info(major, minor, micro)


vtk_version_info = VTKVersionInfo()


_cmd = """\
import pyvista; \
plotter = pyvista.Plotter(notebook=False, off_screen=True); \
plotter.add_mesh(pyvista.Sphere()); \
plotter.show(auto_close=False); \
gpu_info = plotter.render_window.ReportCapabilities(); \
print(gpu_info); \
plotter.close()\
"""


def get_gpu_info():
    """Get all information about the GPU."""
    # an OpenGL context MUST be opened before trying to do this.
    proc = subprocess.run([sys.executable, '-c', _cmd], check=False, capture_output=True)
    gpu_info = '' if proc.returncode else proc.stdout.decode()
    return gpu_info


class GPUInfo:
    """A class to hold GPU details."""

    def __init__(self):
        """Instantiate a container for the GPU information."""
        self._gpu_info = get_gpu_info()

    @property
    def renderer(self):
        """GPU renderer name."""
        regex = re.compile("OpenGL renderer string:(.+)\n")
        try:
            renderer = regex.findall(self._gpu_info)[0]
        except IndexError:
            raise RuntimeError("Unable to parse GPU information for the renderer.") from None
        return renderer.strip()

    @property
    def version(self):
        """GPU renderer version."""
        regex = re.compile("OpenGL version string:(.+)\n")
        try:
            version = regex.findall(self._gpu_info)[0]
        except IndexError:
            raise RuntimeError("Unable to parse GPU information for the version.") from None
        return version.strip()

    @property
    def vendor(self):
        """GPU renderer vendor."""
        regex = re.compile("OpenGL vendor string:(.+)\n")
        try:
            vendor = regex.findall(self._gpu_info)[0]
        except IndexError:
            raise RuntimeError("Unable to parse GPU information for the vendor.") from None
        return vendor.strip()

    def get_info(self):
        """All GPU information as tuple pairs."""
        return [
            ("GPU Vendor", self.vendor),
            ("GPU Renderer", self.renderer),
            ("GPU Version", self.version),
        ]

    def _repr_html_(self):
        """HTML table representation."""
        fmt = "<table>"
        row = "<tr><th>{}</th><td>{}</td></tr>\n"
        for meta in self.get_info():
            fmt += row.format(*meta)
        fmt += "</table>"
        return fmt

    def __repr__(self):
        """Representation method."""
        content = "\n"
        for k, v in self.get_info():
            content += f"{k:>18} : {v}\n"
        content += "\n"
        return content


class Report(scooby.Report):
    """Generate a PyVista software environment report.

    Parameters
    ----------
    additional : sequence[types.ModuleType], sequence[str]
        List of packages or package names to add to output information.

    ncol : int, default: 3
        Number of package-columns in html table; only has effect if
        ``mode='HTML'`` or ``mode='html'``.

    text_width : int, default: 80
        The text width for non-HTML display modes.

    sort : bool, default: False
        Alphabetically sort the packages.

    gpu : bool, default: True
        Gather information about the GPU. Defaults to ``True`` but if
        experiencing rendering issues, pass ``False`` to safely generate a
        report.

    Examples
    --------
    >>> import pyvista as pv
    >>> pv.Report()  # doctest:+SKIP
      Date: Fri Oct 28 15:54:11 2022 MDT
    <BLANKLINE>
                    OS : Linux
                CPU(s) : 6
               Machine : x86_64
          Architecture : 64bit
                   RAM : 62.6 GiB
           Environment : IPython
           File system : ext4
            GPU Vendor : NVIDIA Corporation
          GPU Renderer : Quadro P2000/PCIe/SSE2
           GPU Version : 4.5.0 NVIDIA 470.141.03
    <BLANKLINE>
      Python 3.8.10 (default, Jun 22 2022, 20:18:18)  [GCC 9.4.0]
    <BLANKLINE>
               pyvista : 0.37.dev0
                   vtk : 9.1.0
                 numpy : 1.23.3
               imageio : 2.22.0
                scooby : 0.7.1.dev1+gf097dad
                 pooch : v1.6.0
            matplotlib : 3.6.0
               IPython : 7.31.0
              colorcet : 3.0.1
               cmocean : 2.0
            ipyvtklink : 0.2.3
                 scipy : 1.9.1
                  tqdm : 4.64.1
                meshio : 5.3.4
            jupyterlab : 3.4.7
             pythreejs : Version unknown

    """

    def __init__(self, additional=None, ncol=3, text_width=80, sort=False, gpu=True):
        """Generate a :class:`scooby.Report` instance."""
        from pyvista.plotting.tools import check_math_text_support

        # Mandatory packages
        core = ['pyvista', 'vtk', 'numpy', 'matplotlib', 'scooby', 'pooch']

        # Optional packages.
        optional = [
            'imageio',
            'pyvistaqt',
            'PyQt5',
            'IPython',
            'colorcet',
            'cmocean',
            'ipyvtklink',
            'ipywidgets',
            'scipy',
            'tqdm',
            'meshio',
            'jupyterlab',
            'pythreejs',
            'pytest_pyvista',
            'trame',
            'trame_client',
            'trame_server',
            'trame_vtk',
            'jupyter_server_proxy',
            'nest_asyncio',
        ]

        # Information about the GPU - bare except in case there is a rendering
        # bug that the user is trying to report.
        if gpu:
            try:
                extra_meta = GPUInfo().get_info()
            except:
                extra_meta = [
                    ("GPU Details", "error"),
                ]
        else:
            extra_meta = [
                ("GPU Details", "None"),
            ]

        extra_meta.append(('MathText Support', check_math_text_support()))

        scooby.Report.__init__(
            self,
            additional=additional,
            core=core,
            optional=optional,
            ncol=ncol,
            text_width=text_width,
            sort=sort,
            extra_meta=extra_meta,
        )
