"""Module managing errors."""

from __future__ import annotations

import re
import subprocess
import sys

import scooby

from pyvista._deprecate_positional_args import _deprecate_positional_args

_cmd = """\
import pyvista; \
plotter = pyvista.Plotter(notebook=False, off_screen=True); \
plotter.add_mesh(pyvista.Sphere()); \
plotter.show(auto_close=False); \
gpu_info = plotter.render_window.ReportCapabilities(); \
print(gpu_info); \
plotter.close()\
"""


def get_gpu_info():  # numpydoc ignore=RT01
    """Get all information about the GPU."""
    # an OpenGL context MUST be opened before trying to do this.
    proc = subprocess.run([sys.executable, '-c', _cmd], check=False, capture_output=True)
    return '' if proc.returncode else proc.stdout.decode()


def check_matplotlib_vtk_compatibility() -> bool:
    """Check if VTK and Matplotlib versions are compatible for MathText rendering.

    This function is primarily geared towards checking if MathText rendering is
    supported with the given versions of VTK and Matplotlib. It follows the
    version constraints:

    * VTK <= 9.2.2 requires Matplotlib < 3.6
    * VTK > 9.2.2 requires Matplotlib >= 3.6

    Other version combinations of VTK and Matplotlib will work without
    errors, but some features (like MathText/LaTeX rendering) may
    silently fail.

    Returns
    -------
    bool
        True if the versions of VTK and Matplotlib are compatible for MathText
        rendering, False otherwise.

    Raises
    ------
    RuntimeError
        If the versions of VTK and Matplotlib cannot be checked.

    """
    import matplotlib as mpl  # noqa: PLC0415

    from pyvista import vtk_version_info  # noqa: PLC0415

    mpl_vers = tuple(map(int, mpl.__version__.split('.')[:2]))
    if vtk_version_info < (9, 3, 0):
        return not mpl_vers >= (3, 6)
    elif vtk_version_info >= (9, 3, 0):
        return mpl_vers >= (3, 6)
    msg = 'Uncheckable versions.'  # pragma: no cover
    raise RuntimeError(msg)  # pragma: no cover


def check_math_text_support() -> bool:
    """Check if MathText and LaTeX symbols are supported.

    Returns
    -------
    bool
        ``True`` if both MathText and LaTeX symbols are supported, ``False``
        otherwise.

    """
    # Something seriously sketchy is happening with this VTK code
    # It seems to hijack stdout and stderr?
    # See https://github.com/pyvista/pyvista/issues/4732
    # This is a hack to get around that by executing the code in a subprocess
    # and capturing the output:
    # _vtk.vtkMathTextFreeTypeTextRenderer().MathTextIsSupported()
    _cmd = 'import vtk;print(vtk.vtkMathTextFreeTypeTextRenderer().MathTextIsSupported());'
    proc = subprocess.run([sys.executable, '-c', _cmd], check=False, capture_output=True)
    math_text_support = False if proc.returncode else proc.stdout.decode().strip() == 'True'
    return math_text_support and check_matplotlib_vtk_compatibility()


class GPUInfo:
    """A class to hold GPU details."""

    def __init__(self):
        """Instantiate a container for the GPU information."""
        self._gpu_info = get_gpu_info()

    @property
    def renderer(self):  # numpydoc ignore=RT01
        """GPU renderer name."""
        regex = re.compile('OpenGL renderer string:(.+)\n')
        try:
            renderer = regex.findall(self._gpu_info)[0]
        except IndexError:
            msg = 'Unable to parse GPU information for the renderer.'
            raise RuntimeError(msg) from None
        return renderer.strip()

    @property
    def version(self):  # numpydoc ignore=RT01
        """GPU renderer version."""
        regex = re.compile('OpenGL version string:(.+)\n')
        try:
            version = regex.findall(self._gpu_info)[0]
        except IndexError:
            msg = 'Unable to parse GPU information for the version.'
            raise RuntimeError(msg) from None
        return version.strip()

    @property
    def vendor(self):  # numpydoc ignore=RT01
        """GPU renderer vendor."""
        regex = re.compile('OpenGL vendor string:(.+)\n')
        try:
            vendor = regex.findall(self._gpu_info)[0]
        except IndexError:
            msg = 'Unable to parse GPU information for the vendor.'
            raise RuntimeError(msg) from None
        return vendor.strip()

    def get_info(self):
        """All GPU information as tuple pairs.

        Returns
        -------
        tuple
            Tuples of ``(key, info)``.

        """
        return [
            ('GPU Vendor', self.vendor),
            ('GPU Renderer', self.renderer),
            ('GPU Version', self.version),
        ]

    def _repr_html_(self):
        """HTML table representation."""
        fmt = '<table>'
        row = '<tr><th>{}</th><td>{}</td></tr>\n'
        for meta in self.get_info():
            fmt += row.format(*meta)
        fmt += '</table>'
        return fmt

    def __repr__(self):
        """Representation method."""
        content = '\n'
        for k, v in self.get_info():
            content += f'{k:>18} : {v}\n'
        content += '\n'
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
                 scipy : 1.9.1
                  tqdm : 4.64.1
                meshio : 5.3.4
            jupyterlab : 3.4.7

    """

    @_deprecate_positional_args
    def __init__(  # noqa: PLR0917
        self,
        additional=None,
        ncol=3,
        text_width=80,
        sort=False,  # noqa: FBT002
        gpu=True,  # noqa: FBT002
    ):
        """Generate a :class:`scooby.Report` instance."""
        from vtkmodules.vtkRenderingCore import vtkRenderWindow  # noqa: PLC0415

        # Mandatory packages
        core = ['pyvista', 'vtk', 'numpy', 'matplotlib', 'scooby', 'pooch', 'pillow']

        # Optional packages.
        optional = [
            'imageio',
            'pyvistaqt',
            'PyQt5',
            'IPython',
            'colorcet',
            'cmocean',
            'ipywidgets',
            'scipy',
            'tqdm',
            'meshio',
            'jupyterlab',
            'pytest_pyvista',
            'trame',
            'trame_client',
            'trame_server',
            'trame_vtk',
            'trame_vuetify',
            'jupyter_server_proxy',
            'nest_asyncio',
        ]

        # Information about the GPU - catch all Exception in case there is a rendering
        # bug that the user is trying to report.
        if gpu:
            try:
                extra_meta = GPUInfo().get_info()
            except Exception:  # noqa: BLE001
                extra_meta = [
                    ('GPU Details', 'error'),
                ]
        else:
            extra_meta = [
                ('GPU Details', 'None'),
            ]

        extra_meta.append(('Render Window', vtkRenderWindow().GetClassName()))
        extra_meta.append(('MathText Support', check_math_text_support()))

        scooby.Report.__init__(
            self,
            additional=additional,
            core=core,  # type: ignore[arg-type]
            optional=optional,  # type: ignore[arg-type]
            ncol=ncol,
            text_width=text_width,
            sort=sort,
            extra_meta=extra_meta,
        )
