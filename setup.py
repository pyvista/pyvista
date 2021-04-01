"""
Installation file for python pyvista module
"""
import os
# import sys
from io import open as io_open
from setuptools import setup

package_name = 'pyvista'

__version__ = None
filepath = os.path.dirname(__file__)
version_file = os.path.join(filepath, package_name, '_version.py')
with io_open(version_file, mode='r') as fd:
    exec(fd.read())

# leaving this out for conda compatibility...
# python3_9_linux_wheel = 'https://github.com/pyvista/pyvista/releases/download/0.27.0/vtk-9.0.1-cp39-cp39-manylinux2010_x86_64.whl'

# # Python 3.9 isn't supported at the moment
# if sys.version_info.minor == 9:
#     # but, the user might have installed vtk from a non-pypi wheel.
#     try:
#         import vtk
#     except ImportError:
#         note = ''
#         if os.name == 'linux':
#             note = '\n\nHowever there is an unofficial Linux wheel build by the ``pyvista`` team at ' + python3_9_linux_wheel

#         raise RuntimeError('There are no official Python 3.9 wheels for VTK on yet.  '
#                            'Please use Python 3.6 through 3.8, or build and install '
#                            'VTK from source with a wheel.  Please see:\n'
#                            'https://docs.pyvista.org/building_vtk.html' + note)


# pre-compiled vtk available for python3
install_requires = ['numpy',
                    'imageio',
                    'pillow',
                    'appdirs',
                    'scooby>=0.5.1',
                    'meshio>=4.0.3, <5.0',
                    'vtk',
                    'transforms3d==0.3.1'
                    ]

readme_file = os.path.join(filepath, 'README.rst')

setup(
    name=package_name,
    packages=['pyvista',
              'pyvista.examples',
              'pyvista.core',
              'pyvista.demos',
              'pyvista.jupyter',
              'pyvista.plotting',
              'pyvista.utilities'],
    version=__version__,
    description='Easier Pythonic interface to VTK',
    long_description=io_open(readme_file, encoding="utf-8").read(),
    author='PyVista Developers',
    author_email='info@pyvista.org',
    license='MIT',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'License :: OSI Approved :: MIT License',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX',
        'Operating System :: MacOS',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],

    url='https://github.com/pyvista/pyvista',
    keywords='vtk numpy plotting mesh',
    package_data={'pyvista.examples': ['airplane.ply', 'ant.ply', 'channels.vti',
                                       'hexbeam.vtk', 'sphere.ply',
                                       'uniform.vtk', 'rectilinear.vtk',
                                       'globe.vtk', '2k_earth_daymap.jpg'],
    },

    python_requires='>=3.6.*',
    install_requires=install_requires,
    extras_require={
        'colormaps': ['matplotlib', 'colorcet', 'cmocean']
    },
)
