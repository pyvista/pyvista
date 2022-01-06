"""
Installation file for python pyvista module
"""
from io import open as io_open
import os

from setuptools import setup

package_name = 'pyvista'

__version__ = None
filepath = os.path.dirname(__file__)
version_file = os.path.join(filepath, package_name, '_version.py')
with io_open(version_file, mode='r') as fd:
    exec(fd.read())

install_requires = ['numpy',
                    'imageio',
                    'pillow',
                    'appdirs',
                    'scooby>=0.5.1',
                    'vtk',
                    'dataclasses;python_version=="3.6"',
                    'typing_extensions;python_version<="3.7"',
                    ]

readme_file = os.path.join(filepath, 'README.rst')

setup(
    name=package_name,
    packages=['pyvista',
              'pyvista.examples',
              'pyvista.core',
              'pyvista.core.filters',
              'pyvista.demos',
              'pyvista.jupyter',
              'pyvista.plotting',
              'pyvista.utilities',
              'pyvista.ext'],
    version=__version__,
    description='Easier Pythonic interface to VTK',
    long_description=io_open(readme_file, encoding="utf-8").read(),
    long_description_content_type='text/x-rst',
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
        'Programming Language :: Python :: 3.9',
    ],

    url='https://github.com/pyvista/pyvista',
    keywords='vtk numpy plotting mesh',
    package_data={'pyvista.examples': ['airplane.ply', 'ant.ply', 'channels.vti',
                                       'hexbeam.vtk', 'sphere.ply', 'nut.ply',
                                       'uniform.vtk', 'rectilinear.vtk',
                                       'globe.vtk', '2k_earth_daymap.jpg'],
    },

    python_requires='>=3.6.*',
    install_requires=install_requires,
    extras_require={
        'all': ['matplotlib', 'colorcet', 'cmocean', 'meshio'],
        'colormaps': ['matplotlib', 'colorcet', 'cmocean'],
        'io': ['meshio>=5.2'],
    },
)
