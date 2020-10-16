"""
Installation file for python pyvista module
"""
import os
import sys
from io import open as io_open
from setuptools import setup

package_name = 'pyvista'

__version__ = None
filepath = os.path.dirname(__file__)
version_file = os.path.join(filepath, package_name, '_version.py')
with io_open(version_file, mode='r') as fd:
    exec(fd.read())

# Python 3.9 isn't supported at the moment
if sys.version_info.minor == 9:
    raise RuntimeError('There are no wheels available for VTK on Python 3.9 yet.  '
                       'Please use Python 3.6 through 3.8')


# pre-compiled vtk available for python3
install_requires = ['numpy',
                    'imageio',
                    'appdirs',
                    'scooby>=0.5.1',
                    'meshio>=4.0.3, <5.0',
                    'vtk']

readme_file = os.path.join(filepath, 'README.rst')

setup(
    name=package_name,
    packages=[package_name, 'pyvista.examples', 'pyvista.core',
              'pyvista.plotting', 'pyvista.utilities'],
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
                                       'globe.vtk', '2k_earth_daymap.jpg']},
    python_requires='>=3.6.*',
    install_requires=install_requires,
    extras_require={
        'colormaps': ['matplotlib', 'colorcet', 'cmocean']
    },
)
