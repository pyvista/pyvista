"""
Installation file for python vtki module
"""
import os
import platform
import sys
import warnings
from io import open as io_open

from setuptools import setup

package_name = 'vtki'

__version__ = None
filepath = os.path.dirname(__file__)
version_file = os.path.join(filepath, package_name, '_version.py')
with io_open(version_file, mode='r') as fd:
    exec(fd.read())

# pre-compiled vtk available for python3
install_requires = ['numpy',
                    'imageio',
                    'appdirs',
                    ]

# add vtk if not windows and 2.7
py_ver = int(sys.version[0])
if os.name == 'nt' and (py_ver < 3 or '64' not in platform.architecture()[0]):
    warnings.warn('\nYou will need to install VTK manually.' +
                  '  Try using Anaconda.  See:\n'
                  + 'https://anaconda.org/anaconda/vtk')
# elif os.environ.get('READTHEDOCS') != 'True':
    # pass  # don't install vtk for readthedocs
else:
    install_requires.append(['vtk'])


readme_file = os.path.join(filepath, 'README.rst')

setup(
    name=package_name,
    packages=[package_name, 'vtki.examples'],
    version=__version__,
    description='Easier Pythonic interface to VTK',
    long_description=open(readme_file).read(),
    author='Alex Kaszynski',
    author_email='akascap@gmail.com',
    license='MIT',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'License :: OSI Approved :: MIT License',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX',
        'Operating System :: MacOS',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],

    url='https://github.com/vtkiorg/vtki',
    keywords='vtk numpy plotting mesh',
    package_data={'vtki.examples': ['airplane.ply', 'ant.ply', 'channels.vti',
                                    'hexbeam.vtk', 'sphere.ply',
                                    'uniform.vtk', 'rectilinear.vtk',
                                    'globe.vtk', '2k_earth_daymap.jpg']},
    install_requires=install_requires,
    extras_require={
        'ipy_tools': ['ipython', 'ipywidgets'],
    },
)
