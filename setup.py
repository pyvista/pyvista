"""
Installation file for python vtkInterface module
"""

from setuptools import setup
import os
from io import open as io_open

package_name = 'vtkInterface'

# Get version from tqdm/_version.py
__version__ = None
version_file = os.path.join(os.path.dirname(__file__), package_name, '_version.py')
with io_open(version_file, mode='r') as fd:
    # execute file from raw string
    exec(fd.read())
    

# Actual setup
setup(
    name=package_name,
    packages = [package_name, 'vtkInterface.tests', 'vtkInterface.examples'],

    # Version
    version=__version__,

    description='Easier Pythonic interface to VTK',
    long_description=open('README.rst').read(),
#    long_description=open('pypiREADME.rst').read(),

    # Author details
    author='Alex Kaszynski',
    author_email='akascap@gmail.com',

    license='MIT',
    classifiers=[
        'Development Status :: 4 - Beta',

        # Target audience
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Information Analysis',

        # MIT License
        'License :: OSI Approved :: MIT License',

        # Untested, but will probably work for other python versions
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],

    # Website
    url = 'https://github.com/akaszynski/vtkInterface',
                           
    keywords='vtk numpy plotting mesh',
                           
    package_data={'vtkInterface.examples': ['airplane.ply', 'ant.ply', 
                                            'hexbeam.vtk', 'sphere.ply']},

    install_requires=['numpy'],

)
