"""
Installation file for python vtkInterface module
"""

from setuptools import setup


# Actual setup
setup(
    name='vtkInterface',
    packages = ['vtkInterface', 'vtkInterface.tests', 'vtkInterface.examples'],

    # Version
    version='0.1',

    description='Pythonic interface to VTK',
    long_description=open('README.rst').read(),

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
                                            'hexbeam.vtk']},

    install_requires=['numpy']

)
