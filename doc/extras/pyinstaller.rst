.. _ref_pyinstaller:

Freezing PyVista with pyinstaller
=================================
You can make some fantastic standalone programs with ``pyinstaller``
and ``pyvista``, and you can even make a graphical user interface
incorporating ``PyQt5`` or ``pyside2``.  Depending on your version of
VTK, this requires some extra steps to setup.

When running VTK v9, you need to add several additional
``hiddenimports``.  For clarity and completeness, create a spec file
(we'll name it ``pyvista.spec``) following the directions given at
`Using Spec Files <https://pyinstaller.readthedocs.io/en/stable/spec-files.html>`__.  Modify the ``Analysis`` and add the following hidden imports:

.. code:: python

    main_py = os.path.join(some_path, 'main.py')
    a = Analysis([main_py],
                 pathex=[],
                 binaries=[],
                 hiddenimports=['vtkmodules',
                                'vtkmodules.all',
                                'vtkmodules.qt.QVTKRenderWindowInteractor',
                                'vtkmodules.util',
                                'vtkmodules.util.numpy_support',
                                'vtkmodules.numpy_interface.dataset_adapter',
                               ],

From there, you can freeze an application using ``pyvista`` and create
a standalone application.
