import sys, os
import vtkCommonCorePython

def vtkLoadPythonTkWidgets(interp):
    """vtkLoadPythonTkWidgets(interp) -- load vtk-tk widget extensions

    This is a mess of mixed python and tcl code that searches for the
    shared object file that contains the python-vtk-tk widgets.  Both
    the python path and the tcl path are searched.
    """
    X = vtkCommonCorePython.vtkVersion.GetVTKMajorVersion()
    Y = vtkCommonCorePython.vtkVersion.GetVTKMinorVersion()
    modname = 'vtkRenderingPythonTkWidgets'
    name = '%s-%d.%d' % (modname,X,Y)
    pkgname = modname.lower().capitalize()

    # find out if the module is already loaded
    loadedpkgs = interp.call('info', 'loaded')
    found = False
    try:
        # check for result returned as a string
        found = (loadedpkgs.find(pkgname) >= 0)
    except AttributeError:
        # check for result returned as nested tuples
        for pkgtuple in loadedpkgs:
            found |= (pkgname in pkgtuple)
    if found:
        return

    # create the platform-dependent file name
    prefix = ''
    if sys.platform == 'cygwin':
        prefix = 'cyg'
    elif os.name == 'posix':
        prefix = 'lib'
    extension = interp.call('info', 'sharedlibextension')
    filename = prefix+name+extension

    # create an extensive list of paths to search
    pathlist = sys.path
    # add tcl paths, ensure that {} is handled properly
    try:
        auto_paths = interp.getvar('auto_path').split()
    except AttributeError:
        auto_paths = interp.getvar('auto_path')
    for path in auto_paths:
        prev = str(pathlist[-1])
        try:
            # try block needed when one uses Gordon McMillan's Python
            # Installer.
            if len(prev) > 0 and prev[0] == '{' and prev[-1] != '}':
                pathlist[-1] = prev+' '+path
            else:
                pathlist.append(path)
        except AttributeError:
            pass
    # a common place for these sorts of things
    if os.name == 'posix':
        pathlist.append('/usr/local/lib')

    # if python 3, there is no separate "unicode" type
    if sys.hexversion >= 0x03000000:
        unicode = str
    else:
        unicode = sys.modules['__builtin__'].unicode

    # attempt to load
    for path in pathlist:
        try:
            # If the path object is not str, it means that it is a
            # Tkinter path object.
            if (not isinstance(path, str) and not isinstance(path, unicode)):
                path = path.string
            # try block needed when one uses Gordon McMillan's Python
            # Installer.
            if len(path) > 0 and path[0] == '{' and path[-1] == '}':
                path = path[1:-1]
            fullpath = os.path.join(path, filename)
        except AttributeError:
            pass
        if ' ' in fullpath:
            fullpath = '{'+fullpath+'}'
        if interp.eval('catch {load '+fullpath+' '+pkgname+'}') == '0':
            return

    # re-generate the error
    interp.call('load', filename, pkgname)
