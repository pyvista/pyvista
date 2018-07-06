import vtkInterface as vtki
from vtkInterface import examples

def RunningXServer():
    """ Check if x server is running """
    if os.name != 'posix':  # linux or mac os
        raise Exception('Can only check x server on POSIX')
    p = Popen(["xset", "-q"], stdout=PIPE, stderr=PIPE)
    p.communicate()
    return p.returncode == 0

# we're getting there
# def test_offscreen():
#     sphere = examples.LoadSphere()
#     sphere.Plot(off_screen=True)
