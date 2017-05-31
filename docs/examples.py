"""
associated with examples.rst
"""
import vtkInterface
from vtkInterface import examples
import numpy as np


#==============================================================================
# Plane screenshot
#==============================================================================

filename = examples.planefile
mesh = vtkInterface.LoadMesh(filename)
cpos = mesh.Plot(screenshot='airplane.png', color='yellow')

#==============================================================================
# Make wave gif
#==============================================================================

# Make data
X = np.arange(-10, 10, 0.25)
Y = np.arange(-10, 10, 0.25)
X, Y = np.meshgrid(X, Y)
R = np.sqrt(X**2 + Y**2)
Z = np.sin(R)

# Create and plot structured grid
sgrid = vtkInterface.GenStructSurf(X, Y, Z)

# Make deep copy of points
pts = sgrid.GetNumpyPoints(deep=True)

# Start a plotter object and set the scalars to the Z height
plobj = vtkInterface.PlotClass()
plobj.AddMesh(sgrid, scalars=Z.ravel())
plobj.Plot(autoclose=False)

# Open a gif
plobj.OpenGif('wave.gif')

# Update Z and write a frame for each updated position
nframe = 15
for phase in np.linspace(0, 2*np.pi, nframe + 1)[:nframe]:
    Z = np.sin(R + phase)
    pts[:, -1] = Z.ravel()
    plobj.UpdateCoordinates(pts)
    plobj.UpdateScalars(Z.ravel())

    plobj.WriteFrame()

# Close movie and delete object
plobj.Close()
del plobj


#==============================================================================
# Wave curvature example
#==============================================================================
# Make data
X = np.arange(-10, 10, 0.25)
Y = np.arange(-10, 10, 0.25)
X, Y = np.meshgrid(X, Y)
R = np.sqrt(X**2 + Y**2)
Z = np.sin(R)

# Create and plot structured grid
surf = vtkInterface.GenStructSurf(X, Y, Z)
#sgrid.Plot()

# Extract surface of structured grid and plot mean curvature
#surf = sgrid.ExtractSurface()
surf.PlotCurvature('Mean')
surf.GetNumpyPoints()


# Take screenshot example
c = vtkInterface.GetCurvature(surf)

# Create plotting class
plobj = vtkInterface.PlotClass()

# add a surface and plot
plobj.AddMesh(surf, scalars=c, stitle='Mean\nCurvature')
plobj.Plot(autoclose=False)

# take a screenshot and close the window when the user presses "q"
plobj.TakeScreenShot('curvature.png')
plobj.Close()
del plobj

#==============================================================================
# 
#==============================================================================
