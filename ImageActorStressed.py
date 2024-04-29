import pyvista as pv

image = pv.ImageEllipsoidSource()
image.output.plot(cpos="xy")

shifted_scaled = image.output.shift_scale(shift=100, scale=1)
shifted_scaled.plot(cpos="xy")
