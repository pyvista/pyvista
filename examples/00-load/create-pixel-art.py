"""
.._ref_geometric_example:

Pixel Art of SPACE INVADERS
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here we use :func:`pyvista.Box` to make pixel art.
https://en.wikipedia.org/wiki/Pixel_art
Source of characters:
https://en.wikipedia.org/wiki/Space_Invaders
"""
import pyvista as pv
from pyvista.demos import logo

###############################################################################
# Generate Piexls of each characters
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# You can define the pixel of INVADERS

# SQUID
#
#           % %
#         % % % %
#       % % % % % %
#     % %   % %   % %
#     % % % % % % % %
#         %     %
#       %   % %   %
#     %   %     %   %
#
squid = [
    [False, False, False, False, False, False, False, False, False, False],
    [False, False, False, False, True , True , False, False, False, False],
    [False, False, False, True , True , True , True , False, False, False],
    [False, False, True , True , True , True , True , True , False, False],
    [False, True , True , False, True , True , False, True , True , False],
    [False, True , True , True , True , True , True , True , True , False],
    [False, False, False, True , False, False, True , False, False, False],
    [False, False, True , False, True , True , False, True , False, False],
    [False, True , False, True , False, False, True , False, True , False],
    [False, False, False, False, False, False, False, False, False, False],
]

# CLAB
#
#       %         %
#         %     %
#       % % % % % %
#     % %   % %   % %
#   % % % % % % % % % %
#   %   % % % % % %   %
#   %   %         %   %
#         %     %


clab = [
    [False, False, False, False, False, False, False, False, False, False],
    [False, False, True , False, False, False, False, True , False, False],
    [False, False, False, True , False, False, True , False, False, False],
    [False, False, True , True , True , True , True , True , False, False],
    [False, True , True , False, True , True , False, True , True , False],
    [True , True , True , True , True , True , True , True , True , True ],
    [True , False, True , True , True , True , True , True , False, True ],
    [True , False, True , False, False, False, False, True , False, True ],
    [False, False, False, True , False, False, True , False, False, False],
    [False, False, False, False, False, False, False, False, False, False],
]

# OCTOPUS
#
#         % % % %
#     % % % % % % % %
#   % % % % % % % % % %
#   % %     % %     % %
#   % % % % % % % % % %
#       % %     % %
#     %   % % % %   %
#   %                 %

octopus = [
    [False, False, False, False, False, False, False, False, False, False],
    [False, False, False, True , True , True , True , False, False, False],
    [False, True , True , True , True , True , True , True , True , False],
    [True , True , True , True , True , True , True , True , True , True ],
    [True , True , False, False, True , True , False, False, True , True ],
    [True , True , True , True , True , True , True , True , True , True ],
    [False, False, True , True , False, False, True , True , False, False],
    [False, True , False, True , True , True , True , False, True , False],
    [True , False, False, False, False, False, False, False, False, True ],
    [False, False, False, False, False, False, False, False, False, False],
]


###############################################################################
# Define function to draw pixels
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Define a helper - function to add pixel boxes to plotter


def draw_pixels(plotter, pixels, center, color):
    bounds = [
        center[0] - 1.0,
        center[0] + 1.0,
        center[1] - 1.0,
        center[1] + 1.0,
        -10.0,
        +10.0,
    ]
    for rows in pixels:
        for pixel in rows:
            if pixel == True:
                box = pv.Box(bounds=bounds)
                plotter.add_mesh(box, color=color)
            bounds[0] += 2.0
            bounds[1] += 2.0
        bounds[0] = center[0] - 1.0
        bounds[1] = center[0] + 1.0
        bounds[2] += -2.0
        bounds[3] += -2.0
    return plotter


###############################################################################
# Now that you can plot a pixel art of SPACE INVADERS.

# Display INVADERS
p = pv.Plotter()
p = draw_pixels(p, squid  , [-22.0,  20.0], "green")
p = draw_pixels(p, squid  , [  0.0,  20.0], "green")
p = draw_pixels(p, squid  , [ 22.0,  20.0], "green")
p = draw_pixels(p, clab   , [-22.0,   0.0], "blue" )
p = draw_pixels(p, clab   , [  0.0,   0.0], "blue" )
p = draw_pixels(p, clab   , [ 22.0,   0.0], "blue" )
p = draw_pixels(p, octopus, [-22.0, -20.0], "red"  )
p = draw_pixels(p, octopus, [  0.0, -20.0], "red"  )
p = draw_pixels(p, octopus, [ 22.0, -20.0], "red"  )

text = logo.text_3d("SPACE INVADERS", depth=10.0)
text.points *= 4.0
text.translate([-20.0, 20.0, 0.0])

p.add_mesh(text, color="yellow")
p.show(cpos="xy")
