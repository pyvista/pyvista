"""
.. _create_pixel_art_example:

Pixel Art of ALIEN MONSTERS
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here we use :func:`pyvista.Box` to make `pixel art <https://en.wikipedia.org/wiki/Pixel_art>`_.
Pixel string `source <https://commons.wikimedia.org/wiki/File:Noto_Emoji_Pie_1f47e.svg>`_
and `license <https://github.com/googlefonts/noto-emoji/blob/main/LICENSE>`_.

"""

from __future__ import annotations

import pyvista as pv
from pyvista.demos import logo

# sphinx_gallery_start_ignore
PYVISTA_GALLERY_FORCE_STATIC_IN_DOCUMENT = False
# sphinx_gallery_end_ignore

# %%
# Convert pixel art to an array
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


alien_str = """
    %         %
      %     %
    % % % % % %
  % %   % %   % %
% % % % % % % % % %
%   % % % % % %   %
%   %         %   %
%   % %     % %   %
      %     %
    %         %
"""


alien = []
for line in alien_str.splitlines()[1:]:  # skip first linebreak
    if not line:
        continue
    long_line = line + (20 - len(line)) * ' ' if len(line) < 20 else line
    alien.append([long_line[i : i + 2] == '% ' for i in range(0, len(long_line), 2)])


# %%
# Define function to draw pixels
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Define a helper function to add pixel boxes to plotter.


def draw_pixels(plotter, pixels, center, color):  # noqa: PLR0917
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
            if pixel:
                box = pv.Box(bounds=bounds)
                plotter.add_mesh(box, color=color)
            bounds[0] += 2.0
            bounds[1] += 2.0
        bounds[0] = center[0] - 1.0
        bounds[1] = center[0] + 1.0
        bounds[2] += -2.0
        bounds[3] += -2.0
    return plotter


# %%
# Now you can plot a pixel art of ALIEN MONSTERS.

# Display MONSTERS
p = pv.Plotter()
p = draw_pixels(p, alien, [-22.0, 22.0], 'green')
p = draw_pixels(p, alien, [0.0, 22.0], 'green')
p = draw_pixels(p, alien, [22.0, 22.0], 'green')
p = draw_pixels(p, alien, [-22.0, 0.0], 'blue')
p = draw_pixels(p, alien, [0.0, 0.0], 'blue')
p = draw_pixels(p, alien, [22.0, 0.0], 'blue')
p = draw_pixels(p, alien, [-22.0, -22.0], 'red')
p = draw_pixels(p, alien, [0.0, -22.0], 'red')
p = draw_pixels(p, alien, [22.0, -22.0], 'red')

text = logo.text_3d('ALIEN MONSTERS', depth=10.0)
text.points *= 4.0
text.translate([-20.0, 24.0, 0.0], inplace=True)

p.add_mesh(text, color='yellow')
p.show(cpos='xy')
# %%
# .. tags:: load
