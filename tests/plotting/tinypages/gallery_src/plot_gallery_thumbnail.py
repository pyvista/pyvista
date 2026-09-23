"""
Gallery Example With A Chosen Thumbnail
=======================================

Draw a rising line and then a falling one, and pick the second as the thumbnail.
"""

# sphinx_gallery_thumbnail_number = 2

import matplotlib.pyplot as plt

# %%
# The first image is only here for contrast with the chosen thumbnail.
plt.figure()
plt.plot([1, 2, 3])
plt.show()

# %%
# The second image is the one the gallery shows.
plt.figure()
plt.plot([3, 2, 1])
plt.show()
