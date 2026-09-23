"""
Gallery Example Without A Chosen Thumbnail
==========================================

Draw two lines without selecting a thumbnail, so the gallery uses the first.
"""

import matplotlib.pyplot as plt

# %%
# The first image is the gallery thumbnail by default.
plt.figure()
plt.plot([1, 4, 9])
plt.show()

# %%
# The second image is never used as a thumbnail.
plt.figure()
plt.plot([9, 4, 1])
plt.show()
