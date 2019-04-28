"""
Decimation
~~~~~~~~~~

Decaimate a mesh

"""
# sphinx_gallery_thumbnail_number = 1
import vtki
from vtki import examples

mesh = examples.download_face()

# Define a camera potion the shows this mesh properly
cpos = [(0.4,-0.07,-0.31), (0.05,-0.13,-0.06), (-.1,1,0.08)]

# Preview the mesh
mesh.plot(cpos=cpos, show_edges=True, color=True)

###############################################################################
#  Now let's define a target reduction and compare the
# :func:`vtki.PolyData.decimate` and :func:`vtki.PolyData.decimate_pro` filters.
target_reduction = 0.7
print('Reducing {} percent out of the original mesh'.format(target_reduction * 100.))

###############################################################################
decimated = mesh.decimate(target_reduction)

decimated.plot(cpos=cpos, show_edges=True)


###############################################################################
pro_decimated = mesh.decimate_pro(target_reduction, preserve_topology=True)

pro_decimated.plot(cpos=cpos, show_edges=True, color=True)
