import pyvista as pv
import numpy as np

# 片方のメッシュ作成
verts = np.array(
    [[757.36000143,-1228.43000143,1216.30000048],
     [7828.42000143,5842.62999857,1216.30000048],
     [481.58835676,-952.65835676,1216.30000048],
     [7552.64835676,6118.40164324,1216.30000048],
     [757.36000143,-1228.43000143,1633.69999978],
     [481.58835676,-952.65835676,1633.69999978],
     [7828.42000143,5842.62999857,1633.69999978],
     [7552.64835676,6118.40164324,1633.69999978]]
)
faces = np.array(
    [3,0,1,2,3,1,3,2,3,4,5,6,3,6,5,7,3,0,4,1,3,1,4,6,3,2,3,5,3,3,7,5,3,0,2,4,3,2,5,4,3,1,6,3,3,3,6,7]
)
A_mesh = pv.PolyData(verts, faces)

# もう片方のメッシュ作成
verts = np.array(
    [[4600., 2800.,  600.],
     [-400., 2800.,  600.],
     [4600., 5300.,  600.],
     [-400., 5300.,  600.],
     [4600., 2800., 1600.],
     [4600., 5300., 1600.],
     [-400., 2800., 1600.],
     [-400., 5300., 1600.]]
)
faces = np.array(
    [3,0,1,2,3,1,3,2,3,4,5,6,3,6,5,7,3,0,4,1,3,1,4,6,3,2,3,5,3,3,7,5,3,0,2,4,3,2,5,4,3,1,6,3,3,3,6,7]
)
B_mesh = pv.PolyData(verts, faces)

# 初期状態プロット
pl = pv.Plotter()
_ = pl.add_mesh(A_mesh, color='r', show_edges=True)
_ = pl.add_mesh(B_mesh, color='b', show_edges=True)
pl.show()

# Trueである
print(A_mesh.is_all_triangles())
print(B_mesh.is_all_triangles())

# 法線は描画されない
# A_mesh.plot_normals(mag=100, show_edges=True, show_mesh=True)
# B_mesh.plot_normals(mag=100, show_edges=True, show_mesh=True)

# 共通部分取得
intersection1 = A_mesh.boolean_intersection(B_mesh)
# intersection2 = B_mesh.boolean_intersection(A_mesh)
# result4 = B_mesh.boolean_difference(A_mesh)
# result5 = B_mesh.boolean_union(A_mesh)

# 期待とは異なるplotたち
intersection1.plot()
# intersection2.plot()
# result4.plot()
# result5.plot()

# pl = pv.Plotter()
# _ = pl.add_mesh(intersection1, color='r')
# pl.show()
