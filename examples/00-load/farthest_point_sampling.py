import pyvista as pv
import numpy as np

def farthest_point_sampling(points, k):
    """
    Perform Farthest Point Sampling on a set of points.

    Parameters
    ----------
    points : np.ndarray
        Input points as a numpy array of shape (n, 3).
    k : int
        Number of points to sample.

    Returns
    -------
    sampled_points : np.ndarray
        Sampled points as a numpy array of shape (k, 3).
    """
    n_points = points.shape[0]
    sampled_indices = [np.random.randint(n_points)]
    distances = np.full(n_points, np.inf)

    for _ in range(1, k):
        last_sampled = points[sampled_indices[-1]]
        dist_to_last = np.linalg.norm(points - last_sampled, axis=1)
        distances = np.minimum(distances, dist_to_last)
        next_index = np.argmax(distances)
        sampled_indices.append(next_index)

    return points[sampled_indices]

# Generate random points for demonstration
n_points = 1000
points = np.random.rand(n_points, 3)

# Perform Farthest Point Sampling
k = 10
sampled_points = farthest_point_sampling(points, k)

# Visualize the result
cloud = pv.PolyData(points)
sampled_cloud = pv.PolyData(sampled_points)

plotter = pv.Plotter()
plotter.add_mesh(cloud, color="blue", point_size=5, render_points_as_spheres=True, label="Original Points")
plotter.add_mesh(sampled_cloud, color="red", point_size=10, render_points_as_spheres=True, label="Sampled Points")
plotter.add_legend()
plotter.show()