import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from .. import logger

try:
    import faiss
    faiss_available = True
except ImportError:
    faiss_available = False

    

def shortest_path_on_knn_graph(neighborhood=None, adata=None, basis='encoded', k=10, point_a=None, point_b=None, use_faiss=True, use_ivf=False, ivf_nprobe=10, metric='euclidean'):
    from scipy.sparse import csr_matrix
    from ..model.knn import Neighborhood
    from scipy.sparse.csgraph import dijkstra

    # Initialize the Neighborhood class
    if neighborhood is None:
        neighborhood = Neighborhood(adata.obsm[basis], k=k, use_faiss=use_faiss, use_ivf=use_ivf, ivf_nprobe=ivf_nprobe, metric=metric)

    # Build the adjacency matrix
    graph = neighborhood.get_knn_graph()

    # Compute the shortest path
    dist_matrix, predecessors = dijkstra(csgraph=graph, directed=False, indices=[point_a], return_predecessors=True)

    # Determine point_b if not provided
    if point_b is None:
        point_b = np.argmax(dist_matrix[0])
        logger.info(f"Finding path between point_{point_a} and furthest point_{point_b} from point_{point_a}")
    else:
        logger.info(f"Finding path between point_{point_a} and point_{point_b}")

    # Trace the path from point_b to point_a
    path = []
    i = point_b
    while i != point_a:
        path.append(i)
        i = predecessors[0, i]
    path.append(point_a)
    path = path[::-1]

    return path, dist_matrix


def smooth_matrix(matrix, sigma=2):
    from scipy.ndimage import gaussian_filter1d
    smoothed_matrix = np.copy(matrix)
    for col in range(matrix.shape[1]):
        smoothed_matrix[:, col] = gaussian_filter1d(matrix[:, col], sigma=sigma)
    return smoothed_matrix

# Find peak location function
def find_peak_locations(smoothed_matrix):
    peak_locations = []
    for col in range(smoothed_matrix.shape[1]):
        peak_location = np.argmax(smoothed_matrix[:, col])
        peak_locations.append(peak_location)
    return peak_locations

def sort_and_smooth_signal_along_path(adata, signal_key=None, path=None, sigma=2):
    if path is None:
        path = np.arange(adata.n_obs)
        
    data = adata.obsm[signal_key][path]
    # Smooth each column and find the peak location
    smoothed_data = smooth_matrix(data, sigma=sigma)
    smoothed_peak_locs = find_peak_locations(smoothed_data)
    sorted_columns = np.argsort(smoothed_peak_locs)
    sorted_data = data[:, sorted_columns]
    sorted_smoothed_data = smoothed_data[:, sorted_columns]

    return sorted_data, sorted_smoothed_data, smoothed_peak_locs, sorted_columns


def geodesic_distance_along_path(adata, basis=None, path=None):
    gd_distances = []
    data = adata.obsm[basis][path]
    for i in range(len(path) - 1):
        vec1 = data[i]
        vec2 = data[i + 1]
        dist = np.linalg.norm(vec1 - vec2)
        gd_distances.append(dist)

    geodesic_distances = np.cumsum([0] + gd_distances)
    return geodesic_distances


def euclidian_distance_across_path(adata, basis=None, path=None):
    data = adata.obsm[basis][path]
    # Compute distance between start and end points
    start_point = data[0]
    end_point = data[-1]
    euclidian_distance = np.linalg.norm(start_point - end_point)
    return euclidian_distance


def curvature_along_path(adata, basis=None, path=None):
    data = adata.obsm[basis][path]
    # Compute distance between start and end points
    start_point = data[0]
    end_point = data[-1]
    euclidian_distance = np.linalg.norm(start_point - end_point)
    # Compute geodesic distance
    geodesic_distances = geodesic_distance_along_path(adata, basis=basis, path=path)
    # Compute curvature
    curvature = geodesic_distances[-1] / euclidian_distance
    return curvature


def curvatures_across_time(adata, basis=None, k=30, time_key=None, time_interval_frac=0.05, metric='euclidean'):
    from ..model.knn import Neighborhood
    import pandas as pd

    if time_key is None:
        time_key = "time"

    time_vec = adata.obs[time_key]
    time_interval = time_interval_frac * (time_vec.max() - time_vec.min())
    time_points = np.arange(time_vec.min(), time_vec.max(), time_interval)

    logger.info(f"Computing curvatures at time points: {time_points}")
    neighborhood = Neighborhood(adata.obsm[basis], k=k, metric=metric)

    result_df = pd.DataFrame()
    for t in time_points:
        start_point_index = np.argmin(np.abs(time_vec - t))
        end_point_index = np.argmin(np.abs(time_vec - (t + time_interval)))
        shortest_path, _ = shortest_path_on_knn_graph(neighborhood, point_a=start_point_index, point_b=end_point_index)
        if len(shortest_path) == 0:
            continue

        curvature = curvature_along_path(adata, basis=basis, path=shortest_path)
        # Record start, end points, corresponding time and curvature
        start_time = time_vec[start_point_index]
        end_time = time_vec[end_point_index]
        mid_time = (start_time + end_time) / 2
        result_df = pd.concat([result_df, pd.DataFrame([{
            "start_point": start_point_index, 
            "end_point": end_point_index,
            "start_time": start_time,
            "end_time": end_time,
            "mid_time": mid_time,
            "curvature": curvature
        }])])
    
    return result_df



# Function to project a point onto a line segment
def project_point_onto_segment(point, seg_start, seg_end):
    seg_vector = seg_end - seg_start
    point_vector = point - seg_start
    proj_length = np.dot(point_vector, seg_vector) / np.linalg.norm(seg_vector)**2
    proj_length = np.clip(proj_length, 0, 1)  # Clip to segment bounds
    return seg_start + proj_length * seg_vector, proj_length
    

def compute_pseudotime_from_shortest_path(adata, path, basis, pseudotime_key="pseudotime_SP"):
    """
    Compute pseudotime for each cell by projecting it onto the shortest path in a given embedding.

    Parameters
    ----------
    adata : AnnData
        The AnnData object containing the high-dimensional data and embedding.
    path : list of int
        Indices of cells defining the shortest path.
    basis : str
        The key in `adata.obsm` where the embedding is stored (e.g., "X_pca" or "X_umap").
    pseudotime_key : str, optional
        The key to store the computed pseudotime in `adata.obs`. Default is "pseudotime_SP".

    Returns
    -------
    None
        Modifies `adata.obs` in-place to include the computed pseudotime.
    """
    # Extract the coordinates of the shortest path
    path_coords = adata.obsm[basis][path]

    # Compute cumulative pseudotime along the path
    path_distances = [0] + [np.linalg.norm(path_coords[i + 1] - path_coords[i])
                            for i in range(len(path_coords) - 1)]
    cumulative_pseudotime = np.cumsum(path_distances)

    # Project each cell onto the path
    pseudotimes = []
    for point in adata.obsm[basis]:
        min_dist = np.inf
        best_t = 0
        for i in range(len(path_coords) - 1):
            # Project point onto the segment
            seg_start, seg_end = path_coords[i], path_coords[i + 1]
            proj, t = project_point_onto_segment(point, seg_start, seg_end)
            dist = np.linalg.norm(point - proj)
            if dist < min_dist:
                min_dist = dist
                # Interpolate pseudotime based on the segment
                best_t = cumulative_pseudotime[i] + t * (cumulative_pseudotime[i + 1] - cumulative_pseudotime[i])
        pseudotimes.append(best_t)

    # Normalize pseudotime to the range [0, 1]
    pseudotimes = np.array(pseudotimes)
    pseudotimes = (pseudotimes - pseudotimes.min()) / (pseudotimes.max() - pseudotimes.min())

    # Assign pseudotime to AnnData
    adata.obs[pseudotime_key] = pseudotimes

    return pseudotimes
