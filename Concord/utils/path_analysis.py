import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from .. import logger

try:
    import faiss
    faiss_available = True
except ImportError:
    faiss_available = False


def shortest_path_on_knn_graph(adata, emb_key='encoded', k=10, point_a=None, point_b=None, use_faiss=True, ):
    """
    Finds the shortest path between point_a and point_b on a k-nearest neighbor graph.
    If point_b is not provided, it finds the furthest point from point_a.

    Parameters:
    - adata: AnnData object containing the data in adata.obsm[emb_key]
    - point_a: Starting point index
    - k: Number of nearest neighbors to consider for the graph
    - point_b: Optional ending point index. If None, the furthest point from point_a is used.
    - use_faiss: Boolean to use faiss for nearest neighbor search if available
    - emb_key: Key to access the embeddings in adata.obsm

    Returns:
    - path: List of indices representing the shortest path from point_a to point_b
    """
    from scipy.sparse.csgraph import dijkstra
    X = adata.obsm[emb_key]
    d = X.shape[1]

    if use_faiss and faiss_available:
        index = faiss.IndexFlatL2(d)  # L2 distance index
        index.add(X)
        distances, indices = index.search(X, k + 1)
    else:
        nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(X)
        distances, indices = nbrs.kneighbors(X)

    rows = np.repeat(np.arange(X.shape[0]), k)
    cols = indices[:, 1:].flatten()  # Exclude self-loops
    weights = distances[:, 1:].flatten()

    graph = csr_matrix((weights, (rows, cols)), shape=(X.shape[0], X.shape[0]))

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
    data = adata.obsm[signal_key][path]
    # Smooth each column and find the peak location
    smoothed_data = smooth_matrix(data, sigma=sigma)
    smoothed_peak_locs = find_peak_locations(smoothed_data)
    sorted_columns = np.argsort(smoothed_peak_locs)
    sorted_data = data[:, sorted_columns]
    sorted_smoothed_data = smoothed_data[:, sorted_columns]

    return sorted_data, sorted_smoothed_data, smoothed_peak_locs, sorted_columns


def geodesic_distance_along_path(adata, emb_key=None, path=None):
    gd_distances = []
    data = adata.obsm[emb_key][path]
    for i in range(len(path) - 1):
        vec1 = data[i]
        vec2 = data[i + 1]
        dist = np.linalg.norm(vec1 - vec2)
        gd_distances.append(dist)

    geodesic_distances = np.cumsum([0] + gd_distances)
    return geodesic_distances