import numpy as np

def cosine_sim(matrix):
    # L2 normalize each row of the matrix
    row_norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    l2_normalized_matrix = matrix / row_norms
    
    # Compute the dot product between rows (cosine similarity matrix)
    dot_product_matrix = np.dot(l2_normalized_matrix, l2_normalized_matrix.T)
    
    return dot_product_matrix