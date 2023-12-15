import numpy as np

def jaccard_similarity(signature_matrix, a, b):  # a and b are two columns of the sign_mat
    permutations = signature_matrix.shape[0]  # no. of rows
    intersection = len(np.intersect1d(signature_matrix[:, a], signature_matrix[:, b]))
    similarity = intersection / permutations
    return similarity