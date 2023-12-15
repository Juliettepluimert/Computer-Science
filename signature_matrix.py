import numpy as np

def signature_matrix(character_matrix, k):
    num_title_words = character_matrix.shape[0] # no. of rows
    num_titles = character_matrix.shape[1] # no. of columns

    signature_matrix = np.zeros((k, num_titles))

    for permutation in range(k):
        print(permutation)
        perm = np.random.permutation(num_title_words)
        for tv in range(num_titles):
            indices = np.atleast_1d(np.nonzero(character_matrix[:, tv] == 1)[0])
            if len(indices) > 0:
                signature_matrix[permutation, tv] = np.min(perm[indices])
    print("signature matrix done")

    return signature_matrix