import numpy as np

def lsh(signature_matrix, num_bands):
    num_permutations = signature_matrix.shape[0]
    num_items = signature_matrix.shape[1]

    rows_per_band = num_permutations // num_bands
    candidate_pairs = np.zeros([num_items, num_items])

    for b in range(num_bands):
        print(b)
        for i in range(num_items):
            for j in range(i + 1, num_items):
                band_rows_i = signature_matrix[b * rows_per_band: (b + 1) * rows_per_band, i]

                band_rows_j = signature_matrix[b * rows_per_band: (b + 1) * rows_per_band, j]

                if np.array_equal(band_rows_i, band_rows_j):
                    candidate_pairs[i, j] = 1
    print("lsh done")

    return candidate_pairs