def results(correct_matrix, predicted_matrix):
    num_tvs = len(correct_matrix[0])
    TN = 0
    TP = 0
    FN = 0
    FP = 0

    for i in range(num_tvs):
        for j in range(i + 1, num_tvs):
            if correct_matrix[i, j] == 0 and predicted_matrix[i, j] == 0:
                TN += 1
            if correct_matrix[i, j] == 1 and predicted_matrix[i, j] == 1:
                TP += 1
            if correct_matrix[i, j] == 1 and predicted_matrix[i, j] == 0:
                FN += 1
            if correct_matrix[i, j] == 0 and predicted_matrix[i, j] == 1:
                FP += 1

    pq = TP/(TP+FP)  # precision
    pc = TP/(TP+FN)  # recall
    f1 = (2 * pq * pc)/(pc + pq)
    frac_comp = (TP+FP)/(TN+TP+FN+FP)

    return [pq, pc, f1, frac_comp]
