import numpy as np

def matprint(mat, fmt="g"):
    try:
        col_maxes = [max([len(("{:"+fmt+"}").format(x)) for x in col]) for col in mat.T]
        for x in mat:
            for i, y in enumerate(x):
                print(("{:"+str(col_maxes[i])+fmt+"}").format(y), end="  ")
            print("")
    except TypeError as e:
        print(mat)


def F_score(mat1, mat2):
    '''
    This function measures the F-Score between these two matrixes
    mat1 is the tested one
    mat2 is the real one
    '''
    M = mat1.shape[0]
    # This results in the adjacency matrix, but with numbers in the diagonal
    adj_mat1 = (mat1 != 0) * 1
    np.fill_diagonal(adj_mat1, 0)
    # This results in the adjacency matrix, but with numbers in the diagonal
    adj_mat2 = (mat2 != 0) * 1
    np.fill_diagonal(adj_mat2, 0)

    # print('mata')
    # matprint(adj_mat1)
    # print('matb')
    # matprint(adj_mat2)

    positive_mask1 = (adj_mat1 != 1) * 2 + adj_mat1
    positive_mask2 = (adj_mat2 != 1) * 3 + adj_mat2
    true_positives = (positive_mask1 == positive_mask2) * 1
    tp = np.sum(true_positives)

    negative_mask2 = (adj_mat2 != 1) * 1
    np.fill_diagonal(negative_mask2, 0)
    false_positives = (positive_mask1 == negative_mask2) * 1
    fp = np.sum(false_positives)

    negative_mask1 = (adj_mat1 != 1) * 1
    np.fill_diagonal(negative_mask1, 0)
    false_negatives = (negative_mask1 == positive_mask2) * 1
    fn = np.sum(false_negatives)

    return 2 * tp / (2 * tp + fp + fn)
