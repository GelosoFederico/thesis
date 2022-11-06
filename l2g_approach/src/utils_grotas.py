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


def IEEE14_b_matrix():
    # values taken from https://github.com/thanever/pglib-opf/blob/master/pglib_opf_case14_ieee.m
    # so I can check if the other one is working correctly
    from_bus = [1,1,2,2,2,3,4,4,4,5,6,6,6,7,7,9,9,10,12,13]
    to_bus = [2,5,3,4,5,4,5,7,9,6,11,12,13,8,9,10,14,11,13,14]
    b = [0.05917,0.22304,0.19797,0.17632,0.17388,0.17103,0.04211,0.20912,0.55618,0.25202,0.1989,0.25581,0.13027,0.17615,0.11001,0.0845,0.27038,0.19207,0.19988,0.34802]
    M = 14
    return get_matrix_from_IEEE_format(from_bus, to_bus, b, M)


def IEEE57_b_matrix():
    from_bus = [1, 2, 3, 4, 4, 6, 6, 8, 9, 9, 9, 9, 13, 13, 1, 1, 1, 3, 4, 4, 5, 7, 10, 11, 12, 12, 12, 14, 18, 19, 21, 21, 22, 23, 24, 24, 24, 26, 27, 28, 7, 25, 30, 31, 32, 34, 34, 35, 36, 37, 37, 36, 22, 11, 41, 41, 38, 15, 14, 46, 47, 48, 49, 50, 10, 13, 29, 52, 53, 54, 11, 44, 40, 56, 56, 39, 57, 38, 38, 9]
    to_bus = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 15, 16, 17, 15, 18, 18, 6, 8, 12, 13, 13, 16, 17, 15, 19, 20, 20, 22, 23, 24, 25, 25, 26, 27, 28, 29, 29, 30, 31, 32, 33, 32, 35, 36, 37, 38, 39, 40, 38, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 51, 49, 52, 53, 54, 55, 43, 45, 56, 41, 42, 57, 56, 49, 48, 55]
    b = [0.028, 0.085, 0.0366, 0.132, 0.148, 0.102, 0.173, 0.0505, 0.1679, 0.0848, 0.295, 0.158, 0.0434, 0.0869, 0.091, 0.206, 0.108, 0.053, 0.555, 0.43, 0.0641, 0.0712, 0.1262, 0.0732, 0.058, 0.0813, 0.179, 0.0547, 0.685, 0.434, 0.7767, 0.117, 0.0152, 0.256, 1.182, 1.23, 0.0473, 0.254, 0.0954, 0.0587, 0.0648, 0.202, 0.497, 0.755, 0.036, 0.953, 0.078, 0.0537, 0.0366, 0.1009, 0.0379, 0.0466, 0.0295, 0.749, 0.352, 0.412, 0.0585, 0.1042, 0.0735, 0.068, 0.0233, 0.129, 0.128, 0.22, 0.0712, 0.191, 0.187, 0.0984, 0.232, 0.2265, 0.153, 0.1242, 1.195, 0.549, 0.354, 1.355, 0.26, 0.177, 0.0482, 0.1205]
    M = 57
    return get_matrix_from_IEEE_format(from_bus, to_bus, b, M)


def get_matrix_from_IEEE_format(from_bus, to_bus, b, M=None):
    if not M:
        M = max(max(from_bus), max(to_bus))
    B_real = np.zeros((M,M))
    A = np.zeros((M,M))
    for row in zip(from_bus, to_bus, b):
        m = row[0]-1
        k = row[1]-1
        if m != k:
            b_mk = row[2]
            B_real[m,k] = -b_mk
            B_real[k,m] = -b_mk
            B_real[m,m] += b_mk
            B_real[k,k] += b_mk
            A[m,k] = 1
            A[k,m] = 1
    return B_real, A
