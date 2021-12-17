import numpy as np

def get_U_matrix(M):
    return np.concatenate((np.array([-np.ones(M-1)]), np.eye(M-1)))

def matprint(mat, fmt="g"):
    try:
        col_maxes = [max([len(("{:"+fmt+"}").format(x)) for x in col]) for col in mat.T]
        for x in mat:
            for i, y in enumerate(x):
                print(("{:"+str(col_maxes[i])+fmt+"}").format(y), end="  ")
            print("")
    except TypeError as e:
        print(mat)


def matwrite(mat, file, fmt="g"):
    with open(file, 'w') as fp:
        col_maxes = [max([len(("{:"+fmt+"}").format(x)) for x in col]) for col in mat.T]
        for x in mat:
            for i, y in enumerate(x):
                fp.write(("{:"+str(col_maxes[i])+fmt+"}  ").format(y))
            fp.write("\n")


def frobenius_norm_2(matrix):
    '''
    Get matrix frobenius norm squared, so it works with cvxpy
    '''
    # TODO make this good
    accum = 0
    for row in matrix:
        for column in row:
            accum += column*column
    return accum

