import numpy as np
import simulations

def test_stack_matrix():
    matrix = np.array([[1,2,3],[4,5,6],[7,8,9]])

    vector = simulations.stack_matrix(matrix)

    stacked_vector = np.array([[1,4,7,2,5,8,3,6,9]])
    assert not (vector-stacked_vector).any()

def test_f_score_is_1():
    matrix1 = np.array([[1,2,3],[4,5,6],[7,8,9]])
    matrix2 = np.array([[1,2,3],[4,5,6],[7,8,9]])
    f_score = simulations.F_score(matrix1, matrix2)

    assert f_score == 1

def test_f_score_is_0():
    matrix1 = np.array([[0,1,0],[1,0,0],[0,0,0]])
    matrix2 = np.array([[0,0,1],[0,0,0],[1,0,0]])
    print(matrix1)
    print(matrix2)
    f_score = simulations.F_score(matrix1, matrix2)

    assert f_score == 0

def test_f_score_is_two_thirds():
    matrix1 = np.array([[0,1,0],[1,0,0],[0,0,0]])
    matrix2 = np.array([[0,1,1],[1,0,0],[1,0,0]])
    print(matrix1)
    print(matrix2)
    f_score = simulations.F_score(matrix1, matrix2)

    assert f_score == 2/3
