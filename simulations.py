import numpy as np
from utils import get_U_matrix, matprint
import scipy.linalg


def get_observations(N, SNR, c, B):
    M = B.shape[0]
    # sigma_theta = c**2*I_M
    # SNR = 10 log(1/sigma**2 * Tr{B_tilde * sigma_theta_tilde * B_tilde})
    # sigma ** 2 = Tr{B_tilde * sigma_theta_tilde * B_tilde} / 10 ^ (SNR/10)
    U = get_U_matrix(M)
    sigma_theta = c**2 * np.eye(M)
    sigma_theta_tilde = U.T @ sigma_theta @ U
    B_tilde = B[1:M, 1:M]

    noise_sigma = np.trace(B_tilde @ sigma_theta_tilde @
                           B_tilde) / (10**(SNR/10))

    # TODO send distribution as param
    theta_created = np.random.default_rng().normal(0, c, (M, N))
    noise = np.random.default_rng().normal(0, noise_sigma, (M, N))
    states = B @ theta_created
    return ((states) + noise).T, sigma_theta, states, noise_sigma


def stack_matrix(matrix):
    out_vector = np.array([])
    for column in matrix.T:
        out_vector = np.append(out_vector, column)
    return out_vector


def oracle_mmse_state_estimation(B, sigma_theta, noise_variance, measurements):
    # theta values estimation from equation 11
    M = B.shape[0]
    return sigma_theta @ B @ np.pinv(B.T @ sigma_theta @ B + noise_variance @ np.eye(M)) @ measurements


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


def cramer_rao_bound(M, B_til_est, sigma_sqr, sigma_p, sigma_theta_tilde, N):
    def psi_vector(M, k, l):
        constant = 1 if k == l else 0.5
        ek = el = np.zeros((M - 1, 1))
        ek[k, 0] = 1
        el[l, 0] = 1
        # Ek,l = ek * el.T + el * ek.T
        # TODO check if there is a numpy function for stacking
        stack = stack_matrix(ek @ el.T + el @ ek.T)
        return constant * stack

    # B_til_est is a square matrix R^(M-1,M-1)
    alpha = np.zeros((M * (M - 1) // 2 + 1))
    filled = 0
    for i in range(0, M-1):
        alpha[filled:filled+(M-i-1)] = B_til_est[i:M, i]
        filled += (M-i-1)
    alpha[M*(M-1)//2] = sigma_sqr

    U = get_U_matrix(M)
    sigma_p_tilde = U.T @ sigma_p @ U
    sigma_p_tilde_inv = np.linalg.inv(sigma_p_tilde)
    Q = np.kron(sigma_p_tilde_inv, sigma_p_tilde_inv)
    K = np.kron(B_til_est @ sigma_theta_tilde, np.eye(M-1)) + \
        np.kron(np.eye(M-1), B_til_est @ sigma_theta_tilde)
    K_inv = np.linalg.inv(K)

    U_pseudinv = np.linalg.pinv(U)
    final_col = K_inv @ stack_matrix(U_pseudinv @ U_pseudinv.T)

    Psi = np.zeros(((M-1)**2, M*(M-1)//2 + 1))
    next_col = 0
    for k in range(0, M-1):
        for l in range(k, M - 1):
            Psi[:, next_col] = psi_vector(M, k, l)
            next_col += 1
    Psi[:, M * (M - 1) // 2] = final_col
    return np.trace(np.linalg.pinv(Psi.T @ K.T @ Q @ K @ Psi) * 2 / N)


def MSE_matrix(matrix_real, matrix_est):
    diff_matrix = matrix_real - matrix_est
    mse = np.trace(diff_matrix @ diff_matrix.T)
    if mse != np.real(mse):
        print(mse)
        mse = np.real(mse)
    return mse


def state_estimator(observations, B, sigma_theta, sigma_error):
    # Grotas eq. 12)
    """
    M = B.shape[0]
    sigma_w = sigma_error * np.eye(M)
    aux = (B.T @ sigma_theta @ B) + sigma_w
    # aux_pinv = np.linalg.inv(aux, hermitian=True)
    aux_pinv = np.linalg.inv(aux)
    # print(f"{sigma_error=}")
    # print(f"{np.linalg.norm(aux)=}")
    # print(f"{np.linalg.norm(aux_pinv)=}")
    return sigma_theta @ B @ aux_pinv @ observations.T
    """
    M = B.shape[0]
    sigma_theta_inv = np.linalg.inv(sigma_theta)
    sigma_w = sigma_error * np.eye(M)
    sigma_w_inv = np.linalg.inv(sigma_w)

    aux = sigma_theta_inv + B.T @ sigma_w_inv @ B
    aux_inv = np.linalg.inv(aux)
    # matprint(sigma_w_inv)
    return aux_inv @ B.T @ sigma_w_inv @ observations.T


def MSE_states(observations, B, sigma_theta, sigma_error, states):
    estimation = state_estimator(observations, B, sigma_theta, sigma_error)
    N = observations.shape[0]
    M = observations.shape[1]
    # print(estimation - states)
    ones = np.ones((M, 1))
    return np.sum(ones.T @ np.abs(estimation - states)) / (N )
    
    # return np.sum(np.abs(estimation[:,8] - states[:,8]))


def MSE_states_slow(observations, B, sigma_theta, sigma_error, states):
    N = observations.shape[0]
    M = observations.shape[1]

    error_total = 0
    for i in range(N):
        estimation = state_estimator(observations[i, :], B, sigma_theta, sigma_error)
        error = (states[:, i].T - estimation)**2
        error_total += sum(error) / M

    # print(error_total)
    return error_total / N


def MSE_states_theoretical(observations, B, sigma_theta, sigma_error, states):
    N = observations.shape[0]
    M = observations.shape[1]
    # From Kay eq. 12.28

    # B_inv = np.linalg.inv(B)
    sigma_w = sigma_error * np.eye(M)
    matrix_MSE = sigma_theta - sigma_theta @ B.T @ np.linalg.inv(B @ sigma_theta @ B.T + sigma_w) @ B @ sigma_theta
    return np.trace(matrix_MSE)
