# We are going to make a script to run Grotas algorithm over a network, and check how closely did it estimate it

# Create the network
import pandapower
import pandapower.networks
import numpy as np
import scipy.linalg
import cvxpy as cp
import matplotlib.pyplot
from NetworkMatrix import get_b_matrix_from_network, IEEE14_b_matrix
from utils import matprint, get_U_matrix
from simulations import F_score, cramer_rao_bound

# TODO change all np.transpose to matrix.T

def MSE_matrix(matrix_real, matrix_est):
    diff_matrix = matrix_real - matrix_est
    return np.trace(diff_matrix @ diff_matrix.T)

def ML_symmetric_positive_definite_estimator(sigma_theta_tilde, sigma_p_tilde, sigma_noise_approx, U):
    U_pseudinv = np.linalg.pinv(U)

    # one way to get square root
    # TODO analize if the square root algorithm used here is useful
    # TODO get this from eigenvalues decomposition
    sigma_theta_tilde_sqrt = scipy.linalg.sqrtm(sigma_theta_tilde)

    # another way to get square root
    # TODO check if I whould always send hermitian
    # u_sigma, s_sigma, v_sigma = np.linalg.svd(sigma_theta_tilde,compute_uv=True, hermitian=True)

    sigma_theta_tilde_sqrt_inv = np.linalg.inv(sigma_theta_tilde_sqrt)

    aux = scipy.linalg.sqrtm(sigma_theta_tilde_sqrt @ (sigma_p_tilde - sigma_noise_approx**2 * U_pseudinv @ U_pseudinv.T ) @ sigma_theta_tilde_sqrt)
    B_estimation = sigma_theta_tilde_sqrt_inv @ aux @ sigma_theta_tilde_sqrt_inv

    print("B_estimation")
    matprint(B_estimation)
    
    return B_estimation

def two_phase_topology_recovery(N,M,U,sigma_theta_tilde, sigma_p, sigma_noise_approx):
    # Step 1) Get reduced sample covariance matrix
    sigma_p_tilde =  U.T @ sigma_p @ U 

    # Step 2) Get optimal solution
    B_estimated = ML_symmetric_positive_definite_estimator(sigma_theta_tilde, sigma_p_tilde, sigma_noise_approx, U)

    B = cp.Variable(shape=(M,M))
    # Condition 1, positive semidefinite
    constraints = [B >> 0]
    for m in range(M):
        for k in range(M):
            if k < m:
                # condition 2
                constraints.append(B[m,k] <= 0)
            # if k != m and k > m:
            #     # Simetric condition
            #     constraints.append(B[m,k] == B[k,m])

    for m in range(M):
        # condition 3
        constraints.append(sum(B[m,:]) == 0)
    B_target = np.real(U @ B_estimated @ U.T)
    problem = cp.Problem(cp.Minimize(cp.sum_squares(B_target - B)), constraints=constraints)
    problem.solve()

    print("B_target")
    matprint(B_target)
    print("B")
    matprint(B.value)

    return B.value

def augmented_lagrangian_topology_recovery(N,M,U,sigma_theta_tilde, sigma_p, sigma_noise_approx):
    # Step 1) Get reduced sample covariance matrix (same as in two phase)
    sigma_p_tilde =  np.transpose(U) @ sigma_p @ U 
    U_pseudinv = np.linalg.pinv(U)

    # Step 2) Initialize B 
    # We use the B_PD from two phase
    B_estimated = ML_symmetric_positive_definite_estimator(sigma_theta_tilde, sigma_p_tilde, sigma_noise_approx, U)

    t = 0 # iteration

    # lagrangian multipliers
    mu = np.zeros((M-1, 1))  
    big_lambda = np.zeros(B_estimated.shape)
    big_gamma = np.zeros(B_estimated.shape)

    # other parameters
    gamma = 0.01 # Penalty parameter
    nabla = 0.02 # Learning rate

    W = np.linalg.inv(B_estimated)
    W_inv = B_estimated

    criterion_reached = False
    sigma_theta_tilde_inv = np.linalg.inv(sigma_theta_tilde)

    epsilon = 1
    while not criterion_reached:
        # update big gamma
        big_gamma = big_gamma - gamma * (W - W.T)

        # update big lambda
        W_off = W.copy()
        np.fill_diagonal(W_off,0)
        W_off_inv = np.linalg.inv(W_off)
        big_lambda = big_lambda + gamma * W_off_inv
        big_lambda = np.maximum(big_lambda, np.zeros(big_lambda.shape))

        # update mu
        mu = mu - gamma * W_inv @ np.ones((M-1,1))
        mu = np.maximum(mu, np.zeros(mu.shape))

        # equation 29
        aux1 = -(sigma_p_tilde - sigma_noise_approx**2 * U_pseudinv @ U_pseudinv.T) @ W_inv @ sigma_theta_tilde_inv
        aux2 = W.T @ (big_gamma.T - big_gamma) @ W.T
        W_next = W - nabla * (aux1 - W.T - aux2 - big_lambda + np.ones((M-1,1)).T @ mu)

        W_inv = np.linalg.inv(W_next)

        if np.linalg.norm(W_next - W) < epsilon:
            criterion_reached = True

        W = W_next
        t += 1

    return W_inv

def GrotasAlgorithm(observations, state_covariance_matrix, method='two_phase_topology'):
    # state_covariance_matrix is E_theta_tilde
    M = observations.shape[1]
    N = observations.shape[0]
    U = get_U_matrix(M)
    state_covariance_matrix_tilde = U.T @ state_covariance_matrix @ U

    def step_1(observations, M, N):
        # Step 1) Remove sample mean of all observations
        avg = np.zeros(M)
        for time in observations:
            avg += time
        avg = avg/N
        return observations - np.tile(np.array(avg),(N,1))
    
    def step_2(measurements, M, N):
        # Step 2) Get sample covariance matrix
        E_p = np.zeros((M,M))
        for measurement in measurements:
            meas_formatted = np.array([measurement])
            E_p += meas_formatted.T @ meas_formatted
        return E_p / N

    def step_3_4(E_p):
        # Step 3-4) Get eigenvalues and estimate noise variance with the smallest eigenvalue
        eigenvalues = np.linalg.eig(E_p)[0]
        sigma_sqr_noise_approx = eigenvalues.min()
        return np.sqrt(sigma_sqr_noise_approx)

    def step_5_6(N,M,E_theta_tilde,E_p,sigma_noise_approx, method):
        U = get_U_matrix(M)
        # Step 5) Get \hat(\tilde(B)). The paper explains two methods to approximate maximum likelihood B
        if method == 'two_phase_topology':
            B_approx = two_phase_topology_recovery(N,M,U,E_theta_tilde,E_p,sigma_noise_approx)
        else:
            B_tilde_approx = augmented_lagrangian_topology_recovery(N,M,U,E_theta_tilde,E_p,sigma_noise_approx)
            # Step 6) Get \hat(B) from \hat(\tilde(B))
            B_approx = U @ B_tilde_approx @ U.T
        return B_approx
    
    def step_7(M, B_approx):
        # Step 7) Impose sparsity with a threshold
        sparsity_value = 2/M
        threshold = np.diag(B_approx).min() * sparsity_value
        return np.where(np.abs(B_approx) < threshold, 0, B_approx)

    def step_8(B, E_theta, sigma_noise_approx, measurements, M):
        theta_approx = np.zeros_like(measurements)
        for i in range(0, measurements.shape[0]):
            theta_approx[i,:] = E_theta @ B @ np.linalg.pinv(B.T @ E_theta @ B + sigma_noise_approx**2 * np.eye(M)) @ measurements[i]
        return theta_approx

    observations_no_mean = step_1(observations, M, N)
    sample_covariance_matrix = step_2(observations_no_mean, M, N) # aka E_p
    sigma_noise_estimation = step_3_4(sample_covariance_matrix)
    B_estimation = step_5_6(N, M, state_covariance_matrix_tilde, sample_covariance_matrix, sigma_noise_estimation, method)
    B_estimation = step_7(M, B_estimation)
    theta_estimation = step_8(B_estimation, state_covariance_matrix, sigma_noise_estimation, observations_no_mean, M)
    return B_estimation, theta_estimation, sigma_noise_estimation, sample_covariance_matrix

