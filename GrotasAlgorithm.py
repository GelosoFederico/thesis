# We are going to make a script to run Grotas algorithm over a network, and check how closely did it estimate it

# Create the network
from re import T
import pandapower
import pandapower.networks
import numpy as np
import scipy.linalg
import cvxpy as cp

# TODO change all np.transpose to matrix.T

def frobenius_norm_2(matrix):
    '''
    Get matrix frobenius norm squared, so it works with cvxpy
    '''
    # TODO make this good
    # TODO validate it's a matrix and stuff
    accum = 0
    for row in matrix:
        for column in row:
            accum += column*column
    print(accum)
    return accum

def ML_symmetric_positive_definite_estimator(sigma_theta_tilde, sigma_p_tilde, sigma_noise_approx, U):
    U_pseudinv = np.linalg.pinv(U)

    # one way to get square root
    # TODO analize if the square root algorithm used here is useful
    sigma_theta_tilde_sqrt = scipy.linalg.sqrtm(sigma_theta_tilde)

    # another way to get square root
    # TODO check if I whould always send hermitian
    # u_sigma, s_sigma, v_sigma = np.linalg.svd(sigma_theta_tilde,compute_uv=True, hermitian=True)

    sigma_theta_tilde_sqrt_inv = np.linalg.inv(sigma_theta_tilde_sqrt)

    aux = sigma_theta_tilde_sqrt @ (sigma_p_tilde - sigma_noise_approx**2 * U_pseudinv @ U_pseudinv.T ) @ sigma_theta_tilde_sqrt
    return sigma_theta_tilde_sqrt_inv @ aux @ sigma_theta_tilde_sqrt_inv



def two_phase_topology_recovery(N,M,U,sigma_theta_tilde, sigma_p, sigma_noise_approx):
    # Step 1) Get reduced sample covariance matrix
    sigma_p_tilde =  np.transpose(U) @ sigma_p @ U 

    # Step 2) Get optimal solution
    B_estimated = ML_symmetric_positive_definite_estimator(sigma_theta_tilde, sigma_p_tilde, sigma_noise_approx, U)

    # TODO use CVXPY to solve the optimization
    B = cp.Variable(shape=(M,M))
    constraints = [B >> 0]
    for m in range(M):
        for k in range(M):
            if k < m:
                constraints.append(B[m,k] <= 0)
    for m in range(M):
        constraints.append(sum(B[m,:]) == 0)
    B_target = abs(U @ B_estimated @ U.T)
    print("B_target")
    print(B_target)
    problem = cp.Problem(cp.Minimize(cp.sum_squares(B_target - B)), constraints=constraints)
    problem.solve(verbose=True)
    print("status:",problem.status)
    print("optimal value", problem.value)
    print("optimal var", B.value)

    return B.value

def augmented_lagrangian_topology_recovery(N,M,U,sigma_theta_tilde, sigma_p, sigma_noise_approx):
    # Step 1) Get reduced sample covariance matrix (same as in two phase)
    sigma_p_tilde =  np.transpose(U) @ sigma_p @ U 
    U_pseudinv = np.linalg.pinv(U)

    # Step 2) Initialize B 
    # We use the B_PD from two phase
    B_estimated = ML_symmetric_positive_definite_estimator(sigma_theta_tilde, sigma_p_tilde, sigma_noise_approx, U)
    print(B_estimated)

    t = 0 # iteration

    # lagrangian multipliers
    mu = np.zeros((M-1, 1))  
    big_lambda = np.zeros(B_estimated.shape)
    big_gamma = np.zeros(B_estimated.shape)

    # other parameters
    gamma = 0.1 # Penalty parameter
    nabla = 0.1 # Learning rate

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
        mu = mu - gamma * W @ W_inv @ np.ones((M-1,1))
        mu = np.maximum(mu, np.zeros(mu.shape))

        # equation 29
        aux1 = (sigma_p_tilde - sigma_noise_approx**2 * U_pseudinv @ U_pseudinv.T) @ W_inv @ sigma_theta_tilde_inv
        aux2 = W.T @ (big_gamma.T @ big_gamma) @ W.T
        W_next = W - nabla * (aux1 - W.T - aux2 - big_lambda + np.ones((M-1,1)).T @ mu)

        W_inv = np.linalg.inv(W_next)

        if np.linalg.norm(W_next - W) < epsilon:
            criterion_reached = True

        print(W_next)

        W = W_next
        t += 1





net = pandapower.networks.case4gs()

# Get the required variables from the net: p[n] (active power injected), E(\tilde(theta)) (state covariance matrix)

# We define p[n] = B * \theta(n) + w[n]
# We have the exact values, so we need to add the measurement noise

# net.bus contains all buses
N = 10
noise_sigma = 0.1

pandapower.runpp(net)
actual_p = net.res_bus['p_mw']
M = len(actual_p)
actual_p = np.tile(np.array(actual_p),(N,1))

noise =  np.random.default_rng().normal(0, noise_sigma**2, actual_p.shape)
measurements = actual_p + noise
print("measurements")
print(measurements)

# TODO obtain B matrix from net data

# E(\tilde(theta)) = U(T) E(theta) U
# Where U = [-1(T)_(M-1) I_(M-1) ]

# I will assume they are all independent, but this might be wrong
theta_variance = 2
E_theta = np.diag(np.full(M, theta_variance))

U = np.concatenate((np.array([-np.ones(M-1)]), np.eye(M-1)))
E_theta_tilde = np.transpose(U) @ E_theta @ U

print("E_theta_tilde")
print(E_theta_tilde)

# Step 1) Remove sample mean of all measurements

# TODO investigate how to do this correctly in numpy
avg = np.zeros(M)
for time in measurements:
    avg += time
avg = avg/N
measurements = measurements - np.tile(np.array(avg),(N,1))
print("measurements2")
print(measurements)

# Step 2) Get sample covariance matrix

E_p = np.zeros((M,M))
for measurement in measurements:
    meas_formatted = np.array([measurement])
    E_p += np.transpose(meas_formatted) @ meas_formatted
print("E_p")
print(E_p)

# Step 3-4) Get eigenvalues and estimate noise variance with the smallest eigenvalue

eigenvalues = np.linalg.eig(E_p)[0]
sigma_noise_approx = eigenvalues.min()

# Step 5) Get \hat(\tilde(B)). The paper explains two methods to approximate maximum likelihood B

B_tilde_approx = augmented_lagrangian_topology_recovery(N,M,U,E_theta_tilde,E_p,sigma_noise_approx)
print("B_tilde_approx")
print(B_tilde_approx)
# TODO this steps (6 7 8) are all untested
# Step 6) Get \hat(B) from \hat(\tilde(B))

B_approx = U @ B_tilde_approx @ np.transpose(U)

# Step 7) Impose sparsity with a threshold
sparsity_value = 1/M
threshold = np.diag(B_approx).min() * sparsity_value
B_approx = np.where(np.abs(B_approx) < threshold, 0, B_approx)

# Step 8) Evaluate theta

theta_approx = np.zeros_like(measurements)
for i in range(0, measurements.shape[0]):
    theta_approx[i,:] = E_theta @ B_approx @ np.linalg.pinv(np.transpose(B_approx) @ E_theta @ B_approx + sigma_noise_approx * np.eye(M)) @ measurements[i]
print("theta_approx")
print(theta_approx)




