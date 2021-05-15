# We are going to make a script to run Grotas algorithm over a network, and check how closely did it estimate it

# Create the network
import pandapower
import pandapower.networks
import numpy as np
import scipy.linalg

# TODO change all np.transpose to matrix.T

def two_phase_topology_recovery(N,M,U,sigma_theta_tilde, sigma_p, sigma_noise_approx):
    # Step 1) Get reduced sample covariance matrix
    sigma_p_tilde =  np.transpose(U) @ sigma_p @ U 
    U_pseudinv = np.linalg.pinv(U)

    # Step 2) Get optimal solution
    # one way to get square root
    # TODO analize if the square root algorithm used here is useful
    sigma_theta_tilde_sqrt = scipy.linalg.sqrtm(sigma_theta_tilde)

    # another way to get square root
    # TODO check if I whould always send hermitian
    # u_sigma, s_sigma, v_sigma = np.linalg.svd(sigma_theta_tilde,compute_uv=True, hermitian=True)

    sigma_theta_tilde_sqrt_inv = np.linalg.inv(sigma_theta_tilde_sqrt)

    aux = sigma_theta_tilde_sqrt @ (sigma_p_tilde - sigma_noise_approx**2 * U_pseudinv @ U_pseudinv.T ) @ sigma_theta_tilde_sqrt
    B_estimated = sigma_theta_tilde_sqrt_inv @ aux @ sigma_theta_tilde_sqrt_inv

    # TODO use CVXPY to solve the optimization
    return B_estimated


net = pandapower.networks.mv_oberrhein()

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
print(measurements)

# TODO obtain B matrix from net data

# E(\tilde(theta)) = U(T) E(theta) U
# Where U = [-1(T)_(M-1) I_(M-1) ]

# I will assume they are all independent, but this might be wrong
theta_variance = 2
E_theta = np.diag(np.full(M, theta_variance))

U = np.concatenate((np.array([-np.ones(M-1)]), np.eye(M-1)))
E_theta_tilde = np.transpose(U) @ E_theta @ U

print(E_theta_tilde)

# Step 1) Remove sample mean of all measurements

# TODO investigate how to do this correctly in numpy
avg = np.zeros(M)
for time in measurements:
    avg += time
avg = avg/N
measurements = measurements - np.tile(np.array(avg),(N,1))
print(measurements)

# Step 2) Get sample covariance matrix

E_p = np.zeros((M,M))
for measurement in measurements:
    meas_formatted = np.array([measurement])
    E_p += np.transpose(meas_formatted) @ meas_formatted
print(E_p)

# Step 3-4) Get eigenvalues and estimate noise variance with the smallest eigenvalue

eigenvalues = np.linalg.eig(E_p)[0]
sigma_noise_approx = eigenvalues.min()

# Step 5) Get \hat(\tilde(B)). The paper explains two methods to approximate maximum likelihood B

B_tilde_approx = two_phase_topology_recovery(N,M,U,E_theta_tilde,E_p,sigma_noise_approx)

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
print(theta_approx)




