# We are going to make a script to run Grotas algorithm over a network, and check how closely did it estimate it

# Create the network
import pandapower
import pandapower.networks
import numpy as np

net = pandapower.networks.mv_oberrhein()

# Get the required variables from the net: p[n] (active power injected), E(\tilde(theta)) (state covariance matrix)

# We define p[n] = B * \theta(n) + w[n]
# We have the exact values, so we need to add the measurement noise

# net.bus contains all buses
N = 10
normal_variance = 0.01

pandapower.runpp(net)
actual_p = net.res_bus['p_mw']
M = len(actual_p)
actual_p = np.tile(np.array(actual_p),(N,1))

noise =  np.random.default_rng().normal(0, normal_variance, actual_p.shape)
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
sigma = eigenvalues.min()

# Step 5) Get \hat(\tilde(B)). The paper explains two methods to approximate maximum likelihood B

B_tilde_aprox = np.eye(M)

# Step 6) Get \hat(B) from \hat(\tilde(B))

B_aprox = U @ B_tilde_aprox @ np.transpose(U)

