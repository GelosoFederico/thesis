# We are going to make a script to run Grotas algorithm over a network, and check how closely did it estimate it

# Create the network
import pandapower
import pandapower.networks
import numpy as np
import scipy.linalg
import cvxpy as cp
import matplotlib.pyplot

# TODO change all np.transpose to matrix.T

def matprint(mat, fmt="g"):
    col_maxes = [max([len(("{:"+fmt+"}").format(x)) for x in col]) for col in mat.T]
    for x in mat:
        for i, y in enumerate(x):
            print(("{:"+str(col_maxes[i])+fmt+"}").format(y), end="  ")
        print("")

def oracle_mmse_state_estimation(B, sigma_theta, noise_variance, measurements):
    # theta values estimation from equation 11
    M = B.shape[0]
    return sigma_theta @ B @ np.pinv(B.T @ sigma_theta @ B + noise_variance @ np.eye(M)) @ measurements

def MSE_matrix(matrix_real, matrix_est):
    diff_matrix = matrix_real - matrix_est
    return np.trace(diff_matrix @ diff_matrix.T)

def stack_matrix(matrix):
    out_vector = np.array([])
    for column in matrix.T:
        out_vector = np.append(out_vector, column)
    return out_vector


def cramer_rao_bound(M, B_til_est, sigma_sqr, sigma_p, sigma_theta_tilde, N):
    def psi_vector(M, k, l):
        constant = 1 if k==l else 0.5
        ek = el = np.zeros((M-1,1))
        ek[k,0] = 1
        el[l,0] = 1
        # Ek,l = ek * el.T + el * ek.T
        # TODO check if there is a numpy function for stacking
        stack = stack_matrix(ek @ el.T + el @ ek.T)
        return constant * stack


    # B_til_est is a square matrix R^(M-1,M-1)
    alpha = np.zeros((M*(M-1)//2 + 1 ))
    filled = 0
    for i in range(0, M-1):
        alpha[filled:filled+(M-i-1)] = B_til_est[i:M,i]
        filled += (M-i-1)
    alpha[M*(M-1)//2] = sigma_sqr

    U = get_U_matrix(M)
    sigma_p_tilde =  U.T @ sigma_p @ U 
    sigma_p_tilde_inv = np.linalg.inv(sigma_p_tilde)
    Q = np.kron(sigma_p_tilde_inv, sigma_p_tilde_inv)
    K = np.kron(B_til_est @ sigma_theta_tilde, np.eye(M-1)) + np.kron(np.eye(M-1), B_til_est @ sigma_theta_tilde)
    K_inv = np.linalg.inv(K)

    U_pseudinv = np.linalg.pinv(U)
    final_col = K_inv @ stack_matrix(U_pseudinv @ U_pseudinv.T)

    Psi = np.zeros(((M-1)**2, M*(M-1)//2 + 1))
    next_col = 0
    for k in range(0, M-1):
        for l in range(k, M-1):
            Psi[:, next_col] = psi_vector(M, k, l)
            next_col += 1
    Psi[:,M*(M-1)//2] = final_col
    return np.trace(np.linalg.pinv(Psi.T @ K.T @ Q @ K @ Psi) * 2 / N)

def get_b_matrix_from_network(network):
    M = len(network.res_bus['p_mw'])
    data_for_B = network.line[['from_bus','to_bus','x_ohm_per_km', 'length_km']]
    B_real = np.zeros((M,M))
    for row in data_for_B.iterrows():
        m = network.bus.index.get_loc(int(row[1]['from_bus']))
        k = network.bus.index.get_loc(int(row[1]['to_bus']))
        if m != k:
            b_mk = -1/(row[1]['x_ohm_per_km'] * row[1]['length_km'])
            B_real[m,k] = b_mk
            B_real[k,m] = b_mk
            B_real[m,m] -= b_mk
            B_real[k,k] -= b_mk
    for row in network.trafo.iterrows():
        # From Monticelli 4.7, transformer value should be akm * xkm^-1
        # From pandapower documentation (https://pandapower.readthedocs.io/en/v2.6.0/elements/trafo.html), xkm is
        # sqrt(z^2 - r^2) where alpha = net.sn_mva/sn_mva  z = vk_percent*alpha/100 and r = vkr_percent*alpha/100
        # b = -1/x
        ratio = row[1]['vn_hv_kv'] / row[1]['vn_lv_kv']
        m = network.bus.index.get_loc(int(row[1]['hv_bus']))
        k = network.bus.index.get_loc(int(row[1]['lv_bus']))
        r = row[1]['vkr_percent'] * network.sn_mva / row[1]['sn_mva'] / 100
        z = row[1]['vk_percent'] * network.sn_mva / row[1]['sn_mva'] / 100
        x_mk =  np.sqrt(z**2 - r**2) * ratio
        b_mk = -1 / x_mk
        if m != k:
            B_real[m,k] = b_mk
            B_real[k,m] = b_mk
            B_real[m,m] -= b_mk
            B_real[k,k] -= b_mk

    return B_real

def get_U_matrix(M):
    return np.concatenate((np.array([-np.ones(M-1)]), np.eye(M-1)))

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

def ML_symmetric_positive_definite_estimator(sigma_theta_tilde, sigma_p_tilde, sigma_noise_approx, U):
    U_pseudinv = np.linalg.pinv(U)

    # one way to get square root
    # TODO analize if the square root algorithm used here is useful
    sigma_theta_tilde_sqrt = scipy.linalg.sqrtm(sigma_theta_tilde)

    # another way to get square root
    # TODO check if I whould always send hermitian
    # u_sigma, s_sigma, v_sigma = np.linalg.svd(sigma_theta_tilde,compute_uv=True, hermitian=True)

    sigma_theta_tilde_sqrt_inv = np.linalg.inv(sigma_theta_tilde_sqrt)

    aux = scipy.linalg.sqrtm(sigma_theta_tilde_sqrt @ (sigma_p_tilde - sigma_noise_approx**2 * U_pseudinv @ U_pseudinv.T ) @ sigma_theta_tilde_sqrt)
    B_estimation = sigma_theta_tilde_sqrt_inv @ aux @ sigma_theta_tilde_sqrt_inv
    
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
            if k != m and k > m:
                # Simetric condition
                constraints.append(B[m,k] == B[k,m])

    for m in range(M):
        # condition 3
        constraints.append(sum(B[m,:]) == 0)
    B_target = abs(U @ B_estimated @ U.T)
    print("B_target")
    matprint(B_target)
    print("sigma_theta")
    matprint(U @ sigma_theta_tilde @ U.T )
    print("sigma_noise_approx")
    print(sigma_noise_approx)
    # print("B_target")
    # print(B_target)
    problem = cp.Problem(cp.Minimize(cp.sum_squares(B_target - B)), constraints=constraints)
    problem.solve()
    # print("status:",problem.status)
    # print("optimal value", problem.value)
    # print("optimal var\n", B.value)

    return B.value

def augmented_lagrangian_topology_recovery(N,M,U,sigma_theta_tilde, sigma_p, sigma_noise_approx):
    # Step 1) Get reduced sample covariance matrix (same as in two phase)
    sigma_p_tilde =  np.transpose(U) @ sigma_p @ U 
    U_pseudinv = np.linalg.pinv(U)

    # Step 2) Initialize B 
    # We use the B_PD from two phase
    B_estimated = ML_symmetric_positive_definite_estimator(sigma_theta_tilde, sigma_p_tilde, sigma_noise_approx, U)
    # print(B_estimated)

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
        matprint(B_approx)
        threshold = np.diag(B_approx).min() * sparsity_value
        return np.where(np.abs(B_approx) < threshold, 0, B_approx)

    def step_8(B, E_theta, sigma_noise_approx, measurements, M):
        theta_approx = np.zeros_like(measurements)
        for i in range(0, measurements.shape[0]):
            theta_approx[i,:] = E_theta @ B @ np.linalg.pinv(B.T @ E_theta @ B + sigma_noise_approx**2 * np.eye(M)) @ measurements[i]
        return theta_approx

    observations_no_mean = step_1(observations, M, N)
    sample_covariance_matrix = step_2(observations_no_mean, M, N) # aka E_p
    print("sample_covariance_matrix")
    matprint(sample_covariance_matrix)
    sigma_noise_estimation = step_3_4(sample_covariance_matrix)
    print(f"{sigma_noise_estimation=}")
    B_estimation = step_5_6(N, M, state_covariance_matrix_tilde, sample_covariance_matrix, sigma_noise_estimation, method)
    B_estimation = step_7(M, B_estimation)
    theta_estimation = step_8(B_estimation, state_covariance_matrix, sigma_noise_estimation, observations_no_mean, M)
    return B_estimation, theta_estimation, sigma_noise_estimation, sample_covariance_matrix

# def test_set(net, N, SNR_dB):
#     pandapower.runpp(net)
#     B_real = get_b_matrix_from_network(net)

def get_observations(N, SNR, c, B):
    M = B.shape[0]
    # sigma_theta = c**2*I_M
    # SNR = 10 log(1/sigma**2 * Tr{B_tilde * sigma_theta_tilde * B_tilde})
    # sigma ** 2 = Tr{B_tilde * sigma_theta_tilde * B_tilde} / 10 ^ (SNR/10)
    U = get_U_matrix(M)
    sigma_theta = c**2 * np.eye(M)
    sigma_theta_tilde = U.T @ sigma_theta @ U
    B_tilde = B[1:M, 1:M]
    print("Values in noise sigma")
    print("B")
    matprint(B)
    matprint(U)
    matprint(B_tilde)
    matprint(sigma_theta_tilde)
    print(10**(SNR/10))
    
    noise_sigma = np.trace(B_tilde @ sigma_theta_tilde @ B_tilde) / (10**(SNR/10))
    print("real covariance matrix p")
    matprint(B.T @ sigma_theta @ B + noise_sigma**2 * np.eye(M))
    print(f"{sigma_theta=}")
    print(f"{noise_sigma=}")

    # TODO send distribution as param
    theta_created = np.random.default_rng().normal(0, np.sqrt(c**2), (M,N))
    noise =  np.random.default_rng().normal(0, noise_sigma, (M,N))
    return ((B @ theta_created) + noise).T, sigma_theta


net = pandapower.networks.case14()
pandapower.runpp(net)
MSE_tests = []
B_real = get_b_matrix_from_network(net)
c = 1
range_SNR = np.linspace(0, 25, 10)
for SNR in range_SNR:
    for N in [1500]:
        observations, sigma_theta = get_observations(N, SNR, c, B_real)
        
        B, theta, sigma_est, sigma_p = GrotasAlgorithm(observations, sigma_theta, 'two_phase_topology')
        MSE = MSE_matrix(B_real, B)
        MSE_tests.append({
            "method": 'two_phase_topology',
            "N": N,
            "SNR": SNR,
            "B": B,
            "theta": theta,
            "sigma_est": sigma_est,
            "sigma_p": sigma_p,
            "MSE": MSE,
        })

        M = B.shape[0]
        U = get_U_matrix(M)
        CRB = cramer_rao_bound(M, U.T @ B @ U, sigma_est**2,sigma_p, U.T @ sigma_theta @ U, N)
        MSE_tests.append({
            "method": 'cramer_rao_bound',
            "N": N,
            "SNR": SNR,
            "MSE": CRB,
        })

MSE_two_phase = filter(lambda x: x['method']=='two_phase_topology', MSE_tests)
MSE_two_phase_for_plot = []

# TODO do this with map
for test in MSE_two_phase:
    MSE_two_phase_for_plot.append(test['MSE'])

CRBs = filter(lambda x: x['method']=='cramer_rao_bound', MSE_tests)
CRBs_for_plot = []

# TODO do this with map
for test in CRBs:
    CRBs_for_plot.append(test['MSE'])

import matplotlib.pyplot
print(MSE_two_phase_for_plot)
print(CRBs_for_plot)
matplotlib.pyplot.plot(range_SNR, MSE_two_phase_for_plot)
matplotlib.pyplot.plot(range_SNR, CRBs_for_plot)
matplotlib.pyplot.show()

# matplotlib.pyplot.matshow(B)
# matplotlib.pyplot.show()
# matplotlib.pyplot.matshow(B_real)
# matplotlib.pyplot.show()
