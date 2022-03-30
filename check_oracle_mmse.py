
from GrotasTests import run_MSE_states_oracle, run_test
from NetworkMatrix import IEEE118_b_matrix, IEEE14_b_matrix
import numpy as np
import matplotlib

from simulations import MSE_states, MSE_states_slow, MSE_states_theoretical, get_observations
from utils import matprint


def take_out_average(mat):
    M = mat.shape[1]
    N = mat.shape[0]
    avg = np.zeros(M)
    for time in mat:
        avg += time
    avg = avg/N
    return mat - np.tile(np.array(avg),(N,1))

def theory_mmse(B, sigma_theta, sigma_noise):
    M = B.shape[0]
    return np.trace(sigma_theta - sigma_theta @ B.T @ np.linalg.inv(B @ sigma_theta @ B.T + sigma_noise * np.eye(M)) @ B @ sigma_theta)


B_real, A = IEEE14_b_matrix()
Ns = [200, 1500]
c = 1
range_SNR = np.linspace(-20, 100, 300)
all_mse = []
for N in Ns:
    for SNR in range_SNR:
        observations, sigma_theta, states, noise_sigma = get_observations(N, SNR, c, B_real)
        observations = take_out_average(observations)

        # observations = take_out_average(observations)
        # states = take_out_average(states)
        # B_real, A = IEEE14_b_matrix()
        run = run_MSE_states_oracle(observations, B_real, sigma_theta, noise_sigma, states)
        run['SNR'] = SNR
        # run['MSE_states'] *=  10 / 6.66
        # run['MSE_states'] *=  13.8 / 10.7 
        # run['MSE_states'] *=  6 / 4 
        all_mse.append(run)
        # run = run_test(B_real, observations, sigma_theta, 'augmented_lagrangian', states)
        MSE_states_total = theory_mmse(B_real, sigma_theta, noise_sigma)
        run = {
            "method": 'MSE_oracle_B',
            "N": N,
            "MSE_states": MSE_states_total,
            "SNR": SNR
        }
        all_mse.append(run)

plots = []
legend = []
for N in Ns:
    oracle_mse = [np.abs(x['MSE_states']) for x in all_mse if x['method'] == 'MSE_oracle' and N == x['N']]
    other_mse = [np.abs(x['MSE_states']) for x in all_mse if x['method'] != 'MSE_oracle' and N == x['N']]
    for x in (np.array(oracle_mse)/np.array(other_mse)):
        print(x)
    plots.append([np.abs(x['MSE_states']) for x in all_mse if x['method']=='MSE_oracle' and N == x['N']])
    legend.append("oracle, N={}".format(N))
    plots.append([np.abs(x['MSE_states']) for x in all_mse if x['method']!='MSE_oracle' and N == x['N']])
    legend.append("theory, N={}".format(N))


fig = matplotlib.pyplot.figure()
ax = fig.add_subplot(1,1,1)
colors = ['red', 'blue', 'black', 'magenta', 'green', 'black']
color_gen = (x for x in colors)
for plot in plots:
    ax.semilogy(range_SNR, plot, color=next(color_gen), lw=1)
ax.set_title("MSE states for 14-bus network")
ax.set_yscale('log')
fig.legend(legend)
matplotlib.pyplot.grid(True, which='both')
matplotlib.pyplot.ylabel('MSE')
matplotlib.pyplot.xlabel('SNR [dB]')
matplotlib.pyplot.savefig('plots/0316MSE_states_14_bus_now.png')
matplotlib.pyplot.show()
