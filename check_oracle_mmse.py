
from GrotasTests import run_MSE_states_oracle
from NetworkMatrix import IEEE118_b_matrix, IEEE14_b_matrix
import numpy as np
import matplotlib

from simulations import get_observations


B_real, A = IEEE14_b_matrix()
N = 1500
c = 1
range_SNR = np.linspace(-20, 120, 3000)
points = [1500]
all_mse = []
for SNR in range_SNR:
    observations, sigma_theta, states, noise_sigma = get_observations(N, SNR, c, B_real)
    # if SNR > 100:
    #     noise_sigma = 0
    B_real, A = IEEE14_b_matrix()
    run = run_MSE_states_oracle(observations, B_real, sigma_theta, noise_sigma, states)
    run['SNR'] = SNR
    all_mse.append(run)

plots = []
legend = []
plots.append([x['MSE_states'] for x in all_mse if x['method']=='MSE_oracle' and N == x['N']])
legend.append("oracle, N={}".format(N))


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
# matplotlib.pyplot.savefig('plots/MSE_states_14_bus{}.png'.format(time_now))
matplotlib.pyplot.show()
