import numpy as np
import pandapower.networks
from NetworkMatrix import get_b_matrix_from_network, IEEE5_b_matrix, IEEE5_b_matrix_from_pandapower_net
from utils import matprint

B, A = IEEE5_b_matrix()
print('ieee')
matprint(B)
matprint(A)

net = pandapower.networks.case5()
pandapower.runpp(net)
B2, A2 = get_b_matrix_from_network(net)
print('my_b')
matprint(B2)
matprint(A2)

B3 = np.real(np.array(net._ppc['internal']['Bbus'].A))
print('internal')
matprint(B3)

B4, A4 = IEEE5_b_matrix_from_pandapower_net()
print('my_b2')
matprint(B4)
matprint(A4)


print('branch')
matprint(np.real(np.array(net._ppc['branch'])))

assert np.sum(A - A2) == 0
assert np.sum(np.abs(B - B2)) < 1
