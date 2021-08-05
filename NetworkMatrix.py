import numpy as np
import pandapower as pp

def get_b_matrix_from_network(network):
    M = len(network.bus)
    data_for_B = network.line[['from_bus','to_bus','x_ohm_per_km', 'length_km']]
    B_real = np.zeros((M,M))
    A = np.zeros((M,M))
    for row in data_for_B.iterrows():
        # m = network.bus.index.get_loc(int(row[1]['from_bus']))
        # k = network.bus.index.get_loc(int(row[1]['to_bus']))
        m = int(row[1]['from_bus'])
        k = int(row[1]['to_bus'])
        if m != k:
            b_mk = -1/(row[1]['x_ohm_per_km'] * row[1]['length_km'])
            B_real[m,k] = b_mk
            B_real[k,m] = b_mk
            B_real[m,m] -= b_mk
            B_real[k,k] -= b_mk
            A[m,k] = 1
            A[k,m] = 1
    for row in network.trafo.iterrows():
        # From Monticelli 4.7, transformer value should be akm * xkm^-1
        # From pandapower documentation (https://pandapower.readthedocs.io/en/v2.6.0/elements/trafo.html), xkm is
        # sqrt(z^2 - r^2) where alpha = net.sn_mva/sn_mva  z = vk_percent*alpha/100 and r = vkr_percent*alpha/100
        # b = -1/x
        ratio = row[1]['vn_hv_kv'] / row[1]['vn_lv_kv']
        m = int(row[1]['hv_bus'])
        k = int(row[1]['lv_bus'])
        r = row[1]['vkr_percent'] * network.sn_mva / row[1]['sn_mva'] / 100
        z = row[1]['vk_percent'] * network.sn_mva / row[1]['sn_mva'] / 100
        x_mk =  np.sqrt(z**2 - r**2) * ratio
        b_mk = -1 / x_mk
        if m != k:
            B_real[m,k] = b_mk
            B_real[k,m] = b_mk
            B_real[m,m] -= b_mk
            B_real[k,k] -= b_mk
            A[m,k] = 1
            A[k,m] = 1

    return B_real, A

def IEEE14_b_matrix():
    # values taken from https://github.com/thanever/pglib-opf/blob/master/pglib_opf_case14_ieee.m
    # so I can check if the other one is working correctly
    from_bus = [1,1,2,2,2,3,4,4,4,5,6,6,6,7,7,9,9,10,12,13]
    to_bus = [2,5,3,4,5,4,5,7,9,6,11,12,13,8,9,10,14,11,13,14]
    b = [0.05917,0.22304,0.19797,0.17632,0.17388,0.17103,0.04211,0.20912,0.55618,0.25202,0.1989,0.25581,0.13027,0.17615,0.11001,0.0845,0.27038,0.19207,0.19988,0.34802]
    M = 14
    return get_matrix_from_IEEE_format(from_bus, to_bus, b, M)

def IEEE3_b_matrix():
    # values taken from https://github.com/thanever/pglib-opf/blob/master/pglib_opf_case3_lmbd.m
    # so I can check if the other one is working correctly
    from_bus = [1,3,1]
    to_bus = [3,2,2]
    b = [0.45, 0.7, 0.3]
    M = 3
    return get_matrix_from_IEEE_format(from_bus, to_bus, b, M)

def IEEE5_b_matrix():
    # values taken from https://github.com/thanever/pglib-opf/blob/master/pglib_opf_case3_lmbd.m
    # so I can check if the other one is working correctly
    from_bus = [1,1,1,2,3,4]
    to_bus = [2,4,5,3,4,5]
    b = [
        0.0281,
        0.0304,
        0.0064,
        0.0108,
        0.0297,
        0.0297
    ]
    M = 5
    return get_matrix_from_IEEE_format(from_bus, to_bus, b, M)

def IEEE5_b_matrix_from_pandapower_net():
    # Here I create the network on pandapower and get the b matrix from that
    net = pp.create_empty_network(sn_mva=100)
    pp.create_bus(net, name="Bus 1", vn_kv=230, type="b")
    pp.create_bus(net, name="Bus 2", vn_kv=230, type="b")
    pp.create_bus(net, name="Bus 3", vn_kv=230, type="b")
    pp.create_bus(net, name="Bus 4", vn_kv=230, type="b")
    pp.create_bus(net, name="Bus 5", vn_kv=230, type="b")

    bus1 = pp.get_element_index(net, "bus", "Bus 1")
    bus2 = pp.get_element_index(net, "bus", "Bus 2")
    bus3 = pp.get_element_index(net, "bus", "Bus 3")
    bus4 = pp.get_element_index(net, "bus", "Bus 4")
    bus5 = pp.get_element_index(net, "bus", "Bus 5")

    # we have to create the line types so it can use them
    pp.create_line_from_parameters(net, bus1, bus2, 1, 0.00281, 0.0281, 300, 1)
    pp.create_line_from_parameters(net, bus1, bus4, 1, 0.00304, 0.0304, 300, 1)
    pp.create_line_from_parameters(net, bus1, bus5, 1, 0.00064, 0.0064, 300, 1)
    pp.create_line_from_parameters(net, bus2, bus3, 1, 0.00108, 0.0108, 300, 1)
    pp.create_line_from_parameters(net, bus3, bus4, 1, 0.00297, 0.0297, 300, 1)
    pp.create_line_from_parameters(net, bus4, bus5, 1, 0.00297, 0.0297, 300, 1)
    # pp.runpp(net)
    return get_b_matrix_from_network(net)

def get_matrix_from_IEEE_format(from_bus, to_bus, b, M=None):
    if not M:
        M = max(max(from_bus), max(to_bus))
    B_real = np.zeros((M,M))
    A = np.zeros((M,M))
    for row in zip(from_bus, to_bus, b):
        m = row[0]-1
        k = row[1]-1
        if m != k:
            b_mk = row[2]
            B_real[m,k] = -b_mk
            B_real[k,m] = -b_mk
            B_real[m,m] += b_mk
            B_real[k,k] += b_mk
            A[m,k] = 1
            A[k,m] = 1
    return B_real, A
