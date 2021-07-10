import numpy as np

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

def IEEE14_b_matrix():
    # values taken from https://github.com/thanever/pglib-opf/blob/master/pglib_opf_case14_ieee.m
    # so I can check if the other one is working correctly
    from_bus = [1,1,2,2,2,3,4,4,4,5,6,6,6,7,7,9,9,10,12,13]
    to_bus = [2,5,3,4,5,4,5,7,9,6,11,12,13,8,9,10,14,11,13,14]
    b = [0.05917,0.22304,0.19797,0.17632,0.17388,0.17103,0.04211,0.20912,0.55618,0.25202,0.1989,0.25581,0.13027,0.17615,0.11001,0.0845,0.27038,0.19207,0.19988,0.34802]
    M = 14
    B_real = np.zeros((M,M))
    for row in zip(from_bus, to_bus, b):
        m = row[0]-1
        k = row[1]-1
        if m != k:
            b_mk = row[2]
            B_real[m,k] = -b_mk
            B_real[k,m] = -b_mk
            B_real[m,m] += b_mk
            B_real[k,k] += b_mk
    return B_real


