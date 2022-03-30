import numpy as np
import pandapower as pp

def get_b_matrix_from_network(network):
    M = len(network.bus)
    data_for_B = network.line[['from_bus','to_bus','x_ohm_per_km', 'length_km', 'max_i_ka']]
    B_real = np.zeros((M,M))
    A = np.zeros((M,M))
    for row in data_for_B.iterrows():
        # m = network.bus.index.get_loc(int(row[1]['from_bus']))
        # k = network.bus.index.get_loc(int(row[1]['to_bus']))
        m = int(row[1]['from_bus'])
        k = int(row[1]['to_bus'])
        # TODO: look into how to do this right
        indexes = list(network.bus.index)
        bus_m = network.bus.iloc[indexes.index(m)]
        bus_k = network.bus.iloc[indexes.index(k)]
        v_ms = bus_m['vn_kv'] * bus_k['vn_kv']
        if m != k:
            b_mk = -(row[1]['x_ohm_per_km'] * row[1]['length_km'] / v_ms * 100) # TODO check where does the 100 come from
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
        m = int(row[1]['hv_bus'])
        k = int(row[1]['lv_bus'])
        r = row[1]['vkr_percent'] * network.sn_mva / row[1]['sn_mva']
        z = row[1]['vk_percent'] * network.sn_mva / row[1]['sn_mva']
        x_mk = np.sqrt(z**2 - r**2)
        b_mk = - x_mk
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


def IEEE118_b_matrix():
    from_bus = [1, 1, 4, 3, 5, 6, 8, 8, 9, 4, 5, 11, 2, 3, 7, 11, 12, 13, 14, 12, 15, 16, 17, 18, 19, 15, 20, 21, 22, 23, 23, 26, 25, 27, 28, 30, 8, 26, 17, 29, 23, 31, 27, 15, 19, 35, 35, 33, 34, 34, 38, 37, 37, 30, 39, 40, 40, 41, 43, 34, 44, 45, 46, 46, 47, 42, 42, 45, 48, 49, 49, 51, 52, 53, 49, 49, 54, 54, 55, 56, 50, 56, 51, 54, 56, 56, 55, 59, 59, 60, 60, 61, 63, 63, 64, 38, 64, 49, 49, 62, 62, 65, 66, 65, 47, 49, 68, 69, 24, 70, 24, 71, 71, 70, 70, 69, 74, 76, 69, 75, 77, 78, 77, 77, 79, 68, 81, 77, 82, 83, 83, 84, 85, 86, 85, 85, 88, 89, 89, 90, 89, 89, 91, 92, 92, 93, 94, 80, 82, 94, 80, 80, 80, 92, 94, 95, 96, 98, 99, 100, 92, 101, 100, 100, 103, 103, 100, 104, 105, 105, 105, 106, 108, 103, 109, 110, 110, 17, 32, 32, 27, 114, 68, 12, 75, 76]
    to_bus = [2, 3, 5, 5, 6, 7, 9, 5, 10, 11, 11, 12, 12, 12, 12, 13, 14, 15, 15, 16, 17, 17, 18, 19, 20, 19, 21, 22, 23, 24, 25, 25, 27, 28, 29, 17, 30, 30, 31, 31, 32, 32, 32, 33, 34, 36, 37, 37, 36, 37, 37, 39, 40, 38, 40, 41, 42, 42, 44, 43, 45, 46, 47, 48, 49, 49, 49, 49, 49, 50, 51, 52, 53, 54, 54, 54, 55, 56, 56, 57, 57, 58, 58, 59, 59, 59, 59, 60, 61, 61, 62, 62, 59, 64, 61, 65, 65, 66, 66, 66, 67, 66, 67, 68, 69, 69, 69, 70, 70, 71, 72, 72, 73, 74, 75, 75, 75, 77, 77, 77, 78, 79, 80, 80, 80, 81, 80, 82, 83, 84, 85, 85, 86, 87, 88, 89, 89, 90, 90, 91, 92, 92, 92, 93, 94, 94, 95, 96, 96, 96, 97, 98, 99, 100, 100, 96, 97, 100, 100, 101, 102, 102, 103, 104, 104, 105, 106, 105, 106, 107, 108, 107, 109, 110, 110, 111, 112, 113, 113, 114, 115, 115, 116, 117, 118, 118]
    b = [0.0999, 0.0424, 0.00798, 0.108, 0.054, 0.0208, 0.0305, 0.0267, 0.0322, 0.0688, 0.0682, 0.0196, 0.0616, 0.16, 0.034, 0.0731, 0.0707, 0.2444, 0.195, 0.0834, 0.0437, 0.1801, 0.0505, 0.0493, 0.117, 0.0394, 0.0849, 0.097, 0.159, 0.0492, 0.08, 0.0382, 0.163, 0.0855, 0.0943, 0.0388, 0.0504, 0.086, 0.1563, 0.0331, 0.1153, 0.0985, 0.0755, 0.1244, 0.247, 0.0102, 0.0497, 0.142, 0.0268, 0.0094, 0.0375, 0.106, 0.168, 0.054, 0.0605, 0.0487, 0.183, 0.135, 0.2454, 0.1681, 0.0901, 0.1356, 0.127, 0.189, 0.0625, 0.323, 0.323, 0.186, 0.0505, 0.0752, 0.137, 0.0588, 0.1635, 0.122, 0.289, 0.291, 0.0707, 0.00955, 0.0151, 0.0966, 0.134, 0.0966, 0.0719, 0.2293, 0.251, 0.239, 0.2158, 0.145, 0.15, 0.0135, 0.0561, 0.0376, 0.0386, 0.02, 0.0268, 0.0986, 0.0302, 0.0919, 0.0919, 0.218, 0.117, 0.037, 0.1015, 0.016, 0.2778, 0.324, 0.037, 0.127, 0.4115, 0.0355, 0.196, 0.18, 0.0454, 0.1323, 0.141, 0.122, 0.0406, 0.148, 0.101, 0.1999, 0.0124, 0.0244, 0.0485, 0.105, 0.0704, 0.0202, 0.037, 0.0853, 0.03665, 0.132, 0.148, 0.0641, 0.123, 0.2074, 0.102, 0.173, 0.0712, 0.188, 0.0997, 0.0836, 0.0505, 0.1581, 0.1272, 0.0848, 0.158, 0.0732, 0.0434, 0.182, 0.053, 0.0869, 0.0934, 0.108, 0.206, 0.295, 0.058, 0.0547, 0.0885, 0.179, 0.0813, 0.1262, 0.0559, 0.112, 0.0525, 0.204, 0.1584, 0.1625, 0.229, 0.0378, 0.0547, 0.183, 0.0703, 0.183, 0.0288, 0.1813, 0.0762, 0.0755, 0.064, 0.0301, 0.203, 0.0612, 0.0741, 0.0104, 0.00405, 0.14, 0.0481, 0.0544]
    M = 118
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
