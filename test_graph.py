from random_graph import is_connected_graph


def test_connected_graph():
    assert is_connected_graph([[1], [2], [0]], 3)
    assert is_connected_graph([[2], [2], [0, 1]], 3)
    assert not is_connected_graph([[1], [0], [3], [2]], 4)