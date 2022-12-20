import pytest

from torch_graph_force.model import SpringLayout


def test_iterations():
    model = SpringLayout(n_nodes=10, iterations=50)
    model.to("cpu")
    with pytest.raises(AssertionError):
        # The temperature is supposed to reduce to 0.0 after (n+1) iterations
        # An error should occur at the iteration n+2
        for _ in range(52):
            model.update()


def test_threshold():
    model = SpringLayout(n_nodes=10, iterations=50)
    model.to("cpu")
    # Inital displacements are zeroes
    # If they are used to update the position,
    # it should below any postive threshold
    assert model.update()
