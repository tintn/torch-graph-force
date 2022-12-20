import pytest
import networkx as nx
import numpy as np


from torch_force_graph import from_edgelist, from_networkx, from_pandas_dataframe


@pytest.fixture
def katate_club_graph():
    G = nx.karate_club_graph()
    smtx = nx.to_scipy_sparse_matrix(G)
    return G, smtx


def _compare(ds, smtx):
    for i in range(ds.n):
        row1 = ds[i]
        indices1 = row1[1].numpy()
        weights1 = row1[2].numpy()
        row2 = smtx[i, :]
        indices2 = row2.indices
        weights2 = row2.data
        print(weights1, weights2)
        assert np.all(indices1 == indices2)
        assert np.all(weights1 == weights2)


def test_networkx(katate_club_graph):
    G, smtx = katate_club_graph
    ds = from_networkx(G, device="cpu")
    _compare(ds, smtx)


def test_pandas_dataframe(katate_club_graph):
    G, smtx = katate_club_graph
    df = nx.to_pandas_edgelist(G)
    # Test dataset creation without node IDs
    n_nodes = len(G)
    ds = from_pandas_dataframe(df, n_nodes, device="cpu")
    print(ds.weights)
    _compare(ds, smtx)
    # Test dataset with noide IDs provided
    nodes = list(G)
    ds = from_pandas_dataframe(df, nodes, device="cpu")
    _compare(ds, smtx)


def test_edgelist(katate_club_graph):
    G, smtx = katate_club_graph
    df = nx.to_pandas_edgelist(G)
    edges = df[["source", "target"]].values
    weights = df["weight"].values
    n_nodes = len(G)
    ds = from_edgelist(edges, nodes=n_nodes, weights=weights, device="cpu")
    _compare(ds, smtx)
