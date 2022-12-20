import networkx as nx
import numpy as np
import torch

from torch_graph_force import from_pandas_dataframe
from torch_graph_force import spring_layout


def test_correctness():
    # Use a random graph to avoid the case where
    # the model only produces correct result for a fixed graph
    G = nx.fast_gnp_random_graph(1000, 0.01)
    df = nx.to_pandas_edgelist(G)
    df["weight"] = 1.0
    n_nodes = len(G)
    init_pos_nx = nx.random_layout(G)
    init_pos_pt = torch.stack(
        [torch.tensor(init_pos_nx[n], dtype=torch.float64) for n in init_pos_nx]
    )

    ds = from_pandas_dataframe(df, nodes=n_nodes)
    pos_pt = spring_layout(ds, iterations=1, layout_config={"pos": init_pos_pt})
    pos_nx = nx.spring_layout(G, iterations=1, pos=init_pos_nx)
    pos_nx = np.asarray([pos_nx[n] for n in G])
    assert np.all(np.abs(pos_pt - pos_nx) < 0.001)
