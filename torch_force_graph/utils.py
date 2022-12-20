import numpy as np
import pandas as pd


def rescale_layout(pos, scale=1):
    # Center the points around the origin
    mean = pos.mean(axis=0)
    pos = pos - mean[None, :]
    # Scale the coordinates
    lim = np.abs(pos).max()
    pos *= scale / lim
    return pos


def generate_random_graph(n, p):
    """
    A simple util to generate random undirected graph
    Args:
        n (int): number of nodes
        p (float): probability for edge creation
    Return:
        pandas.DataFrame :
                the dataframe contains three columns "source", "target" and "weight"
                for the generated edges' source indices, target indices and weights
    """
    # Calculate number of edges per node.
    # Take into account the chance two nodes having two directed edges
    num_edges = n**2 * p * 2
    edge_per_node = n * p * 2
    source = (np.arange(num_edges) // edge_per_node).astype(np.int64)
    target = np.random.randint(n, size=len(source), dtype=np.int64)
    edges = np.stack([source, target], axis=1)
    edges = edges[source < target, :]
    df = pd.DataFrame(data=edges, columns=["source", "target"])
    df = df.drop_duplicates()
    df["weight"] = 1.0
    return df
