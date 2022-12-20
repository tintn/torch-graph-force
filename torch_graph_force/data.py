import networkx as nx
import numpy as np
from scipy import sparse
import pandas as pd
import torch
from torch.utils.data import Dataset


class GraphDataset(Dataset):
    def __init__(self, adj_mtx, nodes=None, device=None):
        """
        Args:
            adj_mtx (scipy.sparse.*_matrix): adjacency matrix in sparse format
            nodes (list, optional): list of node IDs corresponds to matrix's indices
            device (str, optional): device to load the dataset to (e.g. 'cpu', 'cuda')

        """
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"

        adj_mtx = adj_mtx.tocsr()
        self.adj_mtx = adj_mtx
        n, m = adj_mtx.shape
        assert n == m
        self.n = n

        self.nodes = nodes
        if nodes is not None:
            assert len(nodes) == self.n

        self.row_indices = torch.arange(n, dtype=torch.long, device=device)
        self.indptr = torch.tensor(adj_mtx.indptr, dtype=torch.long, device=device)
        self.col_indices = torch.tensor(
            adj_mtx.indices, dtype=torch.long, device=device
        )
        self.weights = torch.tensor(adj_mtx.data, dtype=torch.long, device=device)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        start, end = self.indptr[idx], self.indptr[idx + 1]
        neighbors = self.col_indices[start:end]
        weights = self.weights[start:end]
        return self.row_indices[idx], neighbors, weights


def from_networkx(G, device=None):
    """
    Create a dataset from Networkx Graph

    This method is not recommended.
    It's slow to convert Networkx Graph to sparse tensor for large graphs
    """
    adj_mtx = nx.to_scipy_sparse_matrix(G)
    nodes = list(G)
    return GraphDataset(adj_mtx, nodes, device)


def from_pandas_dataframe(
    df, nodes=None, source="source", target="target", weight="weight", device=None
):
    """
    Create a dataset from a Pandas DataFrame

    Args:
        df (pd.DataFrame): a pandas dataframe of the graph's edges.
        nodes (int or list): If int, it's the number of nodes in the graph
                                 (assume the node IDs are consecutive integers from 0)
                             If list, it's the list of node IDs for the graph.
                                 The order of the IDs is used to map nodes
                                 to the sparse matrix's row and col indices
        source (str, optional): column name for source nodes
        target (str, optional): column name for target nodes
        weight (str, optional): column name for edge weights.
                                If None, all weights are 1.0 by default
        device (str, optional): device to load the dataset to
    Return:
        GraphDataset
    """
    if nodes is None:
        n_nodes = max(df[source].values, df[target].values)
    elif isinstance(nodes, int):
        n_nodes = nodes
        nodes = None
    elif isinstance(nodes, list):
        n_nodes = len(nodes)
        # Map node IDs to indices
        idxmap = dict(zip(nodes, range(n_nodes)))
        df[source] = df[source].map(idxmap)
        df[target] = df[target].map(idxmap)
        assert not df[source].isnull().any()
        assert not df[target].isnull().any()
    else:
        raise ValueError('"nodes" must be either integer or list: {nodes}')

    rows = df[source].values
    cols = df[target].values
    if weight not in df.columns:
        weights = np.full(len(rows), 1.0, dtype=np.float64)
    else:
        weights = df[weight].values

    # We're dealing with undirected graph, symmetrize the adjacency matrix
    rows, cols = np.concatenate([rows, cols]), np.concatenate([cols, rows])
    weights = np.concatenate([weights, weights])

    adj_mtx = sparse.coo_matrix(
        (weights, (rows, cols)), shape=(n_nodes, n_nodes)
    ).tocsr()
    return GraphDataset(adj_mtx, nodes, device)


def from_edgelist(edges, nodes=None, weights=None, device=None):
    """
    Create a dataset from edgelist
    Args:
        edges (np.array): an numpy array of shape (n_nodes, 2)
                          the first column is the source nodes,
                          the second column is the target nodes
        nodes (int or list, optional):
                        If int, it's the number of nodes in the graph
                        (assume the node IDs are consecutive integers from 0)
                        If list, it's the list of node IDs for the graph.
                        The order of the IDs is used to map nodes
                        to the sparse matrix's row and col indices
        weights (float, optional): an numpy array of shape (n_nodes) for edge weights
        device (str, optional): device to load the dataset to
    Return:
        GraphDataset
    """
    edges = edges.astype(np.int64)
    df = pd.DataFrame(data=edges, columns=["source", "target"])
    weights = 1.0 if weights is None else weights.astype(np.float64)
    df["weight"] = weights
    return from_pandas_dataframe(df, nodes=nodes, device=device)
