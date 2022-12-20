import gc
import time
import networkx as nx

from .layout import spring_layout
from .data import from_pandas_dataframe
from .utils import generate_random_graph


def benchmark_cpu(start=500, stop=10001, step=500):
    """
    Compare runtime between PyTorch_cpu and Networkx
    """
    device = "cpu"
    nodes = []
    time_pt = []
    time_nx = []
    for n_nodes in range(start, stop, step):
        gc.collect()
        df = generate_random_graph(n_nodes, 0.001)
        G = nx.from_pandas_edgelist(
            df, source="source", target="target", edge_attr="weight"
        )
        nodes.append(n_nodes)

        # PyTorch Implementation
        s = time.time()
        ds = from_pandas_dataframe(df, n_nodes, device=device)
        pos = spring_layout(ds, device="cpu")
        e = time.time()
        time_pt.append(e - s)

        # Networkx Implementation
        s = time.time()
        pos = nx.spring_layout(G)
        e = time.time()

        time_nx.append(e - s)
        print(
            f"#Nodes: {nodes[-1]}."
            f" Elapsed Time (PyTorch_cpu): {time_pt[-1]}."
            f" Elapsed Time (Networkx): {time_nx[-1]}"
        )
        del pos
    return nodes, time_pt, time_nx


def benchmark_gpu(start=5000, stop=100001, step=5000):
    """
    Calculate runtime of the PyTorch implementation
    for large graphs with GPU acceleration
    """
    device = "cuda"
    nodes = []
    time_pt = []
    for n_nodes in range(start, stop, step):
        gc.collect()
        df = generate_random_graph(n_nodes, 0.001)
        nodes.append(n_nodes)

        s = time.time()
        ds = from_pandas_dataframe(df, n_nodes, device=device)
        pos = spring_layout(ds, device=device)
        e = time.time()
        time_pt.append(e - s)

        print(f"#Nodes: {nodes[-1]}." f" Elapsed Time (PyTorch_gpu): {time_pt[-1]}")
        del pos
    return nodes, time_pt


if __name__ == "__main__":
    nodes, time_pt, time_nx = benchmark_cpu()
