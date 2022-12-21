# torch-graph-force [WIP]

A PyTorch-based library for embedding large graphs to low-dimensional space using force-directed layouts with GPU acceleration.

The aim of this project is to speed up the process of obtaining low-dimensional layouts for large graphs, especially with GPU acceleration.

## Install

- Install PyTorch (follow [official instructions](https://pytorch.org/get-started/locally))
- Install `torch-graph-force`:
```shell
pip install git+https://github.com/tintn/torch-graph-force.git
```

## Usage

### Create `GraphDataset` for The Graph.

The dataset can be created from a dataframe, an edgelist or Networkx Graph using `from_pandas_dataframe`, `from_edgelist`, or `from_networkx` respectively. `from_pandas_dataframe` is the recommended way as it's more efficient compared to other methods.

If the node IDs are consecutive integers starting from 0:

```python
import pandas as pd
import torch_graph_force

# The first argument is a dataframe of edges with at least two columns for source and target nodes.
# By default, column names "source", "target" and "weight" are taken as source nodes, target nodes and edge weights.
df = pd.DataFrame([[0, 1], [1, 2], [2, 3]], columns=['source', 'target'])
# Having a column for edge weights is optional. If the column for edge weights does not exist, 1.0 will be used for all edges.
# The second argument is the number of nodes in case the node IDs are consecutive integers starting from 0.
n_nodes = 4
# Create a GraphDataset for the graph
ds = torch_graph_force.from_pandas_dataframe(
    df, n_nodes
)
```

If the node IDs are not consecutive integers, a list of node IDs must be provided:
```python
import pandas as pd
import torch_graph_force

df = pd.DataFrame([["A", "B"], ["B", "C"], ["C", "D"]], columns=['source', 'target'])
# Order of the nodes in "nodes" is used to map the node IDs to node indices.
nodes = ["A", "B", "C", "D"]

ds = torch_graph_force.from_pandas_dataframe(
    df, nodes
)
# the dataset's order follows the order of the provided list of nodes. In this example, calling  ds[0] will return the data for node "A" and ds[1] for node "B"
# List of nodes can be access with ds.nodes
print(ds.nodes)
```
### Compute Graph Layout

Once having the graph dataset ready, we can feed the dataset to `spring_layout` to compute the graph layout.

```python

pos = torch_graph_force.spring_layout(
    ds
)
# pos is a numpy array of size (n_nodes, n_dim)
# each row represents the position of a node with corresponding index
print(pos)
# if node IDs are not consecutive integers, the nodes' positions can be obtained from the node list
node_pos = {nid: pos[idx] for idx, nid in enumerate(ds.nodes)}
```

Optional arguments for `spring_layout`:
- `batch_size`: number of nodes to process in a batch. Larger batch size usually speeds up the processing, but it consumes more memory. (default: 64)
- `iterations`: Maximum number of iterations taken. (default: 50)
- `num_workers`: number of workers to fetch data from GraphDataset. If device is "cuda", `num_workers` must be 0. (default: 0)
- `device`: the device to store the graph and the layout model. If None, it's "cuda" if cuda is available otherwise "cpu". (default: None)
- `iteration_progress`: monitor the progress of each iteration, it's useful for large graph. (default: False)
- `layout_config`: additional config for the layout model. (default: {})

The layout model has some parameters with default values:
```python
default_layout_config = {
    # Tensor of shape (n_nodes, ndim) for initial positions
    "pos": None,
    # Optimal distance between nodes
    "k": None,
    # Dimension of the layout
    "ndim": 2,
    # Threshold for relative error in node position changes.
    "threshold": 1e-4,
}
```

Use the `layout_config` argument to change the parameters if needed. The example below provides intial positions for the layout model:
```python
n_nodes = len(ds)
n_dim = 2
# Generate initial positions for the nodes
init_pos = np.random.rand(n_nodes, n_dim)
pos = torch_graph_force.spring_layout(
    ds,
    layout_config={"pos": init_pos}
)
```
## Benchmarks

The implementation from `torch-graph-force` **without GPU acceleration** is 1.5x faster than Networkx's implementation.

![CPU Benchmark](/assets/cpu-benchmark.jpg)

GPU accelerated `torch-graph-force` can compute layouts of graphs with 100k nodes within minutes. The benchmark was conducted with Tesla P100.

![GPU Benchmark](/assets/gpu-benchmark.jpg)

Code for the benchmarks can be found [here](/torch_graph_force/benchmark.py)