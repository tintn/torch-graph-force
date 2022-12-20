import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader

from .model import SpringLayout
from .utils import rescale_layout


def collate_fn(batch):
    idxes, neighbors, weights = zip(*batch)
    idxes = torch.stack(idxes)
    # Create a matrix for col indices along a row.
    # E.g: Assume the batch have three rows,
    #      col indices for non-zero elements are:
    #    0 -> [1, 3, 7]
    #    1 -> [0, 2]
    #    2 -> [4]
    #    Output:
    #    [[1, 3, 7],
    #     [0, 2, 0],
    #     [4, 0, 0]]
    # Because the numbers of non-zero elements can be different between rows,
    # the rows are padded with zeros to form a matrix
    edges = nn.utils.rnn.pad_sequence(list(neighbors), batch_first=True)
    # Do the same for weights and also pad the matrix with zeros
    weights = nn.utils.rnn.pad_sequence(list(weights), batch_first=True)
    return idxes, edges, weights


def spring_layout(
    ds,
    batch_size=64,
    iterations=50,
    num_workers=0,
    device=None,
    iteration_progress=False,
    layout_config={},
):
    """
    Implementation of Fruchterman-Reingold algorithm based on Networkx's implementation
    Args:
        ds (GraphDataset): GraphDataset for the graph
        batch_size (int, optional): number of rows to process per batch
        iterations (int, optional): Maximum number of iterations taken
        num_workers (int, optional): #workers to fetch nodes from GraphDataset
                                    (if device is "cuda", num_workers must be 0)
        device (str, optional): device to store the graph data and the layout model
        iteration_progress: If True, monitor each iteration progress
                            (It's useful for large graphs)
        layout_config (dict, optional): additional config for the layout model
    """

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
    default_layout_config.update(layout_config)
    layout_config = default_layout_config

    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

    model = SpringLayout(len(ds), iterations=iterations, **layout_config)
    model.to(device)

    dataloader = DataLoader(
        ds, batch_size=batch_size, num_workers=num_workers, collate_fn=collate_fn
    )
    for i in range(iterations):
        if iteration_progress:
            print(f"Iteration {i+1}")
            iter_ = tqdm.tqdm(iter(dataloader))
        else:
            iter_ = iter(dataloader)

        for batch in iter_:
            idxes, edges, weights = batch
            model(idxes, edges, weights)
        if model.update():
            # The relative error is below the threshold
            break
    if "cuda" in device:
        pos = model.pos.detach().cpu().numpy()
    else:
        pos = model.pos.detach().numpy()
    pos = rescale_layout(pos)
    return pos
