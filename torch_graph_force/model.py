import numpy as np
import torch
from torch import nn


class SpringLayout(nn.Module):
    def __init__(
        self, n_nodes, pos=None, k=None, ndim=2, iterations=50, threshold=1e-4
    ):
        super(SpringLayout, self).__init__()
        self.n_nodes = n_nodes
        self.ndim = ndim
        self.iterations = iterations
        self.threshold = threshold

        if pos is not None:
            if isinstance(pos, np.ndarray):
                pos = torch.tensor(pos, dtype=torch.float64)
            assert pos.size(0) == self.n_nodes
            assert pos.size(1) == self.ndim
            self.pos = nn.Parameter(pos.clone().type(torch.float64))
            self.init_pos = False
        else:
            self.pos = nn.Parameter(
                torch.empty((self.n_nodes, ndim), dtype=torch.float64)
            )
            self.init_pos = True

        if k is None:
            self.k = np.sqrt(1.0 / self.n_nodes)
        else:
            self.k = k

        self.displacement = nn.Parameter(
            torch.empty((self.n_nodes, ndim), dtype=torch.float64)
        )
        # We won't do any backprop
        self.pos.requires_grad = False
        self.displacement.requires_grad = False
        self._init_weights()
        # the initial "temperature"  is about .1 of domain area (=1x1)
        # this is the largest step allowed in the dynamics.
        self.t = (
            torch.max(self.pos.max(dim=0).values - self.pos.min(dim=0).values) * 0.1
        )
        # simple cooling scheme.
        # linearly step down by dt on each iteration so last iteration is size dt.
        self.dt = self.t / (iterations + 1)

    def _init_weights(self):
        if self.init_pos:
            nn.init.uniform_(self.pos, 0, 1.0)
        nn.init.zeros_(self.displacement)

    def reset_displacement(self):
        nn.init.zeros_(self.displacement)

    def forward(self, idxes, edges, weights):
        pos_batch = self.pos[idxes, :]
        # difference between selected rows and all rows: (batch_size, n_nodes, ndim)
        delta = pos_batch.unsqueeze(1) - self.pos
        distance = torch.sqrt((delta**2).sum(dim=2))
        distance = torch.where(distance < 0.01, 0.01, distance)
        # delta: (batch_size, n_nodes, ndim)
        # distance: (batch_size, n_nodes)
        # Calculate repulsion between nodes
        repulsion = delta * (self.k * self.k / distance**2).unsqueeze(2)
        # Collect pairs to calculate attraction
        # Note: only connected nodes have attraction force
        attraction_dist = torch.gather(distance, 1, edges)
        # Since delta is a n-dim vector,
        # duplicate the indices to gather the whole vectors
        attraction_delta = torch.gather(
            delta, 1, edges.unsqueeze(2).repeat(1, 1, self.ndim)
        )
        attraction = -attraction_delta * (
            (attraction_dist * weights) / self.k
        ).unsqueeze(2)
        self.displacement[idxes, :] = repulsion.sum(dim=1) + attraction.sum(dim=1)

    def update(self):
        """
        Update the positions after displacements are computed.
        Return:
            True if the norm of delta_pos is lower than threshold.
            Otherwise, return False
        """
        assert self.t > 1e-10, (
            f"The number of update calls"
            f" might have exceeded #iterations ({self.iterations})"
        )
        length = torch.sqrt((self.displacement**2).sum(dim=1))
        length = torch.where(length > 0.01, length, 0.01)
        delta_pos = self.displacement * self.t / length.unsqueeze(1)
        self.pos += delta_pos
        self.t -= self.dt
        self.reset_displacement()
        if (torch.norm(delta_pos) / self.n_nodes).item() < self.threshold:
            return True
        else:
            return False
