from math import pi

import torch
from torch_geometric.nn import MetaLayer


class MetaLayer_mod(MetaLayer):
    """Modified MetaLayer which allows taking in a tuple as edge attributes. Also:
      - allows returning of endpoints.
      - Order of update is global -> node -> edge.
    """
    def forward(self, x, edge_index, edge_attr=None, u=None, batch=None):
        """"""
        endpoints = {}
        row, col = edge_index

        if self.global_model is not None:
            u = self.global_model(x, edge_index, edge_attr, u, batch)

        if self.node_model is not None:
            x, node_endpoints = self.node_model(x, edge_index, edge_attr, u, batch)
            endpoints.update(node_endpoints)

        if self.edge_model is not None:
            edge_attr = self.edge_model(tuple(x[i][row] for i in range(len(x))),
                                        tuple(x[i][col] for i in range(len(x))),
                                        edge_attr, u,
                                        batch if batch is None else batch[row])

        return x, edge_attr, u, endpoints


class SquashRot(torch.nn.Module):
    """Squashes the rotation part of the se(3) vector such it has a maximum magnitude of pi"""
    def __init__(self, max_mag=pi):
        super().__init__()
        self.max_mag = max_mag

    def forward(self, x):
        assert x.shape[-1] == 3 or x.shape[-1] == 6  # Must be so(3) or se(3)
        x_trans = x[..., 0:-3]  # Empty if so(3)
        x_rot = x[..., -3:]

        mag_sq = torch.sum(x_rot ** 2, dim=-1, keepdim=True)
        mag = torch.sqrt(mag_sq)
        x_rot_out = self.max_mag * (mag_sq / (1.0 + mag_sq)) * (x_rot / mag)

        out = torch.cat([x_trans, x_rot_out], dim=-1)
        return out