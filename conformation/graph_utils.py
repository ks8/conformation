""" Neural network graph auxiliary functions. """
import torch


# loop_PyTorch
def contains_self_loops(edge_index):
    """Returns a boolean for existence of self-loops in the graph"""
    row, col = edge_index
    mask = row == col
    return mask.sum().item() > 0


def remove_self_loops(edge_index, edge_attr=None):
    """Remove self-loops from the edge_index and edge_attr attributes"""
    row, col = edge_index
    mask = row != col
    edge_attr = edge_attr if edge_attr is None else edge_attr[mask]
    mask = mask.expand_as(edge_index)
    edge_index = edge_index[mask].view(2, -1)

    return edge_index, edge_attr


# isolated_PyTorch
def contains_isolated_nodes(edge_index, num_nodes):
    """Check if there are any isolated nodes"""
    (row, _), _ = remove_self_loops(edge_index)
    return torch.unique(row).size(0) < num_nodes


def to_undirected(edge_index, num_nodes):
    """
    Returns an undirected (bidirectional) COO format connectivity matrix from an original matrix given by edge_index
    """

    row, col = edge_index
    row, col = torch.cat([row, col], dim=0), torch.cat([col, row], dim=0)
    unique, inv = torch.unique(row*num_nodes + col, sorted=True, return_inverse=True)
    perm = torch.arange(inv.size(0), dtype=inv.dtype, device=inv.device)
    perm = inv.new_empty(unique.size(0)).scatter_(0, inv, perm)
    index = torch.stack([row[perm], col[perm]])

    return index
