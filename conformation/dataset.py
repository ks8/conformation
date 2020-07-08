""" PyTorch dataset class for atomic pairwise distance matrix data. """
import numpy as np
from typing import List, Dict

import torch
# noinspection PyUnresolvedReferences
from torch.utils.data import Dataset

from conformation.data_pytorch import Data


class MolDataset(Dataset):
    """
    Dataset class for loading atomic pairwise distance information for molecules.
    """

    def __init__(self, metadata: List[Dict[str, str]]):
        """
        :param metadata: Metadata JSON file.
        """
        super(Dataset, self).__init__()
        self.metadata = metadata

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> torch.Tensor:
        distmat = np.loadtxt(self.metadata[idx]['path'])
        data = []
        num_atoms = distmat.shape[0]
        for m in range(num_atoms):
            for n in range(1, num_atoms):
                if n > m:
                    data.append(distmat[m][n])
        data = torch.from_numpy(np.array(data))
        data = data.type(torch.float32)

        return data

    def __repr__(self) -> str:
        return '{}({})'.format(self.__class__.__name__, len(self))


# Dataset class
class GraphDataset(Dataset):
    """
    Test.
    """
    def __init__(self, metadata, max_num_nodes=None, transform=None):
        """
        Custom dataset for molecular graphs
        :param metadata: Metadata contents
        :param transform: Transform to apply to the data (can be a Compose() object)
        """
        super(Dataset, self).__init__()
        self.metadata = metadata
        self.transform = transform
        self.max_num_nodes = max_num_nodes

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        """ Output a data object with node features, edge connectivity, and target vector"""
        data = Data()  # Create data object

        # Read in data
        graph = nx.read_gml(self.metadata[idx]['graph_path'], destringizer=int)  # Read in the partial graph
        target = np.loadtxt(self.metadata[idx]['target_path'])  # Read in the target matrix of allowed edges
        num_nodes = len(list(graph.nodes))  # Compute the number of nodes in the graph

        # Compute node features
        #         one_hot_features = np.zeros((num_nodes, num_nodes))  # Create a one-hot mapping for remaining node degree
        one_hot_features = np.zeros(
            (self.max_num_nodes, self.max_num_nodes))  # Create a one-hot mapping for remaining node degree

        np.fill_diagonal(one_hot_features, 1.)
        raw_features = np.array(
            [[graph.nodes[i]['deg'], graph.degree[i], graph.nodes[i]['deg'] - graph.degree[i]] for i in
             range(num_nodes)])  # Compute the original node degree, current node degree, and remaining node degree
        one_hot_features = np.array([one_hot_features[graph.nodes[i]['deg'] - graph.degree[i]] for i in
                                     range(num_nodes)])  # Compute one-hot encoding for remaining node degree
        data.x = torch.tensor(np.concatenate((raw_features, one_hot_features), axis=1),
                              dtype=torch.float)  # Node features: concatenate the raw features and one-hot encoded features

        # Compute targets
        data.y = torch.tensor(target[np.triu_indices(num_nodes, k=1)],
                              dtype=torch.float)  # Target: 1-d tensor derived from upper triangle (not including main diagonal) of the target matrix

        # Compute graph id
        data.uid = torch.tensor([int(self.metadata[idx]['uid'])])  # Unique id

        # Compute edge connectivity in COO format corresponding to a complete graph on num_nodes
        complete_graph = np.ones([num_nodes, num_nodes])  # Create an auxiliary complete graph
        complete_graph = np.triu(complete_graph,
                                 k=1)  # Compute an upper triangular matrix of the complete graph, with zeros on main diagonal
        complete_graph = scipy.sparse.csc_matrix(complete_graph)  # Compute a csc style sparse matrix from this graph
        row, col = complete_graph.nonzero()  # Extract the row and column indices corresponding to non-zero entries
        row = torch.tensor(row, dtype=torch.long)
        col = torch.tensor(col, dtype=torch.long)
        data.edge_index = torch.stack([row, col],
                                      dim=0)  # Edge connectivity in COO format (includes *all* possible edges in graphs with a given num_nodes)

        # Compute edge attributes, i.e., 1 if the edge is actually present in the graph, 0 otherwise
        data.edge_attr = torch.tensor(
            [[i] for i in np.array(nx.to_numpy_matrix(graph))[np.triu_indices(num_nodes, k=1)]],
            dtype=torch.float)  # Attributes: 1-d tensor derived from upper triangle (not including main diagonal) of actual edges that are present in the graph

        # Transform
        data = data if self.transform is None else self.transform(data)

        return data

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, len(self))
