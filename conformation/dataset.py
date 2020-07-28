""" PyTorch dataset classes for molecular data. """
import numpy as np
from typing import Dict, List, Tuple

from rdkit import Chem
from rdkit.Chem import rdmolops
from scipy import sparse
import torch
from torch.utils.data import Dataset

from conformation.distance_matrix import distmat_to_vec
from conformation.graph_data import Data


class MolDataset(Dataset):
    """
    Dataset class for loading atomic pairwise distance information for molecules.
    """

    def __init__(self, metadata: List[Dict[str, str]]):
        super(Dataset, self).__init__()
        self.metadata = metadata

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> torch.Tensor:
        _, data = distmat_to_vec(self.metadata[idx]['path'])
        data = torch.from_numpy(data)
        data = data.type(torch.float32)

        return data

    def __repr__(self) -> str:
        return '{}({})'.format(self.__class__.__name__, len(self))


class GraphDataset(Dataset):
    """
    Dataset class for loading molecular graphs and pairwise distance targets.
    """

    def __init__(self, metadata: List[Dict[str, str]], atom_types: List[int] = None, bond_types: List[float] = None,
                 target: bool = True, max_path_length: int = 10):
        """
        Custom dataset for molecular graphs.
        :param metadata: Metadata contents.
        :param atom_types: List of allowed atomic numbers.
        :param bond_types: List of allowed bond types.
        :param target: Whether or not to load target data from metadata into Data() object.
        :param max_path_length: Maximum shortest path length between any two atoms in a molecule in the dataset.
        """
        super(Dataset, self).__init__()
        if bond_types is None:
            self.bond_types = [0., 1., 1.5, 2., 3.]
        if atom_types is None:
            self.atom_types = [1, 6, 7, 8, 9]
        self.metadata = metadata
        self.target = target
        self.max_path_length = max_path_length

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx) -> Data:
        """
        Output a data object with node features, edge connectivity, and (optionally) target.
        :param idx: Which item to load.
        :return: Data() object.
        """
        data = Data()

        # Molecule from SMILES string
        smiles = self.metadata[idx]['smiles']  # Read smiles string
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        num_atoms = mol.GetNumAtoms()

        # Compute edge connectivity in COO format corresponding to a complete graph on num_nodes
        complete_graph = np.ones([num_atoms, num_atoms])  # Create an auxiliary complete graph
        complete_graph = np.triu(complete_graph, k=1)  # Compute an upper triangular matrix of the complete graph
        complete_graph = sparse.csc_matrix(complete_graph)  # Compute a csc style sparse matrix from this graph
        row, col = complete_graph.nonzero()  # Extract the row and column indices corresponding to non-zero entries
        row = torch.tensor(row, dtype=torch.long)
        col = torch.tensor(col, dtype=torch.long)
        data.edge_index = torch.stack([row, col])  # Edge connectivity in COO format (all possible edges)

        # Edge features
        # Create one-hot encoding
        one_hot_bond_features = np.zeros((len(self.bond_types), len(self.bond_types)))
        np.fill_diagonal(one_hot_bond_features, 1.)
        bond_to_one_hot = dict()
        for i in range(len(self.bond_types)):
            bond_to_one_hot[self.bond_types[i]] = one_hot_bond_features[i]

        # Create one-hot encoding for shortest path length
        one_hot_shortest_path_features = np.zeros((self.max_path_length, self.max_path_length))
        np.fill_diagonal(one_hot_shortest_path_features, 1.)
        bond_to_shortest_path_one_hot = dict()
        for i in range(self.max_path_length):
            bond_to_shortest_path_one_hot[i + 1] = one_hot_shortest_path_features[i]

        # Extract atom indices participating in bonds and bond types
        bonds = []
        bond_types = []
        for bond in mol.GetBonds():
            bonds.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
            bond_types.append([bond_to_one_hot[bond.GetBondTypeAsDouble()]])

        # Compute edge attributes: 1 indicates presence of bond, 0 no bond. This is concatenated with one-hot bond feat.
        full_edges = [list(data.edge_index[:, i].numpy()) for i in range(data.edge_index.shape[1])]
        shortest_path_lengths = [bond_to_shortest_path_one_hot[len(rdmolops.GetShortestPath(mol, int(x[0]), int(x[1])))
                                                               - 1] for x in full_edges]
        no_bond = np.concatenate([np.array([0]), bond_to_one_hot[0]])
        a = np.array([1])
        edge_attr = [np.concatenate([a, bond_types[bonds.index(full_edges[i])][0], shortest_path_lengths[i]]) if
                     full_edges[i] in bonds else np.concatenate([no_bond, shortest_path_lengths[i]]) for i in
                     range(len(full_edges))]
        data.edge_attr = torch.tensor(edge_attr, dtype=torch.float)

        # Vertex features: one-hot representation of atomic number
        # Create one-hot encoding
        one_hot_vertex_features = np.zeros((len(self.atom_types), len(self.atom_types)))
        np.fill_diagonal(one_hot_vertex_features, 1.)
        atom_to_one_hot = dict()
        for i in range(len(self.atom_types)):
            atom_to_one_hot[self.atom_types[i]] = one_hot_vertex_features[i]

        # Add the vertex features as one-hot vectors
        one_hot_features = np.array([atom_to_one_hot[atom.GetAtomicNum()] for atom in mol.GetAtoms()])
        data.x = torch.tensor(one_hot_features, dtype=torch.float)

        # Target
        if self.target:
            # Target: 1-D tensor representing average inter-atomic distance for each edge
            target = np.load(self.metadata[idx]['target'])
            data.y = torch.tensor(target, dtype=torch.float)

        # UID
        data.uid = torch.tensor([int(self.metadata[idx]['uid'])])

        return data

    def __repr__(self) -> str:
        return '{}({})'.format(self.__class__.__name__, len(self))


class CNFDataset(Dataset):
    """
    Dataset class for loading atomic pairwise distance information for molecules for a conditional normalizing flow.
    """

    def __init__(self, metadata: List[Dict[str, str]], padding_dim: int = 528, condition_dim: int = 256):
        """
        :param metadata: Metadata.
        :param padding_dim: Padding size for all distance vectors and conditions.
        :param condition_dim: Dimensionality of the hidden size for the condition matrix.
        """
        super(Dataset, self).__init__()
        self.metadata = metadata
        self.padding_dim = padding_dim
        self.condition_dim = condition_dim

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :param idx: # of data item to retrieve.
        :return: Padded distance vector, condition matrix, and # of pairwise distances in the molecule.
        """
        # Load the pairwise distance matrix
        _, data = distmat_to_vec(self.metadata[idx]['path'])
        dist_vec = torch.from_numpy(data)
        dist_vec = dist_vec.type(torch.float32)

        # Compute the number of pairwise distances before padding
        num_dist = torch.tensor(dist_vec.shape[0])

        # Pad the pairwise distances vector
        padding = torch.zeros(self.padding_dim)
        padding[:dist_vec.shape[0]] = dist_vec
        dist_vec = padding

        # Load the condition matrix
        condition = np.load(self.metadata[idx]['condition'])
        condition = torch.from_numpy(condition)
        condition = condition.type(torch.float32)

        # Pad the condition matrix
        padding = torch.zeros([self.padding_dim, self.condition_dim])
        padding[0:condition.shape[0], :] = condition
        condition = padding

        return dist_vec, condition, num_dist

    def __repr__(self) -> str:
        return '{}({})'.format(self.__class__.__name__, len(self))
