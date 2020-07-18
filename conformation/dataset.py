""" PyTorch dataset classes for atomic pairwise distance matrix data. """
import numpy as np
from typing import List, Dict, Tuple

from rdkit import Chem
from scipy import sparse
import torch
from torch.utils.data import Dataset

from conformation.data_pytorch import Data
from conformation.distance_matrix import distmat_to_vec
from conformation.relational import RelationalNetwork


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


# Dataset class
class GraphDataset(Dataset):
    """
    Dataset class for loading molecular graphs and pairwise distance targets.
    """

    def __init__(self, metadata: List[Dict[str, str]], atom_types: List[int] = None, bond_types: List[float] = None,
                 target: bool = True):
        """
        Custom dataset for molecular graphs.
        :param metadata: Metadata contents.
        :param atom_types: List of allowed atomic numbers.
        :param bond_types: List of allowed bond types.
        """
        super(Dataset, self).__init__()
        if bond_types is None:
            self.bond_types = [0., 1., 1.5, 2., 3.]
        if atom_types is None:
            self.atom_types = [1, 6, 7, 8, 9]
        self.metadata = metadata
        self.target = target

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx) -> Data:
        """ Output a data object with node features, edge connectivity, and target vector"""
        data = Data()  # Create data object

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

        # Extract atom indices participating in bonds and bond types
        bonds = []
        bond_types = []
        for bond in mol.GetBonds():
            bonds.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
            bond_types.append([bond_to_one_hot[bond.GetBondTypeAsDouble()]])

        # Compute edge attributes: 1 indicates presence of bond, 0 no bond. This is concatenated with one-hot bond feat.
        full_edges = [list(data.edge_index[:, i].numpy()) for i in range(data.edge_index.shape[1])]
        no_bond = np.concatenate([np.array([0]), bond_to_one_hot[0]])
        a = np.array([1])
        edge_attr = [np.concatenate([a, bond_types[bonds.index(full_edges[i])][0]]) if full_edges[i] in bonds else
                     no_bond for i in range(len(full_edges))]
        data.edge_attr = torch.tensor(edge_attr, dtype=torch.float)

        # Vertex features: one-hot representation of atomic number
        # Create one-hot encoding
        one_hot_vertex_features = np.zeros((len(self.atom_types), len(self.atom_types)))
        np.fill_diagonal(one_hot_vertex_features, 1.)
        atom_to_one_hot = dict()
        for i in range(len(self.atom_types)):
            atom_to_one_hot[self.atom_types[i]] = one_hot_vertex_features[i]

        # one_hot_vertex_features = np.zeros((self.max_atomic_num, self.max_atomic_num))
        # np.fill_diagonal(one_hot_vertex_features, 1.)
        one_hot_features = np.array([atom_to_one_hot[atom.GetAtomicNum()] for atom in mol.GetAtoms()])
        data.x = torch.tensor(one_hot_features, dtype=torch.float)

        if self.target:
            # Target: 1-D tensor representing average inter-atomic distance for each edge
            target = np.loadtxt(self.metadata[idx]['target'])
            data.y = torch.tensor(target, dtype=torch.float)

        # # Unique ID
        # data.uid = self.metadata[idx]['smiles']  # Unique id

        return data

    def __repr__(self) -> str:
        return '{}({})'.format(self.__class__.__name__, len(self))


class CNFDataset(Dataset):
    """
    Dataset class for loading atomic pairwise distance information for molecules.
    """

    def __init__(self, metadata: List[Dict[str, str]], padding_dim: int, condition_dim: int = 256):
        super(Dataset, self).__init__()
        self.metadata = metadata
        self.padding_dim = padding_dim
        self.condition_dim = condition_dim

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _, data = distmat_to_vec(self.metadata[idx]['path'])
        dist_vec = torch.from_numpy(data)
        dist_vec = dist_vec.type(torch.float32)

        condition = np.load(self.metadata[idx]['condition'])
        condition = torch.from_numpy(condition)
        condition = condition.type(torch.float32)
        padding = torch.zeros([self.padding_dim, self.condition_dim])
        padding[0:condition.shape[0], :] = condition
        condition = padding

        num_dist = torch.tensor(dist_vec.shape[0])

        padding = torch.zeros(self.padding_dim)
        padding[:dist_vec.shape[0]] = dist_vec
        dist_vec = padding

        return dist_vec, condition, num_dist

    def __repr__(self) -> str:
        return '{}({})'.format(self.__class__.__name__, len(self))
