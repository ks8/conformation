""" PyTorch dataset classes for molecular data. """
import itertools
from typing import Dict, List, Tuple, Union
import numpy as np

import torch
from rdkit import Chem
# noinspection PyUnresolvedReferences
from rdkit.Chem import AllChem, rdmolops, rdPartialCharges, rdForceFieldHelpers, rdchem
from scipy import sparse
from torch.utils.data import Dataset

from conformation.distance_matrix import distmat_to_vec
from conformation.graph_data import Data


def to_one_hot(x: int, vals: Union[List, range]) -> List:
    """
    Return a one-hot vector.
    :param x: Data integer.
    :param vals: List of possible data values.
    :return: One-hot vector as list.
    """
    return [x == v for v in vals]


class TestDataset(Dataset):
    """
    Test.
    """
    def __init__(self, data: torch.Tensor, condition: torch.Tensor = None):
        super(Dataset, self).__init__()
        self.data = data
        self.condition = condition

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        sample = self.data[idx]
        if self.condition is not None:
            return sample, self.condition
        else:
            return sample


class BasicDataset(Dataset):
    """
    Dataset class for loading non-molecular data organized as numpy arrays
    """
    def __init__(self, metadata: List[Dict[str, str]], condition: bool = False):
        super(Dataset, self).__init__()
        self.metadata = metadata
        self.condition = condition

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        data = torch.load(self.metadata[idx]['path'])

        if self.condition:
            condition = torch.load(self.metadata[idx]['condition'])
            return data, condition
        else:
            return data


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
        # noinspection PyTypeChecker
        data = data.type(torch.float32)

        return data

    def __repr__(self) -> str:
        return '{}({})'.format(self.__class__.__name__, len(self))


class GraphDataset(Dataset):
    """
    Dataset class for loading molecular graphs and pairwise distance targets.
    """

    # noinspection PyUnresolvedReferences
    def __init__(self, metadata: List[Dict[str, str]], atom_types: List[int] = None, bond_types: List[float] = None,
                 target: bool = True, max_path_length: int = 10, atomic_num: bool = True, partial_charge: bool = True,
                 mmff_atom_types_one_hot: bool = True, valence_types: List[int] = None, valence: bool = True,
                 aromatic: bool = True, hybridization: bool = True, assign_stereo: bool = True,
                 charge_types: List[int] = None, formal_charge: bool = True, r_covalent: bool = True,
                 r_vanderwals: bool = True, default_valence: bool = True, max_ring_size: int = 8,
                 rings: bool = True, chirality: bool = True, mmff94_atom_types: List[int] = None,
                 hybridization_types: List[Chem.HybridizationType] = None,
                 chi_types: List[rdchem.ChiralType] = None, improved_architecture: bool = False, max_atoms: int = 26,
                 degree_types: List[int] = None, degree: bool = True, num_hydrogen_types: List[int] = None,
                 num_hydrogen: bool = True, num_radical_electron_types: List[int] = None,
                 num_radical_electron: bool = True, conjugated: bool = True, bond_type: bool = True,
                 bond_ring: bool = True, bond_stereo: bool = True, bond_stereo_types: List[int] = None,
                 shortest_path: bool = True, same_ring: bool = True, autoencoder: bool = False):
        """
        Custom dataset for molecular graphs.
        :param metadata: Metadata contents.
        :param atom_types: List of allowed atomic numbers.
        :param bond_types: List of allowed bond types.
        :param target: Whether or not to load target data from metadata into Data() object.
        :param max_path_length: Maximum shortest path length between any two atoms in a molecule in the dataset.
        :param partial_charge: Whether or not to include Gasteiger Charge as a vertex feature.\
        :param mmff_atom_types_one_hot: Whether or not to include MMFF94 atom types as vertex features.
        :param valence_types: List of allowed total valence numbers.
        :param valence: Whether or not to include total valence as a vertex feature.
        :param aromatic: Whether or not to include aromaticity as a vertex feature.
        :param hybridization: Whether or not to include hybridization as a vertex feature.
        :param assign_stereo: Whether or not to include stereochemistry information.
        :param charge_types: Formal charge types.
        :param formal_charge: Whether or not to include formal charge as a vertex feature.
        :param r_covalent: Whether or not to include covalent radius as a vertex feature.
        :param r_vanderwals: Whether or not to include vanderwals radius as a vertex feature.
        :param default_valence: Whether or not to include default valence as a vertex feature.
        :param max_ring_size: Maximum ring size.
        :param rings: Whether or not to include ring size as a vertex feature.
        :param chirality: Whether or not to include chirality as a vertex feature.
        :param mmff94_atom_types: MMFF94 atom types.
        :param hybridization_types: Hybridization types.
        :param chi_types: Chiral tag types.
        :param improved_architecture: Whether or not to use Jonas improved relational architecture.
        :param max_atoms: Maximum number of atoms for a given molecule in the dataset (improved_architecture = True)
        :param degree_types: Atomic degree types.
        :param degree: Whether or not to include degree as a vertex feature.
        :param num_hydrogen_types: List of allowed number of H atoms (including neighbors).
        :param num_hydrogen: Whether or not to include number of (neighboring) Hs as a vertex feature.
        :param num_radical_electron_types: List of allowed number of radical electrons.
        :param num_radical_electron: Whether or not to include number of radical electrons as a vertex feature.
        :param conjugated: Whether or not to include conjugated as an edge feature.
        :param bond_type: Whether or not to include bond type as an edge feature.
        :param bond_ring: Whether or not to include bond being in ring as an edge feature.
        :param bond_stereo: Whether or not to include bond stereo as an edge feature.
        :param bond_stereo_types: List of bond stereo types.
        :param shortest_path: Whether or not to include shortest path length as a bond feature.
        :param same_ring: Whether or not to include same ring as bond feature.
        :param autoencoder: Whether or not to prepare data for autoencoder training.
        """
        super(Dataset, self).__init__()
        if bond_types is None:
            self.bond_types = [0., 1., 1.5, 2., 3.]
        else:
            self.bond_types = bond_types
        if atom_types is None:
            self.atom_types = [1, 6, 7, 8, 9]
        else:
            self.atom_types = atom_types
        self.metadata = metadata
        self.target = target
        self.max_path_length = max_path_length
        self.atomic_num = atomic_num
        self.partial_charge = partial_charge
        if mmff94_atom_types is None:
            self.mmff94_atom_types = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 16, 17, 18, 20, 21, 22, 23, 24, 25, 26,
                                      27, 28, 29, 30, 31, 32, 33, 37, 38, 39, 40, 42, 43, 44, 46, 48, 59, 62, 63, 64,
                                      65, 66, 70, 71, 72, 74, 75, 78]
        else:
            self.mmff94_atom_types = mmff94_atom_types
        self.mmff_atom_types_one_hot = mmff_atom_types_one_hot
        if valence_types is None:
            self.valence_types = [1, 2, 3, 4, 5, 6]
        else:
            self.valence_types = valence_types
        self.valence = valence
        self.aromatic = aromatic
        if hybridization_types is None:
            self.hybridization_types = [Chem.HybridizationType.S,
                                        Chem.HybridizationType.SP,
                                        Chem.HybridizationType.SP2,
                                        Chem.HybridizationType.SP3,
                                        Chem.HybridizationType.SP3D,
                                        Chem.HybridizationType.SP3D2,
                                        Chem.HybridizationType.UNSPECIFIED]
        else:
            self.hybridization_types = hybridization_types
        self.hybridization = hybridization
        self.assign_stereo = assign_stereo
        if charge_types is None:
            self.charge_types = [-1, 0, 1]
        else:
            self.charge_types = charge_types
        self.formal_charge = formal_charge
        self.r_covalent = r_covalent
        self.r_vanderwals = r_vanderwals
        self.default_valence = default_valence
        self.max_ring_size = max_ring_size
        self.rings = rings
        if chi_types is None:
            self.chi_types = list(rdchem.ChiralType.values.values())
        else:
            self.chi_types = chi_types
        self.chirality = chirality
        self.improved_architecture = improved_architecture
        self.max_atoms = max_atoms
        if degree_types is None:
            self.degree_types = [1, 2, 3, 4]
        else:
            self.degree_types = degree_types
        self.degree = degree
        if num_hydrogen_types is None:
            self.num_hydrogen_types = [0, 1, 2, 3]
        else:
            self.num_hydrogen_types = num_hydrogen_types
        self.num_hydrogen = num_hydrogen
        if num_radical_electron_types is None:
            self.num_radical_electron_types = [0, 1, 2]
        else:
            self.num_radical_electron_types = num_radical_electron_types
        self.num_radical_electron = num_radical_electron
        self.conjugated = conjugated
        self.bond_type = bond_type
        self.bond_ring = bond_ring
        self.bond_stereo = bond_stereo
        if bond_stereo_types is None:
            self.bond_stereo_types = list(rdchem.BondStereo.values.values())
        else:
            self.bond_stereo_types = bond_stereo_types
        self.shortest_path = shortest_path
        self.same_ring = same_ring
        self.autoencoder = autoencoder

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx) -> Union[Data, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Output a data object with node features, edge connectivity, and (optionally) target.
        :param idx: Which item to load.
        :return: Data() object.
        """
        data = Data()

        # Molecule from binary
        # noinspection PyUnresolvedReferences
        mol = Chem.Mol(open(self.metadata[idx]['binary'], "rb").read())
        num_atoms = mol.GetNumAtoms()

        # Target
        if self.target:
            # Target: 1-D tensor representing average inter-atomic distance for each edge
            target = np.load(self.metadata[idx]['target'])
            data.y = torch.tensor(target, dtype=torch.float)

        # Compute edge connectivity in COO format corresponding to a complete graph on num_nodes
        complete_graph = np.ones([num_atoms, num_atoms])  # Create an auxiliary complete graph
        complete_graph = np.triu(complete_graph, k=1)  # Compute an upper triangular matrix of the complete graph
        complete_graph = sparse.csc_matrix(complete_graph)  # Compute a csc style sparse matrix from this graph
        row, col = complete_graph.nonzero()  # Extract the row and column indices corresponding to non-zero entries
        row = torch.tensor(row, dtype=torch.long)
        col = torch.tensor(col, dtype=torch.long)
        data.edge_index = torch.stack([row, col])  # Edge connectivity in COO format (all possible edges)

        # Edge features
        edge_features = []

        edge_count = 0
        for a, b in itertools.combinations(list(np.arange(num_atoms)), 2):
            bond_feature = []
            bond = mol.GetBondBetweenAtoms(int(a), int(b))
            if bond is None:
                if self.bond_type:
                    bond_feature += [1] + [0]*len(self.bond_types)

                if self.conjugated:
                    bond_feature += [0]

                if self.bond_ring:
                    bond_feature += [0]

                if self.bond_stereo:
                    bond_feature += [0]*len(self.bond_stereo_types)

                if self.shortest_path:
                    path_len = len(rdmolops.GetShortestPath(mol, int(a), int(b))) - 1
                    bond_feature += to_one_hot(path_len - 1, range(self.max_path_length))

                if self.same_ring:
                    ring_info = list(mol.GetRingInfo().AtomRings())
                    membership = [int(a) in r and int(b) in r for r in ring_info]
                    if sum(membership) > 0:
                        bond_feature += [1]
                    else:
                        bond_feature += [0]

                if self.autoencoder:
                    # noinspection PyUnboundLocalVariable
                    bond_feature += [target[:, 0][edge_count]]

            else:
                if self.bond_type:
                    bond_feature += [0]
                    bond_feature += to_one_hot(bond.GetBondTypeAsDouble(), self.bond_types)

                if self.conjugated:
                    bond_feature += [bond.GetIsConjugated()]

                if self.bond_ring:
                    bond_feature += [bond.IsInRing()]

                if self.bond_stereo:
                    bond_feature += to_one_hot(bond.GetStereo(), self.bond_stereo_types)

                if self.shortest_path:
                    path_len = len(rdmolops.GetShortestPath(mol, int(a), int(b))) - 1
                    bond_feature += to_one_hot(path_len - 1, range(self.max_path_length))

                if self.same_ring:
                    ring_info = list(mol.GetRingInfo().AtomRings())
                    membership = [int(a) in r and int(b) in r for r in ring_info]
                    if sum(membership) > 0:
                        bond_feature += [1]
                    else:
                        bond_feature += [0]

                if self.autoencoder:
                    bond_feature += [target[:, 0][edge_count]]

            edge_count += 1

            edge_features.append(bond_feature)

        data.edge_attr = torch.tensor(edge_features, dtype=torch.float)

        # Vertex features
        # List to hold all vertex features
        vertex_features = []

        pt = Chem.GetPeriodicTable()

        if self.partial_charge:
            rdPartialCharges.ComputeGasteigerCharges(mol)

        mmff_p = None
        if self.mmff_atom_types_one_hot:
            # AllChem.EmbedMolecule(mol, maxAttempts=100000)
            # AllChem.MMFFOptimizeMolecule(mol)
            mmff_p = rdForceFieldHelpers.MMFFGetMoleculeProperties(mol)

        if self.assign_stereo:
            rdmolops.AssignStereochemistryFrom3D(mol)

        for i in range(num_atoms):
            atom = mol.GetAtomWithIdx(i)
            atom_feature = []

            if self.atomic_num:
                atom_feature += to_one_hot(atom.GetAtomicNum(), self.atom_types)

            if self.valence:
                atom_feature += to_one_hot(atom.GetTotalValence(), self.valence_types)

            if self.aromatic:
                atom_feature += [atom.GetIsAromatic()]

            if self.hybridization:
                atom_feature += to_one_hot(atom.GetHybridization(), self.hybridization_types)

            if self.partial_charge:
                gc = float(atom.GetProp('_GasteigerCharge'))
                if not np.isfinite(gc):
                    gc = 0.0
                atom_feature += [gc]

            if self.formal_charge:
                atom_feature += to_one_hot(atom.GetFormalCharge(), self.charge_types)

            if self.r_covalent:
                atom_feature += [pt.GetRcovalent(atom.GetAtomicNum())]

            if self.r_vanderwals:
                atom_feature += [pt.GetRvdw(atom.GetAtomicNum())]

            if self.default_valence:
                atom_feature += to_one_hot(pt.GetDefaultValence(atom.GetAtomicNum()), self.valence_types)

            if self.rings:
                atom_feature += [atom.IsInRingSize(r) for r in range(3, self.max_ring_size + 1)]

            if self.chirality:
                atom_feature += to_one_hot(atom.GetChiralTag(), self.chi_types)

            if self.mmff_atom_types_one_hot:
                if mmff_p is None:
                    atom_feature += [0] * len(self.mmff94_atom_types)
                else:
                    atom_feature += to_one_hot(mmff_p.GetMMFFAtomType(i), self.mmff94_atom_types)

            if self.degree:
                atom_feature += to_one_hot(atom.GetDegree(), self.degree_types)

            if self.num_hydrogen:
                atom_feature += to_one_hot(atom.GetTotalNumHs(), self.num_hydrogen_types)

            if self.num_radical_electron:
                atom_feature += to_one_hot(atom.GetNumRadicalElectrons(), self.num_radical_electron_types)

            vertex_features.append(atom_feature)

        data.x = torch.tensor(vertex_features, dtype=torch.float)

        # UID
        data.uid = torch.tensor([int(self.metadata[idx]['uid'])])

        if self.improved_architecture:
            # Vertex features
            v_in = data.x
            padding = torch.zeros([self.max_atoms, v_in.shape[1]])
            padding[:v_in.shape[0], :] = v_in
            v_in = padding

            # Mask
            mask = torch.tensor([1. if x < num_atoms else 0. for x in range(self.max_atoms)])

            # Edge features
            k = 0
            e_in = torch.zeros([num_atoms, num_atoms, data.edge_attr.shape[1]])
            for i, j in itertools.combinations(np.arange(num_atoms), 2):
                e_in[i, j, :] = data.edge_attr[k, :]
                e_in[j, i, :] = data.edge_attr[k, :]
                k += 1
            padding = torch.zeros([self.max_atoms, self.max_atoms, data.edge_attr.shape[1]])
            padding[:e_in.shape[0], :e_in.shape[0], :] = e_in
            e_in = padding

            # Target
            target = data.y
            padding = torch.zeros([self.max_atoms*self.max_atoms - self.max_atoms, data.y.shape[1]])
            padding[:target.shape[0], :] = target
            target = padding

            return v_in, e_in, mask, target

        else:
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
        # noinspection PyTypeChecker
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
        # noinspection PyTypeChecker
        condition = condition.type(torch.float32)

        # Pad the condition matrix
        padding = torch.zeros([self.padding_dim, self.condition_dim])
        padding[0:condition.shape[0], :] = condition
        condition = padding

        return dist_vec, condition, num_dist

    def __repr__(self) -> str:
        return '{}({})'.format(self.__class__.__name__, len(self))
