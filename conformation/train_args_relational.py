""" Train arguments class for relational network training. """
from typing import List

from rdkit import Chem
from rdkit.Chem import rdchem
# noinspection PyPackageRequirements
from tap import Tap


class Args(Tap):
    """
    System arguments.
    """
    data_path: str  # Path to metadata file
    checkpoint_path: str = None  # Directory of checkpoint to load saved model
    save_dir: str  # Save directory
    num_epochs: int  # Number of training epochs
    atomic_num: bool = True  # Whether or not to include atomic number as vertex feature
    atom_types: List[int] = [1, 6, 7, 8]  # Allowed atom types
    num_vertex_features: int = None  # Number of vertex features (set automatically)
    bond_types: List = [0., 1., 1.5, 2., 3.]  # Allowed bond types
    num_edge_features: int = None  # Number of edge features (set automatically)
    max_shortest_path_length: int = 10  # Maximum shortest path length between pairs of atoms
    partial_charge: bool = True  # Whether or not to include partial charge as vertex feature
    mmff_atom_types_one_hot: bool = True  # Whether or not to include MMFF atom type as vertex feature
    valence_types: List[int] = [1, 2, 3, 4, 5, 6]  # Valence types
    valence: bool = True  # Whether or not to include total valence as vertex feature
    aromatic: bool = True  # Whether or not to include aromaticity as vertex feature
    hybridization: bool = True  # Whether or not to include hybridization as vertex feature
    assign_stereo: bool = True  # Whether or not to assign stereochemistry
    charge_types: List[int] = [-1, 0, 1]  # Charge types
    formal_charge: bool = True  # Whether or not to include formal charge as vertex feature
    r_covalent: bool = True  # Whether or not to include covalent radius as vertex feature
    r_vanderwals: bool = True  # Whether or not to include vanderwals radius as vertex feature
    default_valence: bool = True  # Whether or not to include default valence as vertex feature
    max_ring_size: int = 8  # Maximum ring size
    rings: bool = True  # Whether or not to include rings as vertex feature
    # noinspection PyUnresolvedReferences
    chirality: bool = True  # Whether or not to include chirality as vertex feature
    std: bool = False  # Whether or not to additionally train on atomic pairwise distance standard deviation
    final_output_size: int = 1  # Size of output layer
    alpha: float = 20.0  # How much to weight positive std prediction losses
    num_layers: int = 10  # Number of layers
    batch_size: int = 10  # Batch size
    lr: float = 1e-4  # Learning rate
    hidden_size: int = 256  # Hidden size
    final_linear_size: int = 1024  # Size of last linear layer
    cuda: bool = False  # Cuda availability
    log_frequency: int = 10  # Log frequency

    def __init__(self):
        super(Args, self).__init__()
        # noinspection PyUnresolvedReferences
        self._hybridization_types = [Chem.HybridizationType.S,
                                     Chem.HybridizationType.SP,
                                     Chem.HybridizationType.SP2,
                                     Chem.HybridizationType.SP3,
                                     Chem.HybridizationType.SP3D,
                                     Chem.HybridizationType.SP3D2,
                                     Chem.HybridizationType.UNSPECIFIED]
        # noinspection PyUnresolvedReferences
        self._chi_types = list(rdchem.ChiralType.values.values())
        self._mmff94_atom_types = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 16, 17, 18, 20, 21, 22, 23, 24, 25, 26,
                                   27, 28, 29, 30, 31, 32, 33, 37, 38, 39, 40, 42, 43, 44, 46, 48, 59, 62, 63, 64,
                                   65, 66, 70, 71, 72, 74, 75, 78]

    # noinspection PyUnresolvedReferences
    @property
    def hybridization_types(self) -> List[Chem.HybridizationType]:
        """
        Hybridization types property.
        :return: List of types.
        """
        return self._hybridization_types

    # noinspection PyUnresolvedReferences
    @hybridization_types.setter
    def hybridization_types(self, hybridization_types: List[Chem.HybridizationType]) -> None:
        """
        Setter for hybridization types.
        :param hybridization_types: List of hybridization types.
        :return: None.
        """
        self._hybridization_types = hybridization_types

    # noinspection PyUnresolvedReferences
    @property
    def chi_types(self) -> List[rdchem.ChiralType]:
        """
        Chiral tag types property.
        :return: List of types.
        """
        return self._chi_types

    # noinspection PyUnresolvedReferences
    @chi_types.setter
    def chi_types(self, chi_types: List[rdchem.ChiralType]) -> None:
        """
        Setter for chiral types.
        :param chi_types: List of chiral tag types.
        :return: None.
        """
        self._chi_types = chi_types

    @property
    def mmff94_atom_types(self) -> List[int]:
        """
        MMFF94 atom types property.
        :return: List of types.
        """
        return self._mmff94_atom_types

    @mmff94_atom_types.setter
    def mmff94_atom_types(self, mmff94_atom_types: List[int]) -> None:
        """
        Setter for chiral types.
        :param mmff94_atom_types: List of MMFF94 types.
        :return: None.
        """
        self._mmff94_atom_types = mmff94_atom_types