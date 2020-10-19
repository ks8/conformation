""" Plotting of conformation distributions. """
import matplotlib.pyplot as plt
import numpy as np
import os

from rdkit import Chem
from rdkit.Chem import AllChem, rdMolTransforms
from rdkit.Chem.Lipinski import RotatableBondSmarts
import seaborn as sns
# noinspection PyPackageRequirements
from tap import Tap
from tqdm import tqdm


class Args(Tap):
    """
    System arguments.
    """
    data_path: str  # Path to RDKit binary file containing conformations
    save_dir: str  # Path to directory containing output files


def analyze_distributions(args: Args) -> None:
    """
    Plotting of conformation distributions.
    :return: None.
    """
    os.makedirs(args.save_dir)

    # Load molecule
    # noinspection PyUnresolvedReferences
    mol = Chem.Mol(open(args.data_path, "rb").read())

    # Compute total energies
    res = AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=0, numThreads=0)
    energies = []
    for i in range(len(res)):
        energies.append(res[i][1])

    # Plot total energies
    fig, ax = plt.subplots()
    sns.histplot(energies, ax=ax, bins=np.arange(min(energies) - 1., max(energies) + 1., 0.1))
    ax.set_xlabel("Energy (kcal/mol)")
    ax.set_ylabel("Count")
    ax.figure.savefig(os.path.join(args.save_dir, "energy-distribution.png"))
    plt.clf()
    plt.close()

    # Compute marginal distributions of rotatable bond torsional angles
    rotatable_bonds = mol.GetSubstructMatches(RotatableBondSmarts)
    atom_indices = []
    for i, bond in enumerate(rotatable_bonds):
        # Get atom indices for the ith bond
        atom_a_idx = bond[0]
        atom_b_idx = bond[1]

        # Select a neighbor for each atom in order to form a dihedral
        atom_a_neighbors = mol.GetAtomWithIdx(atom_a_idx).GetNeighbors()
        atom_a_neighbor_index = [x.GetIdx() for x in atom_a_neighbors if x.GetIdx() != atom_b_idx][0]
        atom_b_neighbors = mol.GetAtomWithIdx(atom_b_idx).GetNeighbors()
        atom_b_neighbor_index = [x.GetIdx() for x in atom_b_neighbors if x.GetIdx() != atom_a_idx][0]

        atom_indices.append([atom_a_neighbor_index, atom_a_idx, atom_b_idx, atom_b_neighbor_index])

    angles = [[] for _ in range(len(rotatable_bonds))]
    for i in tqdm(range(mol.GetNumConformers())):
        c = mol.GetConformer(i)
        for j in range(len(rotatable_bonds)):
            angles[j].append(rdMolTransforms.GetDihedralRad(c, atom_indices[j][0], atom_indices[j][1],
                                                            atom_indices[j][2], atom_indices[j][3]))

    for i, bond in enumerate(rotatable_bonds):
        fig, ax = plt.subplots()
        sns.histplot(angles[i], ax=ax, bins=np.arange(min(angles[i]) - 1., max(angles[i]) + 1., 0.1))
        ax.set_xlabel("Angle (radians)")
        ax.set_ylabel("Count")
        ax.figure.savefig(os.path.join(args.save_dir, f'rotatable-bond-{bond[0]}-{bond[1]}-distribution.png'))
        plt.clf()
        plt.close()
