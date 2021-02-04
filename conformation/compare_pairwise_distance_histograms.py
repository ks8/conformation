""" Plot matrix of pairwise distance histograms for two sets of conformations. """
import itertools
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
from typing import Dict, List

from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.rdDistGeom import GetMoleculeBoundsMatrix
from rdkit.Chem import AllChem, rdchem, rdmolops, rdMolTransforms
from rdkit.Chem.Lipinski import RotatableBondSmarts
# noinspection PyPackageRequirements
from tap import Tap
from tqdm import tqdm

from conformation.distance_matrix import dist_matrix


class Args(Tap):
    """
    System arguments.
    """
    conf_path_1: str  # Path to binary mol file containing conformations
    conf_path_2: str  # Path to another binary mol file containing conformations
    conf_labels: List[str] = []  # List of method labels for plotting purposes
    num_bins_matrix: int = 50  # Number of histogram bins for plotting matrix of distances
    y_lim_matrix: float = None  # y limit on matrix subplots (z-score = True)
    num_bins_torsions: int = 50  # Number of histogram bins for plotting torsions
    y_lim_torsions: float = 2.0  # y limit on torsion subplots
    line_width: float = 0.5  # Line width for plotting
    weights_1: bool = False  # Whether or not to weight first conf set histogram by empirical Boltzmann probability
    weights_2: bool = False  # Whether or not to weight second conf set histogram by empirical Boltzmann probability
    temp: float = 300.0  # Temperature for computing Boltzmann probabilities
    bounds: bool = False  # Whether or not to include RDKit bounds in plots
    figsize: int = 10  # Height and width for figsize
    dpi: int = 200  # dpi for figure
    z_score: bool = False  # Whether or not to Z-score matrix of distances
    type_emphasis: bool = False  # Whether or not to box certain types for emphasis
    save_dir: str  # Path to directory containing output files


# noinspection PyUnresolvedReferences
def construct_distance_dict(mol: rdchem.Mol) -> Dict:
    """
    Construct a dictionary containing list of distances for each pair of atoms in a molecule.
    :param mol: RDKit Mol object.
    :return: Dictionary of pairwise distances.
    """
    dist = dict()
    for i in tqdm(range(mol.GetNumConformers())):
        pos = mol.GetConformer(i).GetPositions()
        distmat = dist_matrix(pos)
        for j, k in itertools.combinations(np.arange(mol.GetNumAtoms()), 2):
            if (j, k) not in dist:
                dist[(j, k)] = [distmat[j][k]]
            else:
                dist[(j, k)].append(distmat[j][k])

    return dist


# noinspection PyUnresolvedReferences
def compute_torsions(mol: rdchem.Mol) -> List[List[float]]:
    """
    Compute torsion angle values for all bonds.
    :param mol: RDKit Mol object.
    :return: List
    """
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

    return angles


# noinspection PyUnresolvedReferences
def compute_energy_weights(mol: rdchem.Mol, k_b: float, temp: float, avogadro: float) -> np.ndarray:
    """
    Compute empirical Boltzmann probabilities of conformations.
    :param mol: RDKit Mol object.
    :param k_b: Boltzmann's constant in cal/K.
    :param temp: Temperature in K.
    :param avogadro: Avogadro's constant.
    :return: Array of empirical probabilities.
    """
    probabilities = []
    res = AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=0, numThreads=0)
    for i in range(len(res)):
        energy = res[i][1] * (1000.0 * 4.184 / avogadro)  # Convert energy to Joules
        probabilities.append(math.exp(-energy / (k_b * temp * 4.184)))
    Z = sum(probabilities)  # Compute empirical partition function

    # Compute empirical probabilities
    for i in range(len(probabilities)):
        probabilities[i] /= Z
    probabilities = np.array(probabilities)

    return probabilities


def compare_pairwise_distance_histograms(args: Args) -> None:
    """
    Plot matrix of pairwise distance histograms for two sets of conformations.
    :param args: System args.
    """
    os.makedirs(args.save_dir)

    # Define constants
    k_b = 3.297e-24  # Boltzmann constant in cal/K
    temp = args.temp  # Temperature in K
    avogadro = 6.022e23

    # Load the two sets of conformations
    # noinspection PyUnresolvedReferences
    mol_1 = Chem.Mol(open(args.conf_path_1, "rb").read())
    dist_1 = construct_distance_dict(mol_1)
    angles_1 = compute_torsions(mol_1)
    num_atoms = mol_1.GetNumAtoms()

    # noinspection PyUnresolvedReferences
    mol_2 = Chem.Mol(open(args.conf_path_2, "rb").read())
    dist_2 = construct_distance_dict(mol_2)
    angles_2 = compute_torsions(mol_2)

    # Compute energy weights for mol_1
    if args.weights_1:
        weights_1 = compute_energy_weights(mol_1, k_b, temp, avogadro)
    else:
        weights_1 = None

    if args.weights_2:
        weights_2 = compute_energy_weights(mol_2, k_b, temp, avogadro)
    else:
        weights_2 = None

    if args.z_score:
        complete_list = []
        for item in dist_1:
            complete_list += dist_1[item]
        mu = np.mean(complete_list)
        std = np.std(complete_list)
        complete_list -= mu
        complete_list /= std
        min_val = min(complete_list)
        max_val = max(complete_list)

        for dictionary in [dist_1, dist_2]:
            for item in dictionary:
                dictionary[item] -= mu
                dictionary[item] /= std

    bounds = GetMoleculeBoundsMatrix(mol_1)

    # Plot the matrix with emphasis on 1-2 pairs, 1-3 pairs, 1-4, pairs and the rest.
    if not args.type_emphasis:
        # Create figure and grid spec
        fig = plt.figure(constrained_layout=False, dpi=args.dpi, figsize=[args.figsize, args.figsize])
        gs = fig.add_gridspec(num_atoms, num_atoms)

        # Add distance plots to the grid (it will be symmetric)
        for i in range(num_atoms):
            for j in range(num_atoms):
                if i != j:
                    ax = fig.add_subplot(gs[i, j])
                    data_1 = dist_1[(min(i, j), max(i, j))]
                    data_2 = dist_2[(min(i, j), max(i, j))]
                    ax.hist(data_1, density=True, histtype='step',
                            bins=args.num_bins_matrix, linewidth=args.line_width, weights=weights_1)
                    ax.hist(data_2, density=True, histtype='step',
                            bins=args.num_bins_matrix, linewidth=args.line_width, weights=weights_2)
                    if args.y_lim_matrix is not None:
                        ax.set_ylim((0., args.y_lim_matrix))
                    if args.z_score:
                        # noinspection PyUnboundLocalVariable
                        ax.set_xlim(min_val, max_val)
                    if args.bounds:
                        plt.axvline(bounds[i][j], color='r', linewidth=args.line_width)
                        plt.axvline(bounds[j][i], color='r', linewidth=args.line_width)
                    ax.spines['top'].set_visible(False)
                    ax.spines['bottom'].set_visible(False)
                    ax.spines['left'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.set(xticks=[], yticks=[])

        ax = fig.add_subplot(gs[0, 0])
        ax.plot([], label="PT")
        ax.plot([], label="ETKDG")
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set(xticks=[], yticks=[])
        ax.legend()

        plt.savefig(os.path.join(args.save_dir, f'pairwise_distance_matrix_pairs.png'))
        plt.close()

    else:
        for m in range(1, 5):
            # Create figure and grid spec
            fig = plt.figure(constrained_layout=False, dpi=args.dpi, figsize=[args.figsize, args.figsize])
            gs = fig.add_gridspec(num_atoms, num_atoms)

            # Add distance plots to the grid (it will be symmetric)
            for i in range(num_atoms):
                for j in range(num_atoms):
                    if i != j:
                        ax = fig.add_subplot(gs[i, j])
                        data_1 = dist_1[(min(i, j), max(i, j))]
                        data_2 = dist_2[(min(i, j), max(i, j))]
                        ax.hist(data_1, density=True, histtype='step',
                                bins=args.num_bins_matrix, linewidth=args.line_width, weights=weights_1)
                        ax.hist(data_2, density=True, histtype='step',
                                bins=args.num_bins_matrix, linewidth=args.line_width, weights=weights_2)
                        if args.y_lim_matrix is not None:
                            ax.set_ylim((0., args.y_lim_matrix))
                        if args.z_score:
                            # noinspection PyUnboundLocalVariable
                            ax.set_xlim(min_val, max_val)
                        if args.bounds:
                            plt.axvline(bounds[i][j], color='r', linewidth=args.line_width)
                            plt.axvline(bounds[j][i], color='r', linewidth=args.line_width)

                        if m in [1, 2, 3]:
                            if len(rdmolops.GetShortestPath(mol_1, int(i), int(j))) - 1 != m:
                                ax.spines['top'].set_visible(False)
                                ax.spines['bottom'].set_visible(False)
                                ax.spines['left'].set_visible(False)
                                ax.spines['right'].set_visible(False)
                            ax.set(xticks=[], yticks=[])
                        else:
                            if len(rdmolops.GetShortestPath(mol_1, int(i), int(j))) - 1 < m:
                                ax.spines['top'].set_visible(False)
                                ax.spines['bottom'].set_visible(False)
                                ax.spines['left'].set_visible(False)
                                ax.spines['right'].set_visible(False)
                            ax.set(xticks=[], yticks=[])

            ax = fig.add_subplot(gs[0, 0])
            ax.plot([], label="PT")
            ax.plot([], label="ETKDG")
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set(xticks=[], yticks=[])
            ax.legend()

            if m == 1:
                label = "12"
            elif m == 2:
                label = "13"
            elif m == 3:
                label = "14"
            else:
                label = "other"
            plt.savefig(os.path.join(args.save_dir, f'pairwise_distance_matrix_{label}_pairs.png'))
            plt.close()

    d = rdMolDraw2D.MolDraw2DCairo(500, 500)
    # noinspection PyArgumentList
    d.drawOptions().addStereoAnnotation = True
    # noinspection PyArgumentList
    d.drawOptions().addAtomIndices = True
    rdMolDraw2D.PrepareAndDrawMolecule(d, mol_1)
    with open(os.path.join(args.save_dir, 'molecule.png'), 'wb') as f:
        # noinspection PyArgumentList
        f.write(d.GetDrawingText())

    rotatable_bonds = mol_1.GetSubstructMatches(RotatableBondSmarts)
    for i, bond in enumerate(rotatable_bonds):
        fig, ax = plt.subplots()
        if args.weights_1:
            # noinspection PyUnboundLocalVariable
            df = np.concatenate((np.array(angles_1[i])[:, np.newaxis], weights_1[:, np.newaxis]), axis=1)
            df = pd.DataFrame(df)
            df = df.rename(columns={0: 'Angle', 1: 'Probability'})
            sns.histplot(df, x='Angle', ax=ax, bins=len(np.arange(-math.pi - 1., math.pi + 1., 0.1)),
                         weights='Probability', stat='density', color='b', label=args.conf_labels[0])
        else:
            sns.histplot(angles_1[i], ax=ax, bins=np.arange(-math.pi - 1., math.pi + 1., 0.1), stat='density',
                         color='b', label=args.conf_labels[0])
        if args.weights_2:
            # noinspection PyUnboundLocalVariable
            df = np.concatenate((np.array(angles_2[i])[:, np.newaxis], weights_2[:, np.newaxis]), axis=1)
            df = pd.DataFrame(df)
            df = df.rename(columns={0: 'Angle', 1: 'Probability'})
            sns.histplot(df, x='Angle', ax=ax, bins=len(np.arange(-math.pi - 1., math.pi + 1., 0.1)),
                         weights='Probability', stat='density', color='r', label=args.conf_labels[1])
        else:
            sns.histplot(angles_2[i], ax=ax, bins=np.arange(-math.pi - 1., math.pi + 1., 0.1), stat='density',
                         color='r', label=args.conf_labels[1])
        ax.set_xlabel("Angle (radians)")
        ax.set_ylabel("Density")
        atom_0 = mol_1.GetAtomWithIdx(bond[0]).GetSymbol()
        atom_1 = mol_1.GetAtomWithIdx(bond[1]).GetSymbol()
        plt.legend()
        plt.ylim((0., args.y_lim_torsions))
        ax.figure.savefig(os.path.join(args.save_dir,
                                       f'rotatable-bond-{bond[0]}-{bond[1]}-{atom_0}-{atom_1}-distribution.png'))
        plt.clf()
        plt.close()
