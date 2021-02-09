""" Plotting of conformation distributions. """
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from rdkit import Chem
from rdkit.Chem import AllChem, rdMolTransforms
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.Lipinski import RotatableBondSmarts
import seaborn as sns
# noinspection PyPackageRequirements
from tap import Tap
from tqdm import tqdm

from conformation.compare_pairwise_distance_histograms import compute_energy_weights


class Args(Tap):
    """
    System arguments.
    """
    data_path: str  # Path to RDKit binary file containing conformations
    num_energy_decimals: int = 3  # Number of energy decimals used for computing empirical minimized energy probability
    weights: bool = False  # Whether or not to weight histograms by empirical Boltzmann probability
    temp: float = 300.0  # Temperature for Boltzmann weighting (weights = True)
    save_dir: str  # Path to directory containing output files


def analyze_distributions(args: Args) -> None:
    """
    Plotting of conformation distributions.
    :return: None.
    """
    os.makedirs(args.save_dir, exist_ok=True)

    # Define constants
    k_b = 3.297e-24  # Boltzmann constant in cal/K
    temp = args.temp  # Temperature in K
    avogadro = 6.022e23

    print("Loading molecule...")
    # noinspection PyUnresolvedReferences
    mol = Chem.Mol(open(args.data_path, "rb").read())

    print("Computing energies...")
    res = AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=0, numThreads=0)
    energies = []
    for i in range(len(res)):
        energies.append(res[i][1])

    # Plot energy histogram
    # NOTE: defining the bins is useful because sometimes automatic bin placement takes forever
    sns.set_theme()
    fig, ax = plt.subplots()
    sns.histplot(energies, ax=ax, bins=np.arange(min(energies) - 1., max(energies) + 1., 0.1))
    ax.set_xlabel("Energy (kcal/mol)")
    ax.figure.savefig(os.path.join(args.save_dir, "energy-histogram.png"))
    plt.clf()
    plt.close()

    # Plot energy distribution
    fig, ax = plt.subplots()
    sns.histplot(energies, ax=ax, stat='density', kde=True, bins=np.arange(min(energies) - 1., max(energies) + 1., 0.1))
    ax.set_xlabel("Energy (kcal/mol)")
    ax.figure.savefig(os.path.join(args.save_dir, "energy-distribution.png"))
    plt.clf()
    plt.close()

    print("Computing marginal distributions of rotatable bond angles...")
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

    if args.weights:
        weights = compute_energy_weights(mol, k_b, temp, avogadro)
    else:
        weights = None

    print("Computing pairwise joint torsion histograms via Seaborn")
    df = pd.DataFrame(np.array(angles).transpose())
    column_names = dict()
    for i, bond in enumerate(rotatable_bonds):
        atom_0 = mol.GetAtomWithIdx(bond[0]).GetSymbol()
        atom_1 = mol.GetAtomWithIdx(bond[1]).GetSymbol()
        column_names[i] = f'{bond[0]}-{bond[1]} | {atom_0}-{atom_1}'
    df = df.rename(columns=column_names)
    g = sns.PairGrid(df)
    g.set(ylim=(-math.pi - 1., math.pi + 1.), xlim=(-math.pi - 1., math.pi + 1.))
    g.map_upper(sns.histplot, bins=list(np.arange(-math.pi - 1., math.pi + 1., 0.1)), weights=weights)
    g.map_lower(sns.kdeplot, fill=True, weights=weights, bw_adjust=0.25)
    g.map_diag(sns.histplot, bins=list(np.arange(-math.pi - 1., math.pi + 1., 0.1)), weights=weights)
    g.savefig(os.path.join(args.save_dir, "pairwise_joint_histograms_seaborn.png"))
    plt.close()

    print("Computing minimized energies...")
    res = AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=200, numThreads=0)
    energies = []
    for i in range(len(res)):
        energies.append(res[i][1])

    # Plot energy histogram
    # NOTE: defining the bins is useful because sometimes automatic bin placement takes forever
    fig, ax = plt.subplots()
    sns.histplot(energies, ax=ax, bins=np.arange(min(energies) - 1., max(energies) + 1., 0.1))
    ax.set_xlabel("Energy (kcal/mol)")
    ax.figure.savefig(os.path.join(args.save_dir, "minimized-energy-histogram.png"))
    plt.clf()
    plt.close()

    # Plot energy distribution
    fig, ax = plt.subplots()
    sns.histplot(energies, ax=ax, stat='density', kde=True, bins=np.arange(min(energies) - 1., max(energies) + 1., 0.1))
    ax.set_xlabel("Energy (kcal/mol)")
    ax.figure.savefig(os.path.join(args.save_dir, "minimized-energy-distribution.png"))
    plt.clf()
    plt.close()

    # Compute empirical energy probabilities
    energy_counts = dict()
    rounded_energies = []
    for en in energies:
        rounded_energies.append(round(en, args.num_energy_decimals))

    for en in rounded_energies:
        if en in energy_counts:
            energy_counts[en] += 1
        else:
            energy_counts[en] = 1

    probabilities = []
    energies = []
    num_energies = len(rounded_energies)
    for item in energy_counts:
        energies.append(item)
        probabilities.append(energy_counts[item] / num_energies)

    # Plot probabilities
    ax = sns.scatterplot(x=energies, y=probabilities, color="b")
    ax.set_xlabel("Energy (kcal/mol)")
    ax.set_ylabel("Energy Probability")
    ax.figure.savefig(os.path.join(args.save_dir, "probabilities-vs-energies.png"))
    plt.clf()
    plt.close()

    # Draw molecule with atom id labels
    d = rdMolDraw2D.MolDraw2DCairo(500, 500)
    # noinspection PyArgumentList
    d.drawOptions().addStereoAnnotation = True
    # noinspection PyArgumentList
    d.drawOptions().addAtomIndices = True
    rdMolDraw2D.PrepareAndDrawMolecule(d, mol)
    with open(os.path.join(args.save_dir, 'molecule.png'), 'wb') as f:
        # noinspection PyArgumentList
        f.write(d.GetDrawingText())
