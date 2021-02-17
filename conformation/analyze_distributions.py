""" Plotting of conformation distributions. """
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from sklearn.cluster import MeanShift

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
    svd_tol: float = 1e-5  # Tolerance below which a singular value is considered 0.
    corr_heatmap_font_scale: float = 0.4  # Font scale for pairwise torsion correlations heatmap
    corr_heatmap_annot_size: float = 6.0  # Font size for annotations in pairwise torsion correlations heatmap
    corr_heatmap_dpi: int = 200  # DPI for pairwise torsion correlations heatmap
    joint_hist_bw_adjust: float = 0.25  # KDE bw_adjust value for pairwise joint histogram of torsions plot
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

    print("Loading molecule")
    # noinspection PyUnresolvedReferences
    mol = Chem.Mol(open(args.data_path, "rb").read())

    print("Computing energies")
    res = AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=0, numThreads=0)
    energies = []
    for i in range(len(res)):
        energies.append(res[i][1])

    print("Plotting energy histograms")
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

    print("Computing rotatable bond angles")
    # Compute rotatable bond angles
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
        plt.plot(weights)
        plt.ylim((-0.1, 1))
        plt.savefig(os.path.join(args.save_dir, "weights.png"))
        plt.close()
    else:
        weights = None

    df = pd.DataFrame(np.array(angles).transpose())
    column_names = dict()
    for i, bond in enumerate(rotatable_bonds):
        atom_0 = mol.GetAtomWithIdx(bond[0]).GetSymbol()
        atom_1 = mol.GetAtomWithIdx(bond[1]).GetSymbol()
        column_names[i] = f'{bond[0]}-{bond[1]} | {atom_0}-{atom_1}'
    df = df.rename(columns=column_names)

    print("Computing angles of bonds in aromatic rings")
    atom_indices = []
    column_names_aromatic = dict()
    aromatic_count = 0
    for bond in mol.GetBonds():
        if bond.GetBeginAtom().GetIsAromatic() and bond.GetEndAtom().GetIsAromatic():
            # Get atom indices for the ith bond
            atom_a_idx = bond.GetBeginAtom().GetIdx()
            atom_b_idx = bond.GetEndAtom().GetIdx()
            atom_a_symbol = bond.GetBeginAtom().GetSymbol()
            atom_b_symbol = bond.GetEndAtom().GetSymbol()

            # Select a neighbor for each atom in order to form a dihedral
            atom_a_neighbors = mol.GetAtomWithIdx(atom_a_idx).GetNeighbors()
            atom_a_neighbor_index = [x.GetIdx() for x in atom_a_neighbors if x.GetIdx() != atom_b_idx][0]
            atom_b_neighbors = mol.GetAtomWithIdx(atom_b_idx).GetNeighbors()
            atom_b_neighbor_index = [x.GetIdx() for x in atom_b_neighbors if x.GetIdx() != atom_a_idx][0]

            atom_indices.append([atom_a_neighbor_index, atom_a_idx, atom_b_idx, atom_b_neighbor_index])

            column_names_aromatic[aromatic_count] = f'{atom_a_idx}-{atom_b_idx} | {atom_a_symbol}-{atom_b_symbol}'

            aromatic_count += 1

    angles = [[] for _ in range(aromatic_count)]
    for i in tqdm(range(mol.GetNumConformers())):
        c = mol.GetConformer(i)
        for j in range(aromatic_count):
            angles[j].append(rdMolTransforms.GetDihedralRad(c, atom_indices[j][0], atom_indices[j][1],
                                                            atom_indices[j][2], atom_indices[j][3]))
    df_ring = pd.DataFrame(np.array(angles).transpose())
    df_ring = df_ring.rename(columns=column_names_aromatic)

    print("Computing angles of non-rotatable, non-aromatic bonds")
    atom_indices = []
    column_names_other = dict()
    other_count = 0
    for bond in mol.GetBonds():
        if not bond.GetBeginAtom().GetIsAromatic() or not bond.GetEndAtom().GetIsAromatic():
            if (bond.GetBeginAtom().GetIdx(), bond.GetEndAtom().GetIdx()) not in rotatable_bonds:
                if bond.IsInRing():
                    # Get atom indices for the ith bond
                    atom_a_idx = bond.GetBeginAtom().GetIdx()
                    atom_b_idx = bond.GetEndAtom().GetIdx()
                    atom_a_symbol = bond.GetBeginAtom().GetSymbol()
                    atom_b_symbol = bond.GetEndAtom().GetSymbol()

                    if len(mol.GetAtomWithIdx(atom_a_idx).GetNeighbors()) > 3 and len(mol.GetAtomWithIdx(atom_b_idx).GetNeighbors()) > 3:

                        # Select a neighbor for each atom in order to form a dihedral
                        atom_a_neighbors = mol.GetAtomWithIdx(atom_a_idx).GetNeighbors()
                        atom_a_neighbor_index = [x.GetIdx() for x in atom_a_neighbors if x.GetIdx() != atom_b_idx][0]
                        atom_b_neighbors = mol.GetAtomWithIdx(atom_b_idx).GetNeighbors()
                        atom_b_neighbor_index = [x.GetIdx() for x in atom_b_neighbors if x.GetIdx() != atom_a_idx][0]

                        atom_indices.append([atom_a_neighbor_index, atom_a_idx, atom_b_idx, atom_b_neighbor_index])

                        column_names_other[other_count] = f'{atom_a_idx}-{atom_b_idx} | {atom_a_symbol}-{atom_b_symbol}'

                        other_count += 1

    print("Other ring count", other_count)
    angles = [[] for _ in range(other_count)]
    for i in tqdm(range(mol.GetNumConformers())):
        c = mol.GetConformer(i)
        for j in range(other_count):
            angles[j].append(rdMolTransforms.GetDihedralRad(c, atom_indices[j][0], atom_indices[j][1],
                                                            atom_indices[j][2], atom_indices[j][3]))
    df_other = pd.DataFrame(np.array(angles).transpose())
    df_other = df_other.rename(columns=column_names_other)

    print("Plotting pairwise joint histograms of aromatic ring bond angles")
    g = sns.PairGrid(df_ring)
    g.set(ylim=(-math.pi - 1., math.pi + 1.), xlim=(-math.pi - 1., math.pi + 1.))
    g.map_upper(sns.histplot, bins=list(np.arange(-math.pi - 1., math.pi + 1., 0.1)), weights=weights)
    g.map_lower(sns.kdeplot, fill=True, weights=weights, bw_adjust=args.joint_hist_bw_adjust)
    g.map_diag(sns.histplot, bins=list(np.arange(-math.pi - 1., math.pi + 1., 0.1)), weights=weights)
    g.savefig(os.path.join(args.save_dir, "aromatic_pairwise_joint_histograms_seaborn.png"))
    plt.close()

    print("Plotting pairwise joint histograms of other bond angles")
    g = sns.PairGrid(df_other)
    g.set(ylim=(-math.pi - 1., math.pi + 1.), xlim=(-math.pi - 1., math.pi + 1.))
    g.map_upper(sns.histplot, bins=list(np.arange(-math.pi - 1., math.pi + 1., 0.1)), weights=weights)
    g.map_lower(sns.kdeplot, fill=True, weights=weights, bw_adjust=args.joint_hist_bw_adjust)
    g.map_diag(sns.histplot, bins=list(np.arange(-math.pi - 1., math.pi + 1., 0.1)), weights=weights)
    g.savefig(os.path.join(args.save_dir, "other_pairwise_joint_histograms_seaborn.png"))
    plt.close()

    print(f'Rank of rotatable bond angle matrix: {np.linalg.matrix_rank(df.to_numpy(), tol=args.svd_tol)}')

    print("Approximating number of modes in each torsion angle distribution")
    for i in range(df.shape[1]):
        clustering = MeanShift(bandwidth=1.0).fit(df.iloc[:, i].to_numpy().reshape(-1, 1))
        print(f'Rotatable bond {column_names[i]}: {clustering.cluster_centers_.shape[0]}')

    print("Plotting pairwise joint histograms of rotatable bond angles")
    g = sns.PairGrid(df)
    g.set(ylim=(-math.pi - 1., math.pi + 1.), xlim=(-math.pi - 1., math.pi + 1.))
    g.map_upper(sns.histplot, bins=list(np.arange(-math.pi - 1., math.pi + 1., 0.1)), weights=weights)
    g.map_lower(sns.kdeplot, fill=True, weights=weights, bw_adjust=args.joint_hist_bw_adjust)
    g.map_diag(sns.histplot, bins=list(np.arange(-math.pi - 1., math.pi + 1., 0.1)), weights=weights)
    g.savefig(os.path.join(args.save_dir, "pairwise_joint_histograms_seaborn.png"))
    plt.close()

    print("Plotting heatmap of pairwise correlation coefficients")
    sns.set(font_scale=args.corr_heatmap_font_scale)
    # noinspection PyUnresolvedReferences
    sns.heatmap(df.corr(), cmap=plt.cm.Blues, annot=True, annot_kws={'size': args.corr_heatmap_annot_size},
                vmin=-1, vmax=1)
    plt.yticks(va='center')
    plt.savefig(os.path.join(args.save_dir, "pairwise_joint_correlations_seaborn.png"), dpi=args.corr_heatmap_dpi)
    plt.close()
    sns.set_theme()

    print("Plotting heatmap of aromatic pairwise correlation coefficients")
    sns.set(font_scale=args.corr_heatmap_font_scale)
    # noinspection PyUnresolvedReferences
    sns.heatmap(df_ring.corr(), cmap=plt.cm.Blues, annot=True, annot_kws={'size': args.corr_heatmap_annot_size},
                vmin=-1, vmax=1)
    plt.yticks(va='center')
    plt.savefig(os.path.join(args.save_dir, "aromatic_pairwise_joint_correlations_seaborn.png"),
                dpi=args.corr_heatmap_dpi)
    plt.close()
    sns.set_theme()

    print("Plotting heatmap of other pairwise correlation coefficients")
    sns.set(font_scale=args.corr_heatmap_font_scale)
    # noinspection PyUnresolvedReferences
    sns.heatmap(df_other.corr(), cmap=plt.cm.Blues, annot=True, annot_kws={'size': args.corr_heatmap_annot_size},
                vmin=-1, vmax=1)
    plt.yticks(va='center')
    plt.savefig(os.path.join(args.save_dir, "other_pairwise_joint_correlations_seaborn.png"),
                dpi=args.corr_heatmap_dpi)
    plt.close()
    sns.set_theme()

    print("Computing minimized energies")
    res = AllChem.MMFFOptimizeMoleculeConfs(mol, numThreads=0)
    energies = []
    for i in range(len(res)):
        energies.append(res[i][1])

    print("Plotting minimized energy histograms")
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

    print("Drawing molecule with atom id labels")
    d = rdMolDraw2D.MolDraw2DCairo(500, 500)
    # noinspection PyArgumentList
    d.drawOptions().addStereoAnnotation = True
    # noinspection PyArgumentList
    d.drawOptions().addAtomIndices = True
    rdMolDraw2D.PrepareAndDrawMolecule(d, mol)
    with open(os.path.join(args.save_dir, 'molecule.png'), 'wb') as f:
        # noinspection PyArgumentList
        f.write(d.GetDrawingText())
