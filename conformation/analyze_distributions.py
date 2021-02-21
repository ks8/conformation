""" Plotting of conformation distributions. """
import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from scipy.stats import entropy, gaussian_kde

from rdkit import Chem
from rdkit.Chem import AllChem, rdMolTransforms, rdchem
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.Lipinski import RotatableBondSmarts
import seaborn as sns
# noinspection PyPackageRequirements
from tap import Tap

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
    hist_bin_width: float = 0.1  # Bin width for histograms
    corr_heatmap_font_scale: float = 0.4  # Font scale for pairwise torsion correlations heatmap
    mode_count_font_scale: float = 0.6  # Font scale for pairwise torsion correlations heatmap
    mode_count_dpi: int = 200
    corr_heatmap_annot_size: float = 6.0  # Font size for annotations in pairwise torsion correlations heatmap
    corr_heatmap_dpi: int = 200  # DPI for pairwise torsion correlations heatmap
    joint_hist_bw_adjust: float = 0.25  # KDE bw_adjust value for pairwise joint histogram of torsions plot
    entropy_bins: int = 10
    save_dir: str  # Path to directory containing output files


# noinspection PyUnresolvedReferences
def compute_energy(mol: rdchem.Mol, minimize: bool = False) -> pd.DataFrame:
    """
    Compute MMFF energy of each conformation.
    :param mol: RDKit mol object containing conformations.
    :param minimize: Whether or not to compute minimized energy.
    :return: Dataframe.
    """
    mol = Chem.Mol(mol)
    if minimize:
        max_iters = 200
    else:
        max_iters = 0
    res = AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=max_iters, numThreads=0)
    energies = []
    for i in range(len(res)):
        energies.append(res[i][1])
    energies = np.array(energies)
    df = pd.DataFrame(energies)
    if minimize:
        df = df.rename(columns={0: "Minimized Energy (kcal/mol)"})
    else:
        df = df.rename(columns={0: "Energy (kcal/mol)"})
    return df


# noinspection PyUnresolvedReferences
def compute_torsions(mol: rdchem.Mol, bonds: np.ndarray) -> pd.DataFrame:
    """
    Compute torsion angles for a set of bonds defined by pairs of atoms.
    :param mol: RDKit mol object containing conformations.
    :param bonds: Bonds defined by begin and end atoms.
    :return: Dataframe.
    """
    atom_indices = []
    column_names = dict()
    for i, bond in enumerate(bonds):
        # Get atom indices for the ith bond
        atom_a_idx = int(bond[0])
        atom_b_idx = int(bond[1])
        atom_a_symbol = mol.GetAtomWithIdx(atom_a_idx).GetSymbol()
        atom_b_symbol = mol.GetAtomWithIdx(atom_b_idx).GetSymbol()

        # Select a neighbor for each atom in order to form a dihedral
        atom_a_neighbors = mol.GetAtomWithIdx(atom_a_idx).GetNeighbors()
        atom_a_neighbor_index = [x.GetIdx() for x in atom_a_neighbors if x.GetIdx() != atom_b_idx][0]
        atom_b_neighbors = mol.GetAtomWithIdx(atom_b_idx).GetNeighbors()
        atom_b_neighbor_index = [x.GetIdx() for x in atom_b_neighbors if x.GetIdx() != atom_a_idx][0]

        atom_indices.append([atom_a_neighbor_index, atom_a_idx, atom_b_idx, atom_b_neighbor_index])
        column_names[i] = f'{bond[0]}-{bond[1]} | {atom_a_symbol} {atom_b_symbol}'

    results = None
    for i in range(len(bonds)):
        angles = []
        for j in range(mol.GetNumConformers()):
            c = mol.GetConformer(j)
            angles.append(rdMolTransforms.GetDihedralRad(c, atom_indices[i][0], atom_indices[i][1],
                                                         atom_indices[i][2], atom_indices[i][3]))
        angles = np.array(angles)
        if i == 0:
            results = angles[:, np.newaxis]
        else:
            # noinspection PyUnboundLocalVariable
            results = np.concatenate((results, angles[:, np.newaxis]), axis=1)

    df = pd.DataFrame(results)
    df = df.rename(columns=column_names)

    return df


def compute_periodic_modes(df: pd.DataFrame, shift: float = 0.1, bw_method: float = 0.1) -> pd.DataFrame:
    """
    Compute the number of modes for distributions of periodic variables, where each distribution
    is contained in column of an input DataFrame.
    :param df: DataFrame whose columns each contain a different distribution.
    :param shift: Amount by which to do incremental circular shifts of the distribution.
    :param bw_method: Estimator bandwidth (kde.factor).
    :return: DataFrame containing the mode count for each column of the input.
    """
    positions = np.arange(0.0, 2 * math.pi, shift)
    mode_counts = []
    for i in range(df.shape[1]):
        min_count = float('inf')
        for k in positions:
            count = 0
            kernel = gaussian_kde((df.iloc[:, i].to_numpy() + math.pi + k) % (2 * math.pi), bw_method=bw_method)
            Z = kernel(positions)
            diff = np.gradient(Z)
            s_diff = np.sign(diff)
            zc = np.where(s_diff[:-1] != s_diff[1:])[0]
            for j in zc:
                if s_diff[:-1][j] == 1.0 and s_diff[1:][j] == -1.0:
                    count += 1
            if count < min_count:
                min_count = count
        mode_counts.append([df.columns[i], min_count])

    df = pd.DataFrame(mode_counts)
    df = df.rename(columns={0: "Bond", 1: "Mode Count"})

    return df


def compute_entropy(df: pd.DataFrame, bin_width: float = 0.1) -> pd.DataFrame:
    """
    Compute entropy of the distribution in each column of a DataFrame via discrete histogram.
    :param df: DataFrame.
    :param bin_width: Histogram bin width.
    :return: DataFrame.
    """
    entropies = []
    for i in range(df.shape[1]):
        hist = np.histogram(df.iloc[:, i].to_numpy(), bins=np.arange(-math.pi - 1., math.pi + 1., bin_width),
                            density=True)[0]
        hist = np.where(hist == 0, 1e-10, hist)
        entropies.append([df.columns[i], entropy(hist)])

    df = pd.DataFrame(entropies)
    df = df.rename(columns={0: "Bond", 1: "Entropy"})

    return df


# noinspection PyUnresolvedReferences
def plot_torsion_joint_histogram(df: pd.DataFrame, weights: np.ndarray = None, bin_width: float = 0.1,
                                 joint_hist_bw_adjust: float = 0.25) -> matplotlib.figure.Figure:
    """
    Plot pairwise joint histogram of all torsion distributions in the given DataFrame.
    :param df: DataFrame of torsion angles for a set of conformations and bonds (# conformations x # bonds).
    :param weights: Histogram weights.
    :param bin_width: Histogram bin width.
    :param joint_hist_bw_adjust: bw_adjust value for kernel density estimate in lower triangle of grid.
    :return: Figure.
    """
    g = sns.PairGrid(df)
    g.set(ylim=(-math.pi - 1., math.pi + 1.), xlim=(-math.pi - 1., math.pi + 1.))
    g.map_upper(sns.histplot, bins=list(np.arange(-math.pi - 1., math.pi + 1., bin_width)), weights=weights)
    g.map_lower(sns.kdeplot, fill=True, weights=weights, bw_adjust=joint_hist_bw_adjust)
    g.map_diag(sns.histplot, bins=list(np.arange(-math.pi - 1., math.pi + 1., bin_width)), weights=weights)

    return g.fig


# noinspection PyUnresolvedReferences
def plot_torsion_pairwise_correlation(df: pd.DataFrame, ax=None, corr_heatmap_annot_size: float = 6.0) -> \
        matplotlib.axes.Axes:
    """
    Plot pairwise correlations of all torsions distributions in the given DataFrame.
    :param df: DataFrame of torsion angles for a set of conformations and bonds (# conformations x # bonds).
    :param ax: matplotlib Axes.
    :param corr_heatmap_annot_size: Font size for annotations in pairwise torsion correlations heatmap
    :return: matplotlib Axes.
    """
    # noinspection PyUnresolvedReferences
    ax = sns.heatmap(df.corr(), cmap=plt.cm.seismic, annot=True, annot_kws={'size': corr_heatmap_annot_size},
                     vmin=-1, vmax=1, ax=ax)
    plt.yticks(va='center')
    return ax


# noinspection PyUnresolvedReferences
def plot_energy_histogram(df: pd.DataFrame, ax=None, hist_bin_width: float = 0.1) -> matplotlib.axes.Axes:
    """
    Plot energy histogram.
    :param df: DataFrame containing energies of each conformation (# conformations x 1)
    :param ax: Axes object.
    :param hist_bin_width: Bin width for histogram.
    :return: Axes object.
    """
    return sns.histplot(df, bins=np.arange(min(df.iloc[:, 0]) - 1., max(df.iloc[:, 0]) + 1., hist_bin_width), ax=ax)


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
    df = compute_energy(mol)
    sns.set_theme()
    ax = plot_energy_histogram(df)
    ax.figure.savefig(os.path.join(args.save_dir, "energy-histogram.png"))
    plt.close()

    print("Computing minimized energies")
    df = compute_energy(mol, minimize=True)
    sns.set_theme()
    ax = plot_energy_histogram(df)
    ax.figure.savefig(os.path.join(args.save_dir, "minimized-energy-histogram.png"))
    plt.close()

    if args.weights:
        print("Computing and plotting Boltzmann weights")
        weights = compute_energy_weights(mol, k_b, temp, avogadro)
        plt.plot(weights)
        plt.ylim((-0.1, 1))
        plt.savefig(os.path.join(args.save_dir, "weights.png"))
        plt.close()
    else:
        weights = None

    distributions = []

    print("Computing rotatable bond angles")
    rotatable_bonds = mol.GetSubstructMatches(RotatableBondSmarts)
    distributions.append([compute_torsions(mol, np.array(rotatable_bonds)), "rotatable_bond"])

    print("Computing angles of bonds in aromatic rings")
    aromatic_bonds = []
    for bond in mol.GetBonds():
        if bond.GetBeginAtom().GetIsAromatic() and bond.GetEndAtom().GetIsAromatic():
            aromatic_bonds.append([bond.GetBeginAtom().GetIdx(), bond.GetEndAtom().GetIdx()])
    distributions.append([compute_torsions(mol, np.array(aromatic_bonds)), "aromatic_bond"])

    print("Computing angles of non-rotatable, non-aromatic ring bonds")
    non_aromatic_ring_bonds = []
    for bond in mol.GetBonds():
        if not bond.GetBeginAtom().GetIsAromatic() or not bond.GetEndAtom().GetIsAromatic():
            if (bond.GetBeginAtom().GetIdx(), bond.GetEndAtom().GetIdx()) not in rotatable_bonds:
                if bond.IsInRing():
                    non_aromatic_ring_bonds.append([bond.GetBeginAtom().GetIdx(), bond.GetEndAtom().GetIdx()])
    distributions.append([compute_torsions(mol, np.array(non_aromatic_ring_bonds)), "non_aromatic_ring_bond"])

    print("Computing angles of non-rotatable, non-ring, non-terminal bonds")
    non_rotatable_non_ring_bonds = []
    for bond in mol.GetBonds():
        if (bond.GetBeginAtom().GetIdx(), bond.GetEndAtom().GetIdx()) not in rotatable_bonds:
            if not bond.IsInRing() and len(bond.GetBeginAtom().GetNeighbors()) > 1 and \
                    len(bond.GetEndAtom().GetNeighbors()) > 1:
                non_rotatable_non_ring_bonds.append([bond.GetBeginAtom().GetIdx(), bond.GetEndAtom().GetIdx()])
    distributions.append([compute_torsions(mol, np.array(non_rotatable_non_ring_bonds)),
                          "non_rotatable_non_ring_non_terminal_bonds"])

    for i in range(len(distributions)):
        df = distributions[i][0]
        label = distributions[i][1]
        if not df.empty:
            print("Approximating number of modes")
            df_modes = compute_periodic_modes(df)
            sns.set(font_scale=args.mode_count_font_scale)
            sns.barplot(y="Bond", x="Mode Count", data=df_modes, color="steelblue")
            plt.savefig(os.path.join(args.save_dir, f'{label}_mode_count.png'), dpi=args.mode_count_dpi)
            plt.close()
            sns.set_theme()

            print(f'Rank of {label} angle matrix: {np.linalg.matrix_rank(df.to_numpy(), tol=args.svd_tol)}')

            print("Computing entropy distributions")
            df_entropies = compute_entropy(df)
            sns.set(font_scale=args.mode_count_font_scale)
            sns.barplot(y="Bond", x="Entropy", data=df_entropies, color="steelblue")
            plt.savefig(os.path.join(args.save_dir, f'{label}_entropy.png'), dpi=args.mode_count_dpi)
            plt.close()
            sns.set_theme()

            print("Plotting pairwise joint histograms")
            g = plot_torsion_joint_histogram(df, weights)
            g.savefig(os.path.join(args.save_dir, f'{label}_joint_histograms_seaborn.png'))
            plt.close()

            print("Plotting heatmap of pairwise correlation coefficients")
            sns.set(font_scale=args.corr_heatmap_font_scale)
            ax = plot_torsion_pairwise_correlation(df)
            ax.figure.savefig(os.path.join(args.save_dir, f'{label}_joint_correlations_seaborn.png'),
                              dpi=args.corr_heatmap_dpi)
            plt.close()
            sns.set_theme()

    results = None
    column_names = dict()
    num_cols = 0
    for i in range(len(distributions)):
        df = distributions[i][0]
        label = distributions[i][1]
        if not df.empty:
            if results is None:
                results = pd.DataFrame(compute_entropy(df).iloc[:, 1].to_numpy())
            else:
                results = pd.concat([results, pd.DataFrame(compute_periodic_modes(df).iloc[:, 1].to_numpy())],
                                    axis=1, ignore_index=True)
            column_names[num_cols] = label
            num_cols += 1
    results = results.rename(columns=column_names)
    sns.histplot(results, bins=args.entropy_bins)
    plt.savefig(os.path.join(args.save_dir, "entropies.png"))
    plt.close()

    results = None
    column_names = dict()
    num_cols = 0
    for i in range(len(distributions)):
        df = distributions[i][0]
        label = distributions[i][1]
        if not df.empty:
            if results is None:
                results = pd.DataFrame(compute_periodic_modes(df).iloc[:, 1].to_numpy())
            else:
                results = pd.concat([results, pd.DataFrame(compute_periodic_modes(df).iloc[:, 1].to_numpy())],
                                    axis=1, ignore_index=True)
            column_names[num_cols] = label
            num_cols += 1
    results = results.rename(columns=column_names)
    sns.histplot(results, bins=args.entropy_bins, discrete=True, binwidth=0.2)
    plt.savefig(os.path.join(args.save_dir, "mode_counts.png"))
    plt.close()

    print("Computing minimized energies")
    res = AllChem.MMFFOptimizeMoleculeConfs(mol, numThreads=0)
    energies = []
    for i in range(len(res)):
        energies.append(res[i][1])

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
    # noinspection PyUnresolvedReferences
    mol = Chem.Mol(mol)
    mol.RemoveAllConformers()
    d = rdMolDraw2D.MolDraw2DCairo(500, 500)
    # noinspection PyArgumentList
    d.drawOptions().addAtomIndices = True
    rdMolDraw2D.PrepareAndDrawMolecule(d, mol)
    with open(os.path.join(args.save_dir, 'molecule.png'), 'wb') as f:
        # noinspection PyArgumentList
        f.write(d.GetDrawingText())
