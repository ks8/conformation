""" Plot matrix of pairwise distance histograms for two sets of conformations. """
import itertools
import math
import matplotlib.pyplot as plt
import numpy as np
import os

from rdkit import Chem
from rdkit.Chem import AllChem, rdmolops
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
    num_bins: int = 50  # Number of histogram bins
    line_width: float = 0.5  # Plot line width
    weight_2: bool = False
    save_dir: str  # Path to directory containing output files


def compare_pairwise_distance_histograms(args: Args) -> None:
    """
    Plot matrix of pairwise distance histograms for two sets of conformations.
    :param args: System args.
    """
    os.makedirs(args.save_dir)

    # noinspection PyUnresolvedReferences
    mol = Chem.Mol(open(args.conf_path_1, "rb").read())
    num_atoms = mol.GetNumAtoms()
    num_conformers = mol.GetNumConformers()
    dist = dict()

    for i in tqdm(range(num_conformers)):
        pos = mol.GetConformer(i).GetPositions()
        distmat = dist_matrix(pos)
        for j, k in itertools.combinations(np.arange(num_atoms), 2):
            if (j, k) not in dist:
                dist[(j, k)] = [distmat[j][k]]
            else:
                dist[(j, k)].append(distmat[j][k])

    # noinspection PyUnresolvedReferences
    mol2 = Chem.Mol(open(args.conf_path_2, "rb").read())
    num_atoms2 = mol2.GetNumAtoms()
    num_conformers2 = mol2.GetNumConformers()
    dist2 = dict()

    for i in tqdm(range(num_conformers2)):
        pos = mol2.GetConformer(i).GetPositions()
        distmat = dist_matrix(pos)
        for j, k in itertools.combinations(np.arange(num_atoms2), 2):
            if (j, k) not in dist2:
                dist2[(j, k)] = [distmat[j][k]]
            else:
                dist2[(j, k)].append(distmat[j][k])

    # Compute energy weights for mol2
    k_b = 3.297e-24
    temp = 300.0
    avogadro = 6.022e23
    res = AllChem.MMFFOptimizeMoleculeConfs(mol2, maxIters=0, numThreads=0)
    probabilities = []
    for i in range(len(res)):
        energy = res[i][1] * (1000.0 * 4.184 / avogadro)
        probabilities.append(math.exp(-energy / (k_b * temp * 4.184)))
    Z = sum(probabilities)
    for i in range(len(probabilities)):
        probabilities[i] /= Z
    probabilities = np.array(probabilities)

    if args.weight_2:
        weights = probabilities
    else:
        weights = None

    fig = plt.figure(constrained_layout=False)
    gs = fig.add_gridspec(num_atoms, num_atoms)

    for i in range(num_atoms):
        for j in range(num_atoms):
            if i != j:
                ax = fig.add_subplot(gs[i, j])
                ax.hist(dist[(min(i, j), max(i, j))], density=True, histtype='step',
                        bins=args.num_bins, linewidth=args.line_width)
                ax.hist(dist2[(min(i, j), max(i, j))], density=True, histtype='step',
                        bins=args.num_bins, linewidth=args.line_width, weights=weights)
                if len(rdmolops.GetShortestPath(mol, int(i), int(j))) - 1 != 3:
                    ax.spines['top'].set_visible(False)
                    ax.spines['bottom'].set_visible(False)
                    ax.spines['left'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                ax.set(xticks=[], yticks=[])

    plt.savefig(os.path.join(args.save_dir, "test.png"))
