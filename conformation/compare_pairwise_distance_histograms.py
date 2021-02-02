""" Plot matrix of pairwise distance histograms for two sets of conformations. """
import itertools
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns

from rdkit import Chem
from rdkit.Chem import rdmolops
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

    fig = plt.figure(constrained_layout=False)
    gs = fig.add_gridspec(14, 14)
    axes_list = []
    for i in range(14):
        for j in range(14):
            axes_list.append(fig.add_subplot(gs[i, j]))

    torsion_list = []
    for i in range(14):
        for j in range(14):
            if i != j:
                if len(rdmolops.GetShortestPath(mol, int(i), int(j))) - 1 == 3:
                    torsion_list.append(1)
                else:
                    torsion_list.append(0)
            else:
                torsion_list.append(0)

    for index, ax in enumerate(axes_list):
        if torsion_list[index] == 0:
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['right'].set_visible(False)
        ax.set(xticks=[], yticks=[])

    index = 0
    for i in range(14):
        for j in range(14):
            if i < j:
                sns.histplot(dist[(i, j)], ax=axes_list[index], color="blue", stat='density',
                             bins=np.arange(min(min(dist[(i, j)]), min(dist2[(i, j)])), max(max(dist[(i, j)]),
                                                                                            max(dist2[i, j])), 0.01),
                             kde=True)

                sns.histplot(dist2[(i, j)], ax=axes_list[index], color="red", stat='density',
                             bins=np.arange(min(min(dist[(i, j)]), min(dist2[(i, j)])), max(max(dist[(i, j)]),
                                                                                            max(dist2[i, j])), 0.01),
                             kde=True)
            elif i > j:
                sns.histplot(dist[(j, i)], ax=axes_list[index], color="blue", stat='density',
                             bins=np.arange(min(min(dist[(j, i)]), min(dist2[(j, i)])), max(max(dist[(j, i)]),
                                                                                            max(dist2[j, i])), 0.01),
                             kde=True)

                sns.histplot(dist2[(j, i)], ax=axes_list[index], color="red", stat='density',
                             bins=np.arange(min(min(dist[(j, i)]), min(dist2[(j, i)])), max(max(dist[(j, i)]),
                                                                                            max(dist2[j, i])), 0.01),
                             kde=True)

            index += 1

    plt.savefig(os.path.join(args.save_dir, "test.png"))
