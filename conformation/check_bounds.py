""" Generate distance matrices from molecular conformations. """
import itertools
import os
import matplotlib.pyplot as plt
import numpy as np
import pickle

from rdkit import Chem
from rdkit.Chem.rdDistGeom import GetMoleculeBoundsMatrix
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import rdmolops
import seaborn as sns
# noinspection PyPackageRequirements
from tap import Tap
from tqdm import tqdm

from conformation.distance_matrix import dist_matrix


class Args(Tap):
    """
    System arguments.
    """
    conf_path: str  # Path to binary file containing mol with conformations
    save_dir: str  # Path to directory containing output files


def conf_to_distmat(args: Args) -> None:
    """
    Generate distance matrices from molecular conformations.
    :param args: System arguments.
    """
    os.makedirs(args.save_dir)

    # noinspection PyUnresolvedReferences
    mol = Chem.Mol(open(args.conf_path, "rb").read())
    num_atoms = mol.GetNumAtoms()
    num_conformers = mol.GetNumConformers()
    bounds = GetMoleculeBoundsMatrix(mol)
    bounds_check = np.zeros([int(num_atoms*(num_atoms - 1)/2)])
    bounds_check_num = np.zeros([int(num_atoms*(num_atoms - 1)/2)])
    dist = dict()

    for i in tqdm(range(num_conformers)):
        pos = mol.GetConformer(i).GetPositions()
        distmat = dist_matrix(pos)
        index = 0
        for j, k in itertools.combinations(np.arange(num_atoms), 2):
            if bounds[k][j] > distmat[j][k]:
                bounds_check[index] += distmat[j][k] - bounds[k][j]
                bounds_check_num[index] += 1
            elif bounds[j][k] < distmat[j][k]:
                bounds_check[index] += distmat[j][k] - bounds[j][k]
                bounds_check_num[index] += 1
            if (j, k) not in dist:
                dist[(j, k)] = [distmat[j][k]]
            else:
                dist[(j, k)].append(distmat[j][k])
            index += 1
    bounds_check /= num_conformers
    bounds_check_num /= num_conformers

    np.save(os.path.join(args.save_dir, "bounds.npy"), bounds)
    np.save(os.path.join(args.save_dir, "bounds_check.npy"), bounds_check)
    np.save(os.path.join(args.save_dir, "bounds_check_num.npy"), bounds_check_num)
    pickle.dump(dist, open(os.path.join(args.save_dir, "dist.p"), "wb"))

    index_dict = dict()
    index = 0
    for j, k in itertools.combinations(np.arange(num_atoms), 2):
        index_dict[index] = (j, k)
        index += 1

    sort_indices = np.argsort(bounds_check_num)
    sort_indices = np.flip(sort_indices)

    path_lengths = []

    for i in tqdm(range(len(sort_indices))):
        (j, k) = index_dict[sort_indices[i]]
        path_lengths.append(len(rdmolops.GetShortestPath(mol, int(j), int(k))) - 1)
        sns.histplot(dist[(j, k)])
        plt.axvline(bounds[j][k], color='r')
        plt.axvline(bounds[k][j], color='r')
        plt.xlim((min(bounds[k][j], min(dist[(j, k)])) - 0.01, max(bounds[j][k], max(dist[(j, k)])) + 0.01))
        plt.savefig(os.path.join(args.save_dir, f'dist_{i}.png'))
        plt.close()

        d = rdMolDraw2D.MolDraw2DCairo(500, 500)
        rdMolDraw2D.PrepareAndDrawMolecule(d, mol, highlightAtoms=[int(j), int(k)])

        with open(os.path.join(args.save_dir, f'mol_{i}.png'), 'wb') as f:
            # noinspection PyArgumentList
            f.write(d.GetDrawingText())

    fig, axes = plt.subplots(2)
    axes[0].plot(path_lengths, marker='o', markersize=2)
    axes[1].plot(np.flip(np.sort(bounds_check_num)), marker='o', markersize=2, color='r')
    axes[1].set_ylim([0., 1.])
    plt.savefig(os.path.join(args.save_dir, "path_lengths.png"))
    plt.close()
