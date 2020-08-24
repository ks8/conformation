""" Compute distribution of singular values for distance matrices. """
import matplotlib.pyplot as plt
import numpy as np
import os

from rdkit import Chem
import seaborn as sns
# noinspection PyPackageRequirements
from tap import Tap

from conformation.distance_matrix import dist_matrix


class Args(Tap):
    """
    System arguments.
    """
    distmat_dir: str  # Path to directory containing distance matrices
    conf_path: str  # Path to binary file containing conformations
    save_path: str  # Path to output file
    precision: float = 1e-10  # Cutoff for non-zero singular values


def svd(args: Args):
    """
    Compute and plot distribution of singular values of a set of distance matrices.
    :return: None
    """
    # Compute the average conformation
    # noinspection PyUnresolvedReferences
    mol = Chem.Mol(open(args.conf_path, "rb").read())
    conformers = mol.GetConformers()
    c = conformers[0].GetPositions()
    avg_conf = np.zeros([c.shape[0], c.shape[1]])
    for c in conformers:
        avg_conf += c.GetPositions()
    avg_conf /= float(len(conformers))

    # Compute the distance matrix of the average conformation
    avg_conf_distmat = dist_matrix(avg_conf)

    # Compute singular values for the Euclidean distance matrix of the average conformation
    # Only include singular values as non-zero if they are greater than the specified precision
    avg_conf_s_vals = [x for x in list(np.linalg.svd(avg_conf_distmat*avg_conf_distmat)[1]) if x >= args.precision]
    avg_conf_num_s_vals = len(avg_conf_s_vals)

    print(f'average conformation num singular values: {avg_conf_num_s_vals}')

    # Compute singular values for the Euclidean distance matrix of each conformation
    # In addition, compute the average of the Euclidean distance matrices (data*data is the Euclidean distance matrix),
    # and save the values of each distance matrix in a list
    s_vals = []
    num_s_vals = []
    avg_distmat = None
    distmat_vals = []
    count = 0
    for root, _, files in os.walk(args.distmat_dir):
        for f in files:
            data = np.load(os.path.join(root, f))
            if avg_distmat is None:
                avg_distmat = data
            else:
                avg_distmat += data
            s = [x for x in list(np.linalg.svd(data*data)[1]) if x >= args.precision]
            s_vals += s
            num_s_vals.append(len(s))
            distmat_vals += list(data[np.triu_indices(data.shape[1], k=1)])
            count += 1

    # Compute singular values for the average Euclidean distance matrix
    avg_distmat /= float(count)
    avg_distmat_s_vals = [x for x in list(np.linalg.svd(avg_distmat*avg_distmat)[1]) if x >= args.precision]
    avg_distmat_num_s_vals = len(avg_distmat_s_vals)

    # Plot the distribution of singular values and counts
    fig = sns.distplot(s_vals, kde=False, label='all conf singular vals')
    for i in range(avg_conf_num_s_vals):
        if i < avg_conf_num_s_vals - 1:
            plt.axvline(avg_conf_s_vals[i], 0.0, 1.0, color='g')
        else:
            plt.axvline(avg_conf_s_vals[i], 0.0, 1.0, color='g', label='avg conf singular vals')
    plt.legend()
    fig.figure.savefig(args.save_path + "-distribution-singular-vals-with-avg-conf")
    plt.close()

    fig = sns.distplot(s_vals, kde=False, label='all conf singular vals')
    for i in range(avg_distmat_num_s_vals):
        if i < avg_distmat_num_s_vals - 1:
            plt.axvline(avg_distmat_s_vals[i], 0.0, 1.0, color='r')
        else:
            plt.axvline(avg_distmat_s_vals[i], 0.0, 1.0, color='r', label='avg distmat singular vals')
    plt.legend()
    fig.figure.savefig(args.save_path + "-distribution-singular-vals-with-avg-distmat")
    plt.close()

    fig = sns.distplot(num_s_vals, kde=False, bins=range(0, 40, 2), label='all conf num singular vals')
    plt.axvline(avg_conf_num_s_vals, 0.0, 1.0, color='g', label='avg conf num singular vals')
    plt.axvline(avg_distmat_num_s_vals, 0.0, 1.0, color='r', label='avg distmat num singular vals')
    plt.legend()
    fig.figure.savefig(args.save_path + "-distribution-singular-count-with-avgs")
    plt.close()

    avg_conf_dist_vec = avg_conf_distmat[np.triu_indices(avg_conf_distmat.shape[1], k=1)]
    avg_distmat_dist_vec = avg_distmat[np.triu_indices(avg_distmat.shape[1], k=1)]

    fig, ax = plt.subplots()
    sns.distplot(avg_conf_dist_vec, ax=ax, label='avg conf')
    sns.distplot(avg_distmat_dist_vec, ax=ax, label='avg distmat')
    sns.distplot(distmat_vals, ax=ax, label='all distmat vals')
    ax.legend()
    ax.figure.savefig(args.save_path + "-distribution-distmat-vals")

    print(f'average distmat num singular values: {avg_distmat_num_s_vals}')
