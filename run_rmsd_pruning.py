""" RMSD pruning of RDKit conformations. """
from conformation.run_rmsd_pruning import run_rmsd_pruning, Args

if __name__ == '__main__':
    run_rmsd_pruning(Args().parse_args())