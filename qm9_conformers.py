""" Generate conformations from QM9 sdf file. """
from conformation.qm9_conformers import qm9_conformers, Args

if __name__ == '__main__':
    qm9_conformers(Args().parse_args())
