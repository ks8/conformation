""" Generate conformations using RDKit. """
from conformation.rdkit_conformers import rdkit_conformers, Args

if __name__ == '__main__':
    args = Args().parse_args()
    rdkit_conformers(args)
