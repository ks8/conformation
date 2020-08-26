""" Metropolis-Hastings conformational search using RDKit. """
from conformation.rdkit_metropolis import rdkit_metropolis, Args

if __name__ == '__main__':
    rdkit_metropolis(Args().parse_args())
