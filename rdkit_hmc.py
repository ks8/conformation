""" Hamiltonian Monte Carlo conformational search using RDKit. """
from conformation.rdkit_hmc import rdkit_hmc, Args

if __name__ == '__main__':
    rdkit_hmc(Args().parse_args())
