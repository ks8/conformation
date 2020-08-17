""" Analyze properties of a single molecule in RDKit. """
from conformation.single_molecule_analysis import single_molecule_analysis, Args

if __name__ == '__main__':
    single_molecule_analysis(Args().parse_args())
