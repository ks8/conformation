""" Compare two sets of conformations. """
from conformation.compare_conformations import compare_conformations, Args

if __name__ == '__main__':
    compare_conformations(Args().parse_args())
