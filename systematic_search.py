""" Systematic conformer search using Confab via Open Babel
https://open-babel.readthedocs.io/en/latest/3DStructureGen/multipleconformers.html. """
from conformation.systematic_search import systematic_search, Args

if __name__ == '__main__':
    systematic_search(Args().parse_args())
