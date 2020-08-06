""" Embed molecules from smiles strings. """
from conformation.embed_molecules import embed_molecules, Args

if __name__ == '__main__':
    embed_molecules(Args().parse_args())
