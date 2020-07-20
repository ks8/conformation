""" QM9 to SMILES. Dataset from: http://moleculenet.ai/datasets-1."""
from conformation.qm9_to_smiles import qm9_to_smiles, Args

if __name__ == '__main__':
    args = Args().parse_args()
    qm9_to_smiles(args)
