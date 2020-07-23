""" Null model for predicting pairwise distances based on minimum path length between pairs of atoms. """
from conformation.null_model import null_model, Args

if __name__ == '__main__':
    args = Args().parse_args()
    null_model(args)
