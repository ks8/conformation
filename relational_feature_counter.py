""" Determine which category of features are present in a dataset of molecules (bond types, atom types, etc.) """
from conformation.relational_feature_counter import relational_feature_counter, Args

if __name__ == '__main__':
    args = Args().parse_args()
    relational_feature_counter(args)
