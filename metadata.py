""" Generate metadata. """
from conformation.metadata import metadata, Args

if __name__ == '__main__':
    metadata(Args().parse_args())
