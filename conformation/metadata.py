""" Generate metadata. """
import argparse
from argparse import Namespace
import json
import os


def process_metadata(args: Namespace) -> None:
    """
    Create metadata folder and file.
    :param args: Folder name info
    """

    data = []
    for root, _, files in os.walk(args.data_dir):
        for f in files:
            path = os.path.join(root, f)
            data.append({'path': path})
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    json.dump(data, open(os.path.join(args.out_dir, args.out_dir + ".json"), "w"), indent=4, sort_keys=True)


def main():
    """
    Parse arguments and execute metadata creation.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, dest='data_dir', default=None, help='Directory containing data')
    parser.add_argument('--out_dir', type=str, dest='out_dir', default=None, help='Directory name for metadata')

    args = parser.parse_args()
    process_metadata(args)


if __name__ == "__main__":
    main()
