""" Generate metadata. """
import json
import os

# noinspection PyPackageRequirements
from tap import Tap


class Args(Tap):
    """
    System arguments.
    """
    data_dir: str  # Path to directory containing data
    save_dir: str  # Path to directory containing output file
    mpnn: bool = False  # Whether or not to produce metadata for graph neural network training
    smiles_dir: str = None  # Path to directory containing smiles strings (mpnn = True)


def metadata(args: Args) -> None:
    """
    Create metadata folder and file.
    :param args: Folder name info.
    :return None.
    """
    data = []
    for _, _, files in os.walk(args.data_dir):
        for f in files:
            path = os.path.join(args.data_dir, f)
            if args.mpnn:
                molecule_name = f[f.find("qm9"):f.find(".")]
                with open(os.path.join(args.smiles_dir, molecule_name + ".smiles")) as tmp:
                    smiles = tmp.readlines()[0].split()[0]
                data.append({'smiles': smiles, 'target': path})
            else:
                data.append({'path': path})
    os.makedirs(args.save_dir)
    json.dump(data, open(os.path.join(args.save_dir, args.save_dir + ".json"), "w"), indent=4, sort_keys=True)
