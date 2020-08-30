""" Systematic conformer search using Confab via Open Babel
https://open-babel.readthedocs.io/en/latest/3DStructureGen/multipleconformers.html. """
import os

# noinspection PyPackageRequirements
from tap import Tap


class Args(Tap):
    """
    System arguments.
    """
    smiles: str  # Molecular SMILES string
    save_path: str  # Path to output file
    rcutoff: float = 0.5  # RMSD cutoff
    ecutoff: float = 50.0  # Energy cutoff
    conf: int = 1000000  # Maximum number of conformations to check


def systematic_search(args: Args):
    """
    Systematic conformer search using Confab via Open Babel.
    :param args: System arguments.
    :return: None.
    """
    # Create SMILES file
    with open("tmp.smi", "w") as f:
        f.write(args.smiles)

    # Generate 3D conformation
    os.system("obabel -ismi tmp.smi -O tmp.sdf --gen3D")

    # Generate conformers
    os.system("obabel tmp.sdf -O " + args.save_path + ".sdf --confab --rcutoff " + str(args.rcutoff) + " --ecutoff " +
              str(args.ecutoff) + " --conf " + str(args.conf) + " --verbose")

    # Remove auxiliary files
    os.remove("tmp.smi")
    os.remove("tmp.sdf")
