""" Generate conformations using RDKit. """
import os

# noinspection PyUnresolvedReferences
from rdkit import Chem
# noinspection PyUnresolvedReferences
from rdkit.Chem import AllChem, rdMolTransforms
# noinspection PyPackageRequirements
from tap import Tap
from typing import List

from conformation.distance_matrix import dist_matrix


class Args(Tap):
    """
    System arguments.
    """
    bin_path: str  # Path to RDKit binary file containing molecule
    save_dir: str  # Directory for saving conformations
    num_configs: int = 10000  # Number of conformations to generate
    minimize: bool = False  # Whether or not to minimize conformations
    max_iter: int = 200  # Max iter for MMFF optimization
    dihedral: bool = False  # Use when computing dihedral angle values
    dihedral_vals: List[int] = [2, 0, 1, 5]  # Atom IDs for dihedral


def rdkit_conformers(args: Args) -> None:
    """
    Generate conformations for a molecule.
    :param args: Argparse arguments.
    :return: None.
    """
    os.makedirs(args.save_dir)
    os.makedirs(os.path.join(args.save_dir, "properties"))
    os.makedirs(os.path.join(args.save_dir, "distmat"))

    # noinspection PyUnresolvedReferences
    m2 = Chem.Mol(open(args.bin_path, "rb").read())
    _ = AllChem.EmbedMultipleConfs(m2, numConfs=args.num_configs, numThreads=0)
    if args.minimize:
        res = AllChem.MMFFOptimizeMoleculeConfs(m2, maxIters=args.max_iter, numThreads=0)
    else:
        res = AllChem.MMFFOptimizeMoleculeConfs(m2, maxIters=0, numThreads=0)
    rms_list = [0.0]
    AllChem.AlignMolConformers(m2, RMSlist=rms_list)

    i = 0
    for c in m2.GetConformers():
        pos = c.GetPositions()
        dist_matrix(pos, os.path.join(args.save_dir, "distmat", "distmat-" + str(i)))
        with open(os.path.join(args.save_dir, "properties", "energy-rms-dihedral-" + str(i) + ".txt"), "w") as f:
            f.write("energy: " + str(res[i][1]))
            f.write('\n')
            f.write("rms: " + str(rms_list[i]))
            f.write('\n')
            if args.dihedral:
                dihedral = rdMolTransforms.GetDihedralRad(c, args.dihedral_vals[0], args.dihedral_vals[1],
                                                          args.dihedral_vals[2], args.dihedral_vals[3])
            else:
                dihedral = "nan"
            f.write("dihedral: " + str(dihedral))
        i += 1

    # Print the conformations to a binary file
    bin_str = m2.ToBinary()
    with open(os.path.join(args.save_dir, "conformations.bin"), "wb") as f:
        f.write(bin_str)
