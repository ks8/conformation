""" Systematic conformer search using Confab via Open Babel
https://open-babel.readthedocs.io/en/latest/3DStructureGen/multipleconformers.html. """
from logging import Logger
import os
from typing_extensions import Literal
import time

from rdkit import Chem
from rdkit.Chem import AllChem, rdmolfiles
# noinspection PyPackageRequirements
from tap import Tap
from tqdm import tqdm


class Args(Tap):
    """
    System arguments.
    """
    smiles: str  # Molecular SMILES string
    generator: Literal["rdkit", "obabel"] = "obabel"  # Specify which program to use for generating initial structure
    rcutoff: float = 0.5  # RMSD cutoff
    ecutoff: float = 50.0  # Energy cutoff
    conf: int = 1000000  # Maximum number of conformations to check
    init_minimize: bool = False  # Whether or not to FF-minimize the initial ETKDG-generated conformation
    # (generator="rdkit)
    save_dir: str  # Path to output directory


def confab(args: Args, logger: Logger):
    """
    Systematic conformer search using Confab via Open Babel.
    :param args: System arguments.
    :param logger: System logger.
    :return: None.
    """
    # Set up logger
    debug, info = logger.debug, logger.info

    print(f'Generating initial conformation...')
    if args.generator == "rdkit":
        # Load molecule
        mol = Chem.MolFromSmiles(args.smiles)
        mol = Chem.AddHs(mol)

        # Embed molecule
        # NOTE: This will produce the same embedding each time the program is run
        AllChem.EmbedMolecule(mol)

        # Minimize if required
        if args.init_minimize:
            AllChem.MMFFOptimizeMoleculeConfs(mol)

        # Save molecule as PDB file
        print(rdmolfiles.MolToPDBBlock(mol), file=open(os.path.join(args.save_dir, "tmp.pdb"), "w+"))

        # Convert PDB to SDF using Open Babel
        os.system("obabel -ipdb " + os.path.join(args.save_dir, "tmp.pdb") + " -osdf -O " +
                  os.path.join(args.save_dir, "tmp.sdf"))

    else:
        # Create SMILES file
        with open(os.path.join(args.save_dir, "tmp.smi"), "w") as f:
            f.write(args.smiles)

        os.system("obabel -ismi " + os.path.join(args.save_dir, "tmp.smi") + " -O " +
                  os.path.join(args.save_dir, "tmp.sdf") + " --gen3D")

    start_time = time.time()
    # Generate conformers using Confab
    debug(f'Generating conformers...')
    os.system("obabel " + os.path.join(args.save_dir, "tmp.sdf") + " -O " +
              os.path.join(args.save_dir, "conformations.sdf") + " --confab --rcutoff " + str(args.rcutoff) +
              " --ecutoff " + str(args.ecutoff) + " --conf " + str(args.conf) + " --verbose")
    end_time = time.time()
    debug(f'Total Time (s): {end_time - start_time}')

    # Load conformations into RDKit and save
    debug(f'Loading conformations into RDKit...')
    suppl = Chem.SDMolSupplier(os.path.join(args.save_dir, "conformations.sdf"), removeHs=False)
    mol = None
    for i, tmp in enumerate(tqdm(suppl)):
        if i == 0:
            mol = tmp
        else:
            c = tmp.GetConformer()
            c.SetId(i)
            mol.AddConformer(c)

    debug(f'Saving conformations from RDKit...')
    bin_str = mol.ToBinary()
    with open(os.path.join(args.save_dir, "conformations.bin"), "wb") as b:
        b.write(bin_str)

    # Remove auxiliary files
    if args.generator == "rdkit":
        os.remove(os.path.join(args.save_dir, "tmp.pdb"))
    os.remove(os.path.join(args.save_dir, "tmp.sdf"))
