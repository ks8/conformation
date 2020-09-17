""" Systematic conformer search using Confab via Open Babel
https://open-babel.readthedocs.io/en/latest/3DStructureGen/multipleconformers.html. """
import copy
import matplotlib.pyplot as plt
import numpy as np
import os

from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign
import seaborn as sns
# noinspection PyPackageRequirements
from tap import Tap
from tqdm import tqdm


class Args(Tap):
    """
    System arguments.
    """
    smiles: str  # Molecular SMILES string
    rcutoff: float = 0.5  # RMSD cutoff
    ecutoff: float = 50.0  # Energy cutoff
    conf: int = 1000000  # Maximum number of conformations to check
    rmsd_remove_Hs: bool = False  # Whether or not to remove Hydrogen when computing RMSD values via RDKit
    post_rmsd_threshold: float = 0.65  # RMSD threshold for post minimized conformations
    save_dir: str  # Path to output file


def systematic_search(args: Args):
    """
    Systematic conformer search using Confab via Open Babel.
    :param args: System arguments.
    :return: None.
    """
    os.makedirs(args.save_dir)

    # Create SMILES file
    with open(os.path.join(args.save_dir, "tmp.smi"), "w") as f:
        f.write(args.smiles)

    # Generate 3D conformation
    print(f'Generating initial conformation...')
    os.system("obabel -ismi " + os.path.join(args.save_dir, "tmp.smi") + " -O " +
              os.path.join(args.save_dir, "tmp.sdf") + " --gen3D")

    # Generate conformers
    print(f'Generating conformers...')
    os.system("obabel " + os.path.join(args.save_dir, "tmp.sdf") + " -O " +
              os.path.join(args.save_dir, "conformations.sdf") + " --confab --rcutoff " + str(args.rcutoff) +
              " --ecutoff " + str(args.ecutoff) + " --conf " + str(args.conf) + " --verbose")

    # Compute energy distribution with Open Babel (kcal/mol)
    print(f'Computing energies via Open Babel...')
    energies = []
    os.system("obenergy -h -ff MMFF94 " + os.path.join(args.save_dir, "conformations.sdf") + " > " +
              os.path.join(args.save_dir, "tmp.energy"))
    with open(os.path.join(args.save_dir, "tmp.energy"), "r") as f:
        for line in f:
            if "TOTAL ENERGY" in line:
                energies.append(float(line.split()[3]))

    # Compute minimized energies with RDKit and perform RMSD pruning
    print(f'Loading conformations into RDKit...')
    suppl = Chem.SDMolSupplier(os.path.join(args.save_dir, "conformations.sdf"), removeHs=False)
    mol = None
    for i, tmp in enumerate(tqdm(suppl)):
        if i == 0:
            mol = tmp
        else:
            c = tmp.GetConformer()
            c.SetId(i)
            mol.AddConformer(c)

    print(f'Minimizing conformations via RDKit...')
    res = AllChem.MMFFOptimizeMoleculeConfs(mol, numThreads=0)
    rdkit_energies = []
    for i in range(len(res)):
        rdkit_energies.append(res[i][1])

    print('RMSD pruning via RDKit...')
    # List of unique post-minimization molecules
    post_conformation_molecules = []

    # Add an initial molecule to the list
    post_mol = copy.deepcopy(mol)
    post_mol.RemoveAllConformers()
    c = mol.GetConformers()[0]
    c.SetId(0)
    post_mol.AddConformer(c)
    post_conformation_molecules.append(post_mol)

    # Loop through the remaining conformations to find unique ones
    for i in tqdm(range(1, mol.GetNumConformers())):
        # Create a molecule with the current conformation we are checking for uniqueness
        post_mol = copy.deepcopy(mol)
        post_mol.RemoveAllConformers()
        c = mol.GetConformers()[i]
        c.SetId(0)
        post_mol.AddConformer(c)
        unique = True
        if args.post_rmsd_threshold > 0.0:
            for j in range(len(post_conformation_molecules)):
                # Check for uniqueness
                if args.rmsd_remove_Hs:
                    rmsd = rdMolAlign.GetBestRMS(Chem.RemoveHs(post_conformation_molecules[j]), Chem.RemoveHs(post_mol))
                else:
                    rmsd = rdMolAlign.GetBestRMS(post_conformation_molecules[j], post_mol)
                if rmsd < args.post_rmsd_threshold:
                    unique = False
                    break

        if unique:
            post_conformation_molecules.append(post_mol)

    print(f'Number of unique post minimization conformations identified: {len(post_conformation_molecules)}')
    # Save unique conformers in molecule object
    print(f'Saving RDKit-minimized conformations...')
    post_mol = post_conformation_molecules[0]
    for i in range(1, len(post_conformation_molecules)):
        c = post_conformation_molecules[i].GetConformer()
        c.SetId(i)
        post_mol.AddConformer(c)

    # Save pruned energies
    res = AllChem.MMFFOptimizeMoleculeConfs(post_mol, maxIters=0)
    post_rmsd_energies = []
    for i in range(len(res)):
        post_rmsd_energies.append(res[i][1])

    # Save molecule to binary file
    bin_str = post_mol.ToBinary()
    with open(os.path.join(args.save_dir, "post-minimization-conformations.bin"), "wb") as b:
        b.write(bin_str)

    # Plot energy histograms
    print(f'Plotting energy histograms...')
    info = ["obabel-energy", "post-minimization-rdkit-energy", "post-rmsd-rdkit-energy"]
    for i, elements in enumerate([energies, rdkit_energies, post_rmsd_energies]):
        fig, ax = plt.subplots()
        sns.histplot(elements, ax=ax, bins=np.arange(min(elements) - 1., max(elements) + 1., 0.1))
        ax.set_xlabel("Energy (kcal/mol)")
        ax.set_ylabel("Frequency")
        ax.figure.savefig(os.path.join(args.save_dir, info[i] + "-histogram.png"))
        plt.clf()
        plt.close()

    # Remove auxiliary files
    os.remove(os.path.join(args.save_dir, "tmp.smi"))
    os.remove(os.path.join(args.save_dir, "tmp.sdf"))
    os.remove(os.path.join(args.save_dir, "tmp.energy"))
