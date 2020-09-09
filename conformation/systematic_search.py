""" Systematic conformer search using Confab via Open Babel
https://open-babel.readthedocs.io/en/latest/3DStructureGen/multipleconformers.html. """
import copy
import matplotlib.pyplot as plt
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
    save_path: str  # Path to output file
    rcutoff: float = 0.5  # RMSD cutoff
    ecutoff: float = 50.0  # Energy cutoff
    conf: int = 1000000  # Maximum number of conformations to check
    rmsd_remove_Hs: bool = False  # Whether or not to remove Hydrogen when computing RMSD values via RDKit
    post_rmsd_threshold: float = 0.65  # RMSD threshold for post minimized conformations


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

    # Compute energy distribution with Open Babel (kcal/mol)
    energies = []
    os.system("obenergy -h -ff MMFF94 " + args.save_path + ".sdf > tmp.energy")
    with open("tmp.energy", "r") as f:
        for line in f:
            if "TOTAL ENERGY" in line:
                energies.append(float(line.split()[3]))

    # Compute minimized energies with RDKit
    print(f'Minimizing conformations...')
    rdkit_energies = []
    post_conformation_molecules = []
    suppl = Chem.SDMolSupplier(args.save_path + ".sdf", removeHs=False)
    for i, mol in enumerate(tqdm(suppl)):
        res = AllChem.MMFFOptimizeMoleculeConfs(mol, numThreads=0)
        rdkit_energies.append(res[0][1])
        post_mol = copy.deepcopy(mol)
        if i == 0:
            post_conformation_molecules.append(post_mol)
        else:
            unique = True
            for j in range(len(post_conformation_molecules)):
                # Check for uniqueness
                if args.rmsd_remove_Hs:
                    rmsd = rdMolAlign.GetBestRMS(Chem.RemoveHs(post_conformation_molecules[j]),
                                                 Chem.RemoveHs(post_mol))
                else:
                    rmsd = rdMolAlign.GetBestRMS(post_conformation_molecules[j], post_mol)
                if rmsd < args.post_rmsd_threshold:
                    unique = False
                    break

            if unique:
                post_conformation_molecules.append(post_mol)

    print(f'Number of unique post minimization conformations identified: {len(post_conformation_molecules)}')
    # Save unique conformers in molecule object
    print(f'Saving conformations...')
    post_mol = post_conformation_molecules[0]
    for i in range(1, len(post_conformation_molecules)):
        c = post_conformation_molecules[i].GetConformer()
        c.SetId(i)
        post_mol.AddConformer(c)

    # Save molecule to binary file
    bin_str = post_mol.ToBinary()
    with open(args.save_path + "-post-minimization-rmsd-conformations.bin", "wb") as b:
        b.write(bin_str)

    # Plot energy distributions
    print(f'Plotting energy distributions...')
    info = ["energy", "post-minimization-rdkit-energy"]
    for i, elements in enumerate([energies, rdkit_energies]):
        fig, ax = plt.subplots()
        sns.distplot(elements, ax=ax)
        ax.set_xlabel("Energy (kcal/mol)")
        ax.set_ylabel("Density")
        ax.figure.savefig(args.save_path + "-" + info[i] + "-distribution.png")
        plt.clf()
        plt.close()

    # Remove auxiliary files
    os.remove("tmp.smi")
    os.remove("tmp.sdf")
    os.remove("tmp.energy")
