""" Compare two sets of conformations. """
import copy

from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign
# noinspection PyPackageRequirements
from tap import Tap


class Args(Tap):
    """
    System arguments.
    """
    conf_path_1: str  # Path to binary file containing first set of conformations
    conf_path_2: str  # Path to binary file containing second set of conformations
    rmsd_threshold: float = 0.65  # RMSD threshold
    post_rmsd_threshold: float = 0.5
    rmsd_remove_Hs: bool = False  # Whether or not to remove Hydrogen when computing RMSD values via RDKit


def compare_conformations(args: Args):
    """
    Compare two sets of conformations.
    :param args: System arguments.
    :return: None.
    """

    # Load molecules
    # noinspection PyUnresolvedReferences
    mol1 = Chem.Mol(open(args.conf_path_1, "rb").read())
    # noinspection PyUnresolvedReferences
    mol2 = Chem.Mol(open(args.conf_path_2, "rb").read())

    # Compute common conformations
    conformation_molecules = []
    for molecule in [mol1]:
        for i in range(molecule.GetNumConformers()):
            # Create a molecule with the current conformation we are checking for uniqueness
            check_mol = copy.deepcopy(molecule)
            check_mol.RemoveAllConformers()
            c = molecule.GetConformers()[i]
            c.SetId(0)
            check_mol.AddConformer(c)
            unique = True
            for j in range(len(conformation_molecules)):
                # Check for uniqueness
                if args.rmsd_remove_Hs:
                    rmsd = rdMolAlign.GetBestRMS(Chem.RemoveHs(conformation_molecules[j]), Chem.RemoveHs(check_mol))
                else:
                    rmsd = rdMolAlign.GetBestRMS(conformation_molecules[j], check_mol)
                if rmsd < args.rmsd_threshold:
                    unique = False
                    break

            if unique:
                conformation_molecules.append(check_mol)

    print(f'Number of Unique Conformations Identified: {len(conformation_molecules)}')
    # noinspection SpellCheckingInspection
    num_confs_mol1 = 0
    # noinspection SpellCheckingInspection
    num_confs_mol2 = 0
    for molecule in [(mol1, "mol1"), (mol2, "mol2")]:
        for i in range(len(conformation_molecules)):
            for j in range(molecule[0].GetNumConformers()):
                # Create a molecule with the current conformation we are checking for uniqueness
                check_mol = copy.deepcopy(molecule[0])
                check_mol.RemoveAllConformers()
                c = molecule[0].GetConformers()[j]
                c.SetId(0)
                check_mol.AddConformer(c)
                check = AllChem.MMFFOptimizeMoleculeConfs(check_mol, maxIters=0)
                print(check)
                # Check for uniqueness
                if args.rmsd_remove_Hs:
                    rmsd = rdMolAlign.GetBestRMS(Chem.RemoveHs(conformation_molecules[i]), Chem.RemoveHs(check_mol))
                else:
                    rmsd = rdMolAlign.GetBestRMS(conformation_molecules[i], check_mol)
                print(rmsd)
                if rmsd < args.post_rmsd_threshold:
                    if molecule[1] == "mol1":
                        num_confs_mol1 += 1
                    else:
                        num_confs_mol2 += 1
                    break

    print(f'Number of Conformations Identified by mol1: {num_confs_mol1}')
    print(f'Number of Conformations Identified by mol2: {num_confs_mol2}')
