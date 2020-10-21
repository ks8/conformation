""" RMSD pruning of RDKit conformations. """
import os
import copy
from typing import List
from typing_extensions import Literal

from rdkit import Chem
from rdkit.Chem import AllChem, rdchem, rdMolAlign
# noinspection PyPackageRequirements
from tap import Tap
from tqdm import tqdm


class Args(Tap):
    """
    System arguments.
    """
    data_path: str  # Path to RDKit binary file containing conformations
    minimize: bool = False  # Whether or not to minimize conformations before RMSD pruning
    rmsd_func: Literal["GetBestRMS", "AlignMol"] = "GetBestRMS"  # RMSD computation options
    remove_Hs: bool = False  # Whether or not to do RMSD calculations without Hydrogen atoms
    energy_threshold: float = 2.0  # Energy threshold above which 2 conformers are considered different (kcal/mol)
    rmsd_threshold: float = 0.5  # RMSD threshold for deciding two conformers are the same (Angstroms)
    save_dir: str  # Path to output file containing pruned conformations


# noinspection PyUnresolvedReferences
def compute_rmsd(rmsd_func: Literal["GetBestRMS", "AlignMol"], mol_1: rdchem.Mol, mol_2: rdchem.Mol, index_1: int,
                 index_2: int) -> float:
    """
    Function that returns the RMSD computation function specified in args.
    :param rmsd_func: Which RMSD function to use.
    :param mol_1: First molecule.
    :param mol_2: Second molecule.
    :param index_1: Conformation index of first molecule.
    :param index_2: Conformation index of second molecule.
    :return: RMSD computation function.
    """
    if rmsd_func == "GetBestRMS":
        rmsd = rdMolAlign.GetBestRMS(mol_1, mol_2, index_1, index_2)
    else:
        rmsd = rdMolAlign.AlignMol(mol_1, mol_2, index_1, index_2)

    return rmsd


# noinspection PyUnresolvedReferences,PyPep8Naming
def rmsd_pruning(mol: rdchem.Mol, energies: List, rmsd_func: Literal["GetBestRMS", "AlignMol"],
                 remove_Hs: bool, energy_threshold: float, rmsd_threshold: float) -> rdchem.Mol:
    """
    RMSD pruning of RDKit conformations.
    :param mol: Mol object containing all conformations to be pruned.
    :param energies: List of energies of the conformations.
    :param rmsd_func: Which RMSD function to use.
    :param remove_Hs: Whether or not to remove Hydrogen atoms for RMSD computation.
    :param energy_threshold: Energy threshold for determining if two conformations are different.
    :param rmsd_threshold: RMSD threshold for determining if two conformations are the same.
    :return: Mol object containing pruned conformations.
    """
    print(f'Beginning RMSD pruning...')
    unique_conformer_indices = []

    if remove_Hs:
        # noinspection PyPep8Naming
        mol_no_Hs = Chem.RemoveHs(mol)

    # Loop through conformations to find unique ones
    unique_conformer_indices.append(0)
    for i in tqdm(range(1, mol.GetNumConformers())):
        unique = True
        for j in unique_conformer_indices:
            energy_diff = abs(energies[i] - energies[j])
            if energy_diff < energy_threshold:
                if remove_Hs:
                    # noinspection PyUnboundLocalVariable
                    rmsd = compute_rmsd(rmsd_func, mol_no_Hs, mol_no_Hs, j, i)
                else:
                    rmsd = compute_rmsd(rmsd_func, mol, mol, j, i)
                if rmsd < rmsd_threshold:
                    unique = False
                    break
        if unique:
            unique_conformer_indices.append(i)

    pruned_mol = copy.deepcopy(mol)
    pruned_mol.RemoveAllConformers()
    count = 0
    for i in unique_conformer_indices:
        c = mol.GetConformer(i)
        c.SetId(count)
        pruned_mol.AddConformer(c)
        count += 1

    return pruned_mol


def run_rmsd_pruning(args: Args) -> None:
    """
    Run RMSD pruning of RDKit conformations.
    :return: None.
    """
    os.makedirs(args.save_dir, exist_ok=True)

    print(f'Loading molecule...')
    # noinspection PyUnresolvedReferences
    mol = Chem.Mol(open(args.data_path, "rb").read())

    print(f'Computing energies...')
    energies = []
    if args.minimize:
        max_iters = 200
    else:
        max_iters = 0
    res = AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=max_iters, numThreads=0)
    for i in range(len(res)):
        energies.append(res[i][1])

    # Save unique conformers in molecule object
    print(f'Saving pruned conformations...')

    pruned_mol = rmsd_pruning(mol, energies, args.rmsd_func, args.remove_Hs, args.energy_threshold, args.rmsd_threshold)

    # Save molecule to binary file
    bin_str = pruned_mol.ToBinary()
    with open(os.path.join(args.save_dir, "pruned-conformations.bin"), "wb") as b:
        b.write(bin_str)
