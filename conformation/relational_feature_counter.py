""" Determine which category of features are present in a dataset of molecules (bond types, atom types, etc.) """
import itertools
import json
import numpy as np

from rdkit import Chem
from rdkit.Chem import rdmolops
# noinspection PyPackageRequirements
from tap import Tap


class Args(Tap):
    """
    System arguments.
    """
    data_path: str  # Path to metadata file
    save_path: str  # Path to save file


def relational_feature_counter(args: Args):
    """
    Determine which category of features are present in a dataset of molecules (bond types, atom types, etc.).
    """

    # Load metadata
    metadata = json.load(open(args.data_path))

    # Feature dictionaries
    atom_types = dict()
    bond_types = dict()
    path_lengths = dict()
    max_atoms = 0
    atom_degrees = dict()
    atom_num_hydrogen = dict()
    num_radical_electrons = dict()

    # Process each molecule in the dataset
    for i in range(len(metadata)):
        # Read molecule from SMILES string
        smiles = metadata[i]['smiles']
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        num_atoms = mol.GetNumAtoms()

        # Process atom types and atom degrees
        for atom in mol.GetAtoms():
            atomic_num = atom.GetAtomicNum()
            degree = atom.GetDegree()
            num_hydrogen = atom.GetTotalNumHs(includeNeighbors=True)
            num_radical = atom.GetNumRadicalElectrons()
            if atomic_num in atom_types:
                atom_types[atomic_num] += 1
            else:
                atom_types[atomic_num] = 1
            if degree in atom_degrees:
                atom_degrees[degree] += 1
            else:
                atom_degrees[degree] = 1
            if num_hydrogen in atom_num_hydrogen:
                atom_num_hydrogen[num_hydrogen] += 1
            else:
                atom_num_hydrogen[num_hydrogen] = 1
            if num_radical in num_radical_electrons:
                num_radical_electrons[num_radical] += 1
            else:
                num_radical_electrons[num_radical] = 1

        # Process bond types
        for bond in mol.GetBonds():
            bond_num = bond.GetBondTypeAsDouble()
            if bond_num in bond_types:
                bond_types[bond_num] += 1
            else:
                bond_types[bond_num] = 1

        # Process path length types
        for m, n in itertools.combinations(list(np.arange(num_atoms)), 2):
            path_len = len(rdmolops.GetShortestPath(mol, int(m), int(n))) - 1
            if path_len in path_lengths:
                path_lengths[path_len] += 1
            else:
                path_lengths[path_len] = 1

        # Process number of atoms
        if num_atoms > max_atoms:
            max_atoms = num_atoms

    # Save results to text file
    with open(args.save_path + ".txt", "w") as f:
        f.write("Atom Types: ")
        f.write(json.dumps(atom_types))
        f.write("\n")
        f.write("Bond Types: ")
        f.write(json.dumps(bond_types))
        f.write("\n")
        f.write("Shortest Path Lengths: ")
        f.write(json.dumps(path_lengths))
        f.write("\n")
        f.write("Max number of atoms: ")
        f.write(str(max_atoms))
        f.write("\n")
        f.write("Atom Degrees: ")
        f.write(json.dumps(atom_degrees))
        f.write("\n")
        f.write("Atom Total Num Hs: ")
        f.write(json.dumps(atom_num_hydrogen))
        f.write("\n")
        f.write("Atom Num Radical Electrons: ")
        f.write(json.dumps(num_radical_electrons))
