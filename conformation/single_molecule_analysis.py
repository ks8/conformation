""" Analyze properties of a single molecule in RDKit. """
from typing import List

from rdkit import Chem
from rdkit.Chem import AllChem, rdmolfiles, rdMolTransforms
from rdkit.Chem.Draw import rdMolDraw2D
# noinspection PyUnresolvedReferences
from rdkit.Geometry.rdGeometry import Point3D
import seaborn as sns
# noinspection PyPackageRequirements
from tap import Tap


class Args(Tap):
    """
    System arguments.
    """
    smiles: str  # Molecular SMILES string
    save_path: str  # Save path
    atoms_to_rotate: List[int]  # List of atoms defining dihedral angle
    theta: float  # Total angle sweep
    increment: int  # Number of increments of angle to rotate


def single_molecule_analysis(args: Args):
    """
    Analyze properties of a single molecule in RDKit.
    :return: None
    """
    mol = Chem.MolFromSmiles(args.smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol)
    AllChem.MMFFOptimizeMolecule(mol)

    d = rdMolDraw2D.MolDraw2DCairo(500, 500)
    # noinspection PyArgumentList
    d.drawOptions().addStereoAnnotation = True
    # noinspection PyArgumentList
    d.drawOptions().addAtomIndices = True
    rdMolDraw2D.PrepareAndDrawMolecule(d, mol)

    print(rdmolfiles.MolToPDBBlock(mol), file=open(args.save_path + '_original.pdb', "w+"))

    c = mol.GetConformer()
    tmp = Chem.MolFromSmiles(args.smiles)
    tmp = Chem.AddHs(tmp)
    counter = 0
    energies = []
    angles = []
    for i in range(args.increment):
        rdMolTransforms.SetDihedralRad(c, args.atoms_to_rotate[0], args.atoms_to_rotate[1],
                                       args.atoms_to_rotate[2], args.atoms_to_rotate[3],
                                       float(i)*args.theta/float(args.increment))
        c.SetId(counter)
        counter += 1
        tmp.AddConformer(c)

        angles.append(float(i)*args.theta/float(args.increment))

    res = AllChem.MMFFOptimizeMoleculeConfs(tmp, maxIters=0)
    for i in range(len(res)):
        energies.append(res[i][1])
    fig = sns.jointplot(angles, energies).set_axis_labels("Dihedral Angle (rad)", "Energy (kcal/mol)")
    fig.savefig(args.save_path + "energy-vs-angle.png")
    print(rdmolfiles.MolToPDBBlock(tmp), file=open(args.save_path + '_rotated.pdb', "w+"))
