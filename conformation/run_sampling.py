""" Run neural network sampling. """
import itertools
import numpy as np
import os

from rdkit import Chem
from rdkit.Chem import AllChem, rdMolTransforms
# noinspection PyPackageRequirements
from tap import Tap
import torch
from typing import List

from conformation.utils import load_checkpoint, param_count


class Args(Tap):
    """
    System arguments.
    """
    checkpoint_path: str  # Path to saved model checkpoint file
    conditional: bool = False  # Whether or not to use a conditional normalizing flow
    condition_path: str = None  # Path to condition numpy file for conditional normalizing flow
    smiles: str = None  # Molecular SMILES string
    num_atoms: int = 8  # Number of atoms  # TODO: might not be necessary
    num_layers: int = 10  # Number of RealNVP layers
    num_samples: int = 1000  # Number of samples to attempt
    save_dir: str = None  # Save directory
    offset: float = 0.0005  # Distance bounds matrix offset
    dihedral: bool = False  # Use when computing dihedral angle values
    dihedral_vals: List[int] = [2, 0, 1, 5]  # Atom IDs for dihedral
    cuda: bool = False  # Whether or not to use cuda
    gpu_device: int = 0  # Which GPU to use (0 or 1)
    random_coords: bool = True  # Whether or not to use random coordinates for EDG algorithm


def run_sampling(args: Args) -> None:
    """
    Perform neural network training.
    :param args: System parameters.
    :return: None.
    """

    os.makedirs(args.save_dir)
    os.makedirs(os.path.join(args.save_dir, "distmat"))
    os.makedirs(os.path.join(args.save_dir, "properties"))

    args.cuda = torch.cuda.is_available()

    print(args)

    # Load model
    if args.checkpoint_path is not None:
        print('Loading model from {}'.format(args.checkpoint_path))
        model = load_checkpoint(args.checkpoint_path, args.cuda, args.gpu_device)

        print(model)
        print('Number of parameters = {:,}'.format(param_count(model)))

        if args.cuda:
            with torch.cuda.device(args.gpu_device):
                print('Moving model to cuda')
                model = model.cuda()
            device = torch.device(args.gpu_device)
        else:
            device = torch.device('cpu')

        # Conformation counter
        counter = 0

        with torch.no_grad():
            model.eval()
            num_atoms = args.num_atoms

            # Create molecule from SMILES
            mol = Chem.MolFromSmiles(args.smiles)
            mol = Chem.AddHs(mol)
            ps = AllChem.ETKDG()

            # Create a random conformation object
            tmp = Chem.MolFromSmiles(args.smiles)
            tmp = Chem.AddHs(tmp)

            for j in range(args.num_samples):
                if args.conditional:
                    gen_sample = model.sample(args.num_layers, args.condition_path, device)
                else:
                    gen_sample = model.sample(args.num_layers)
                distmat = np.zeros([num_atoms, num_atoms])
                boundsmat = np.zeros([num_atoms, num_atoms])
                indices = []
                for m, n in itertools.combinations(np.arange(num_atoms), 2):
                    indices.append((m, n))
                for i in range(len(indices)):
                    distmat[indices[i][0], indices[i][1]] = gen_sample[i].item()
                    distmat[indices[i][1], indices[i][0]] = distmat[indices[i][0], indices[i][1]]

                    boundsmat[indices[i][0], indices[i][1]] = distmat[indices[i][0], indices[i][1]] + args.offset
                    boundsmat[indices[i][1], indices[i][0]] = distmat[indices[i][1], indices[i][0]] - args.offset

                # Set the bounds matrix
                ps.SetBoundsMat(boundsmat)
                ps.useRandomCoords = args.random_coords

                # Create a conformation from the distance bounds matrix
                # noinspection PyUnusedLocal
                AllChem.EmbedMolecule(tmp, params=ps)

                try:
                    # Test that the conformation is valid
                    c = tmp.GetConformer()

                    # Set the conformer Id and increment the conformation counter
                    c.SetId(counter)
                    counter += 1

                    # Add the conformer to the overall molecule object
                    mol.AddConformer(c)

                    # noinspection PyTypeChecker
                    np.save(os.path.join(args.save_dir, "distmat", "distmat-" + str(counter)), distmat)

                    # Compute properties of the conformation
                    res = AllChem.MMFFOptimizeMoleculeConfs(tmp, maxIters=0)
                    with open(os.path.join(args.save_dir, "properties", "energy-rms-dihedral-" + str(counter) + ".txt"),
                              "w") \
                            as o:
                        o.write("energy: " + str(res[0][1]))
                        o.write('\n')
                        o.write("rms: " + "nan")
                        o.write('\n')
                        if args.dihedral:
                            dihedral_val = rdMolTransforms.GetDihedralRad(c, args.dihedral_vals[0],
                                                                          args.dihedral_vals[1],
                                                                          args.dihedral_vals[2], args.dihedral_vals[3])
                        else:
                            dihedral_val = "nan"
                        o.write("dihedral: " + str(dihedral_val))

                except ValueError:
                    continue
                except AttributeError:
                    continue

            bin_str = mol.ToBinary()
            with open(os.path.join(args.save_dir, "conformations.bin"), "wb") as f:
                f.write(bin_str)

    else:
        print('Must specify a model to load.')
        exit()
