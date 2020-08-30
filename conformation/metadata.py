""" Generate metadata. """
import json
import pickle
import os
import re
from typing import List

import numpy as np
from rdkit import Chem
from scipy import sparse
# noinspection PyPackageRequirements
from tap import Tap
import torch

from conformation.graph_data import Data
from conformation.relational import RelationalNetwork
from conformation.run_relational_training import Args as graph_Args


class Args(Tap):
    """
    System arguments.
    """
    data_dir: str  # Path to directory containing data
    save_dir: str  # Path to directory containing output file
    mpnn: bool = False  # Whether or not to produce metadata for graph neural network training
    autoencoder: bool = False  # Whether or not to produce metadata for graph neural network autoencoder training
    cnf: bool = False  # Whether or not to produce metadata for conditional normalizing flow training
    graph_model_path: str = None  # Path to saved graph model (cnf = True)
    smiles_dir: str = None  # Path to directory containing smiles strings (mpnn or cnf = True)
    binary_dir: str = None  # Path to directory containing RDKit mol binary files (mpnn = True)
    atom_types: List[int] = [1, 6, 7, 8, 9]  # Graph neural net allowed atom types (cnf = True)
    bond_types: List[float] = [0., 1., 1.5, 2., 3.]  # Graph neural net allowed bond types (cnf = True)


def metadata(args: Args) -> None:
    """
    Create metadata folder and file.
    :param args: Folder name info.
    :return None.
    """
    os.makedirs(args.save_dir)
    if args.cnf:
        # Load trained relational network
        state = torch.load(args.graph_model_path, map_location=lambda storage, loc: storage)
        loaded_args = graph_Args().from_dict(state['args'])
        loaded_state_dict = state['state_dict']

        graph_model = RelationalNetwork(loaded_args.hidden_size, loaded_args.num_layers, loaded_args.num_edge_features,
                                        loaded_args.num_vertex_features, loaded_args.final_linear_size,
                                        loaded_args.final_output_size, cnf=True)
        graph_model.load_state_dict(loaded_state_dict)

        if torch.cuda.is_available():
            graph_model = graph_model.cuda()

        graph_model.eval()

        os.makedirs(os.path.join(args.save_dir, "conditions"))

    data = []
    conditional_dict = dict()
    uid_dict = dict()
    binary_dict = dict()
    uid = 0
    for root, _, files in os.walk(args.data_dir):
        for f in files:
            path = os.path.join(root, f)
            if args.mpnn:
                if args.autoencoder:
                    dashes = [m.start() for m in re.finditer("-", f)]
                    molecule_name = f[dashes[1] + 1:dashes[2]]
                else:
                    molecule_name = f[[m.start() for m in re.finditer("-", f)][1] + 1:f.find(".")]
                with open(os.path.join(args.smiles_dir, molecule_name + ".smiles")) as tmp:
                    smiles = tmp.readlines()[0].split()[0]
                binary_path = os.path.join(args.binary_dir, molecule_name + ".bin")
                data.append({'smiles': smiles, 'target': path, 'uid': uid, 'binary': binary_path})
                uid_dict[uid] = smiles
                binary_dict[uid] = binary_path
                uid += 1
            elif args.cnf:
                molecule_name = f[[m.end() for m in re.finditer("-", f)][-1]:f.find(".")]
                with open(os.path.join(args.smiles_dir, molecule_name + ".smiles")) as tmp:
                    smiles = tmp.readlines()[0].split()[0]
                if molecule_name not in conditional_dict:
                    sample = Data()  # Create data object

                    # Molecule from SMILES string
                    mol = Chem.MolFromSmiles(smiles)
                    mol = Chem.AddHs(mol)
                    num_atoms = mol.GetNumAtoms()

                    # Compute edge connectivity in COO format corresponding to a complete graph on num_nodes
                    complete_graph = np.ones([num_atoms, num_atoms])  # Create an auxiliary complete graph
                    complete_graph = np.triu(complete_graph,
                                             k=1)  # Compute an upper triangular matrix of the complete graph
                    complete_graph = sparse.csc_matrix(
                        complete_graph)  # Compute a csc style sparse matrix from this graph
                    row, col = complete_graph.nonzero()  # Extract row and column indices of non-zero entries
                    row = torch.tensor(row, dtype=torch.long)
                    col = torch.tensor(col, dtype=torch.long)
                    sample.edge_index = torch.stack([row, col])  # Edge connectivity in COO format (all possible edges)

                    # Edge features
                    # Create one-hot encoding
                    one_hot_bond_features = np.zeros((len(args.bond_types), len(args.bond_types)))
                    np.fill_diagonal(one_hot_bond_features, 1.)
                    bond_to_one_hot = dict()
                    for i in range(len(args.bond_types)):
                        bond_to_one_hot[args.bond_types[i]] = one_hot_bond_features[i]

                    # Extract atom indices participating in bonds and bond types
                    bonds = []
                    bond_types = []
                    for bond in mol.GetBonds():
                        bonds.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
                        bond_types.append([bond_to_one_hot[bond.GetBondTypeAsDouble()]])

                    # Compute edge attributes: 1 indicates presence of bond, 0 no bond.
                    full_edges = [list(sample.edge_index[:, i].numpy()) for i in range(sample.edge_index.shape[1])]
                    no_bond = np.concatenate([np.array([0]), bond_to_one_hot[0]])
                    a = np.array([1])
                    edge_attr = [
                        np.concatenate([a, bond_types[bonds.index(full_edges[i])][0]]) if full_edges[i] in bonds else
                        no_bond for i in range(len(full_edges))]
                    sample.edge_attr = torch.tensor(edge_attr, dtype=torch.float)

                    # TODO: add additional shortest path length bond features

                    # Vertex features: one-hot representation of atomic number
                    # Create one-hot encoding
                    one_hot_vertex_features = np.zeros((len(args.atom_types), len(args.atom_types)))
                    np.fill_diagonal(one_hot_vertex_features, 1.)
                    atom_to_one_hot = dict()
                    for i in range(len(args.atom_types)):
                        atom_to_one_hot[args.atom_types[i]] = one_hot_vertex_features[i]

                    one_hot_features = np.array([atom_to_one_hot[atom.GetAtomicNum()] for atom in mol.GetAtoms()])
                    sample.x = torch.tensor(one_hot_features, dtype=torch.float)

                    with torch.no_grad():
                        if torch.cuda.is_available():
                            sample.x = sample.x.cuda()
                            sample.edge_attr = sample.edge_attr.cuda()
                        # noinspection PyUnboundLocalVariable
                        embedding = graph_model(sample).cpu().numpy()
                    np.save(os.path.join(args.save_dir, "conditions", molecule_name), embedding)

                    conditional_dict[molecule_name] = os.path.join(args.save_dir, "conditions", molecule_name + ".npy")

                data.append({'smiles': smiles, 'path': path, 'condition': conditional_dict[molecule_name]})
            else:
                data.append({'path': path})
    if len(uid_dict) > 0:
        pickle.dump(uid_dict, open(os.path.join(args.save_dir, "uid_dict.p"), "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    if len(binary_dict) > 0:
        pickle.dump(binary_dict, open(os.path.join(args.save_dir, "binary_dict.p"), "wb"),
                    protocol=pickle.HIGHEST_PROTOCOL)
    json.dump(data, open(os.path.join(args.save_dir, args.save_dir + ".json"), "w"), indent=4, sort_keys=True)
