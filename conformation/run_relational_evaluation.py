""" Run relational network evaluation. """
import collections
import itertools
import json
from logging import Logger
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import pandas as pd
from typing import Dict, List, Tuple
from typing_extensions import Literal

import rdkit
from rdkit import Chem
from rdkit.Chem import rdmolops
import rdkit.Chem.Draw
from rdkit.Chem.Draw import rdMolDraw2D
import seaborn as sns
from sklearn.model_selection import train_test_split
# noinspection PyPackageRequirements
from tap import Tap
import torch
from tqdm import tqdm

from conformation.batch import Batch
from conformation.dataloader import DataLoader
from conformation.dataset import GraphDataset
from conformation.relational_utils import load_relational_checkpoint
from conformation.run_relational_training import Args as RelationalTrainArgs
from conformation.utils import param_count


class Args(Tap):
    """
    System arguments.
    """
    data_path: str  # Path to metadata file
    uid_path: str  # Path to uid dictionary
    binary_dict_path: str  # Path to uid-binary dictionary
    checkpoint_path: str  # Directory of checkpoint to load saved model
    save_dir: str  # Directory for logger
    batch_size: int = 10  # Batch size
    dataset: Literal["train", "val", "test"] = "test"  # Which dataset to evaluate
    cuda: bool = False  # Cuda availability
    distance_analysis: bool = False  # Whether or not to print selective bond/edge information
    distance_analysis_lo: float = 1.0  # Lower distance analysis bound
    distance_analysis_hi: float = 2.0  # Upper distance analysis bound
    error_threshold: float = 0.05  # Decimal percentage corresponding to lower/upper error cutoffs for plotting


def simple_plot(array_x: np.ndarray, array_y: np.ndarray, x_label: str, y_label: str, save_path: str, size: float = 0.5,
                x_lim: Tuple = None, y_lim: Tuple = None, color: str = "b") -> None:
    """
    Plot one array against another.
    :param array_x: Data measured on the x-axis.
    :param array_y: Data measured on the y-axis.
    :param x_label: x-axis label.
    :param y_label: y-axis label.
    :param save_path: Name to save figure as.
    :param size: Marker size.
    :param x_lim: x-axis limits.
    :param y_lim: y-axis limits.
    :param color: Color.
    :return: None.
    """
    sns.set_style("dark")
    fig = sns.jointplot(array_x, array_y, xlim=x_lim, ylim=y_lim, s=size, color=color).set_axis_labels(x_label, y_label)
    fig.savefig(save_path)
    plt.close()


def double_axis_plot(array_x: np.ndarray, array_y_1: np.ndarray, array_y_2: np.ndarray, title: str, x_label: str,
                     y_label_1: str, y_label_2: str, save_path: str, color_1: str = 'tab:red',
                     color_2: str = 'tab:blue', style: str = 'o', size: float = 0.5, x_lim: Tuple = None,
                     y_lim_1: Tuple = None, y_lim_2: Tuple = None) -> None:
    """
    Plot one array against two others using two separate y-axes.
    :param array_x: Data measured on the x-axis.
    :param array_y_1: Data measured on the left y-axis.
    :param array_y_2: Data measured on the right y-axis.
    :param title: Plot title.
    :param x_label: x-axis label.
    :param y_label_1: Left y-axis label.
    :param y_label_2: Right y-axis label.
    :param save_path: Name to save figure as.
    :param color_1: Color of left axis info.
    :param color_2: Color of right axis info.
    :param style: Plot style.
    :param size: Marker size.
    :param x_lim: x-axis limits.
    :param y_lim_1: Left y-axis limits.
    :param y_lim_2: Right y-axis limits.
    :return: None.
    """
    sns.set()

    fig, ax1 = plt.subplots()

    color = color_1
    ax1.set_xlabel(x_label)
    ax1.set_ylabel(y_label_1, color=color)
    ax1.plot(array_x, array_y_1, style, color=color, markersize=size)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(y_lim_1)
    ax1.set_xlim(x_lim)

    ax2 = ax1.twinx()
    color = color_2
    ax2.set_ylabel(y_label_2, color=color)
    ax2.plot(array_x, array_y_2, style, color=color, markersize=size)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(y_lim_2)

    fig.tight_layout()
    plt.title(title)
    plt.savefig(save_path, bbox_inches='tight')
    fig.clf()
    plt.close(fig)


def tensor_to_list(tensor: torch.Tensor, destination_list: List) -> None:
    """
    Transfer data from 1D tensor to list.
    :param tensor: Tensor containing data.
    :param destination_list: List.
    :return: None
    """
    for i in range(tensor.shape[0]):
        destination_list.append(tensor[i])


def path_and_bond_extraction(batch: Batch, uid_dict: Dict, binary_dict: Dict, shortest_paths: List, bond_types: List,
                             triplet_types: List, carbon_carbon_ring_types: List, carbon_carbon_non_ring_types,
                             carbon_carbon_chain_types: List, carbon_carbon_no_chain_types: List,
                             carbon_carbon_aromatic_types: List, carbon_carbon_non_aromatic_types: List,
                             carbon_carbon_quadruplet_types: List, carbon_carbon_non_ring_molecules: List) -> None:
    """
    Compute shortest path, bond type information for each edge and add to relevant lists.
    :param batch: Data batch.
    :param uid_dict: uid-smiles dictionary.
    :param binary_dict: uid-binary dictionary.
    :param shortest_paths: List containing shortest path lengths.
    :param bond_types: List containing bond types (pars of atoms, not actual molecular bond types) for edges.
    :param triplet_types: List containing bond types for "bonds" with shortest path length 2.
    :param carbon_carbon_ring_types: List containing bond types for CC "bonds" in a ring.
    :param carbon_carbon_non_ring_types: List containing bond types for CC "bonds" not in a ring.
    :param carbon_carbon_chain_types: List containing bond types for CC "bonds" in a linear chain.
    :param carbon_carbon_no_chain_types: List containing bond types for CC "bonds" not in a linear chain.
    :param carbon_carbon_aromatic_types: List containing bond types for CC "bonds" in aromatic rings.
    :param carbon_carbon_non_aromatic_types: List containing bond types for CC "bonds" in non-aromatic rings.
    :param carbon_carbon_quadruplet_types: List containing bond types for "bonds" with shortest path length 3.
    :param carbon_carbon_non_ring_molecules: List of molecules corresponding to CC no ring "bonds".
    :return: None.
    """
    for i in range(batch.uid.shape[0]):
        # noinspection PyUnresolvedReferences
        uid = batch.uid[i].item()
        smiles = uid_dict[uid]
        # noinspection PyUnresolvedReferences
        mol = Chem.Mol(open(binary_dict[uid], "rb").read())

        # Acquire the SMILES string with Hs removed corresponding to this molecule. Then, create a new molecule
        # from this string, add Hs, and then use this molecule for drawing. Drawing in this way will generate
        # an automated 2D conformation that has nothing to do with the conformation of the original molecule, but that
        # is easy for viewing. In order to visualize a 2D representation of the actual molecule from the binary,
        # Use the molecule from the binary file directly.
        mol = Chem.AddHs(Chem.MolFromSmiles(Chem.MolToSmiles(Chem.RemoveHs(mol))))
        for m, n in itertools.combinations(list(np.arange(mol.GetNumAtoms())), 2):
            shortest_paths.append(len(rdmolops.GetShortestPath(mol, int(m), int(n))) - 1)
            atom_a = str(mol.GetAtoms()[int(m)].GetSymbol())
            atom_b = str(mol.GetAtoms()[int(n)].GetSymbol())
            path_len = len(rdmolops.GetShortestPath(mol, int(m), int(n))) - 1
            key = ''.join(sorted([atom_a, atom_b])) + str(path_len)
            bond_types.append(key)

            if path_len == 2:
                atom_intermediate = mol.GetAtoms()[rdmolops.GetShortestPath(mol, int(m),
                                                                            int(n))[1]].GetSymbol()
                key = sorted([atom_a, atom_b])[0] + atom_intermediate + sorted([atom_a, atom_b])[1]
                triplet_types.append(key)
            else:
                triplet_types.append(None)

            # Determine if two Carbons are part of the same ring, and if so, what is the ring size
            if atom_a == atom_b == "C":
                ring_info = list(mol.GetRingInfo().AtomRings())
                atom_a_idx = mol.GetAtoms()[int(m)].GetIdx()
                atom_b_idx = mol.GetAtoms()[int(n)].GetIdx()
                membership = [atom_a_idx in r and atom_b_idx in r for r in ring_info]
                if sum(membership) > 0:
                    key = ''.join(sorted([atom_a, atom_b])) + str(path_len) + "-ring"
                    carbon_carbon_ring_types.append(key)
                    carbon_carbon_non_ring_types.append(None)
                    carbon_carbon_non_ring_molecules.append(None)

                    if mol.GetAtoms()[int(m)].GetIsAromatic() and mol.GetAtoms()[int(n)].GetIsAromatic():
                        key = ''.join(sorted([atom_a, atom_b])) + str(path_len) + "-aromatic"
                        carbon_carbon_aromatic_types.append(key)
                        carbon_carbon_non_aromatic_types.append(None)
                    else:
                        key = ''.join(sorted([atom_a, atom_b])) + str(path_len) + "-non-aromatic"
                        carbon_carbon_aromatic_types.append(None)
                        carbon_carbon_non_aromatic_types.append(key)

                else:
                    key = ''.join(sorted([atom_a, atom_b])) + str(path_len) + "-non-ring"
                    carbon_carbon_ring_types.append(None)
                    carbon_carbon_non_ring_types.append(key)
                    carbon_carbon_non_ring_molecules.append([mol, m, n, uid, smiles])
                    carbon_carbon_aromatic_types.append(None)
                    carbon_carbon_non_aromatic_types.append(None)

                path = rdmolops.GetShortestPath(mol, int(m), int(n))
                all_carbon = True
                for j in range(len(path)):
                    atom_type = mol.GetAtoms()[path[j]].GetSymbol()
                    if atom_type != "C":
                        all_carbon = False
                if all_carbon:
                    key = ''.join(sorted([atom_a, atom_b])) + str(path_len) + "-chain"
                    carbon_carbon_chain_types.append(key)
                    carbon_carbon_no_chain_types.append(None)
                else:
                    key = ''.join(sorted([atom_a, atom_b])) + str(path_len) + "-non-chain"
                    carbon_carbon_chain_types.append(None)
                    carbon_carbon_no_chain_types.append(key)

                if path_len == 3:
                    atom_intermediate_1 = mol.GetAtoms()[rdmolops.GetShortestPath(mol, int(m),
                                                                                  int(n))[1]].GetSymbol()
                    atom_intermediate_2 = mol.GetAtoms()[rdmolops.GetShortestPath(mol, int(m),
                                                                                  int(n))[2]].GetSymbol()
                    intermediates = sorted([atom_intermediate_1, atom_intermediate_2])
                    key = atom_a + intermediates[0] + intermediates[1] + atom_b
                    carbon_carbon_quadruplet_types.append(key)

                else:
                    carbon_carbon_quadruplet_types.append(None)

            else:
                carbon_carbon_ring_types.append(None)
                carbon_carbon_non_ring_types.append(None)
                carbon_carbon_non_ring_molecules.append(None)
                carbon_carbon_chain_types.append(None)
                carbon_carbon_no_chain_types.append(None)
                carbon_carbon_aromatic_types.append(None)
                carbon_carbon_non_aromatic_types.append(None)
                carbon_carbon_quadruplet_types.append(None)


def create_bond_dictionary(types: List, dictionary: Dict, *args: np.ndarray) -> None:
    """
    Create a dictionary from a list of keys and values.
    :param types: List of keys.
    :param dictionary: Dictionary object.
    :param args: Lists of values.
    :return: None
    """
    # Fill dictionary
    for i in range(len(types)):
        if types[i] is not None:
            values = []
            for val in args:
                values += [val[i]]
            if types[i] in dictionary:
                dictionary[types[i]].append(values)
            else:
                dictionary[types[i]] = [values]

    # Turn value lists into numpy arrays
    for key in dictionary:
        dictionary[key] = np.array(dictionary[key])


def pair_bar_plots(list_1: List, list_2: List, save_name_1: str, save_name_2: str, args: Args,
                   y_lim: Tuple[float, float] = None) -> None:
    """
    Plot pair of bar plots.
    :param list_1: One list of data.
    :param list_2: Another list of data.
    :param save_name_1: Plot save name 1.
    :param save_name_2: Plot save name 2.
    :param y_lim: Shared y_lim values.
    :param args: System arguments.
    :return: None
    """
    counter = collections.Counter(list_1)
    counter_keys_1 = []
    counter_values_1 = []
    for key, value in counter.items():
        counter_keys_1.append(key)
        counter_values_1.append(value / len(list_1))

    counter = collections.Counter(list_2)
    counter_keys_2 = []
    counter_values_2 = []
    for key, value in counter.items():
        counter_keys_2.append(key)
        counter_values_2.append(value / len(list_2))

    total_keys = set(counter_keys_1 + counter_keys_2)
    for i in total_keys:
        if i not in counter_keys_1:
            counter_keys_1.append(i)
            counter_values_1.append(0.0)
        if i not in counter_keys_2:
            counter_keys_2.append(i)
            counter_values_2.append(0.0)

    counter_keys_sorted = list(np.array(counter_keys_1)[np.argsort(counter_keys_1)])
    counter_values_sorted = list(np.array(counter_values_1)[np.argsort(counter_keys_1)])
    df = pd.DataFrame({"groups": counter_keys_sorted, "percent": counter_values_sorted})

    ax = sns.barplot(x="groups", y="percent", data=df)
    if y_lim is not None:
        ax.set_ylim(y_lim)
    ax.figure.savefig(os.path.join(args.save_dir, save_name_1))
    plt.close()

    counter_keys_sorted = list(np.array(counter_keys_2)[np.argsort(counter_keys_2)])
    counter_values_sorted = list(np.array(counter_values_2)[np.argsort(counter_keys_2)])
    df = pd.DataFrame({"groups": counter_keys_sorted, "percent": counter_values_sorted})

    ax = sns.barplot(x="groups", y="percent", data=df)
    if y_lim is not None:
        ax.set_ylim(y_lim)
    ax.figure.savefig(os.path.join(args.save_dir, save_name_2))
    plt.close()


def run_relational_evaluation(args: Args, logger: Logger) -> None:
    """
    Run evaluation of a relational neural network.
    :param args: System arguments.
    :param logger: Logging.
    :return: None.
    """

    # Set up logger
    debug, info = logger.debug, logger.info

    # Print args
    debug(args)

    # Check cuda availability
    args.cuda = torch.cuda.is_available()

    # Load/build model
    debug('Loading model from {}'.format(args.checkpoint_path))
    model, loaded_args = load_relational_checkpoint(args.checkpoint_path, RelationalTrainArgs())

    # Load metadata
    debug("loading data")
    metadata = json.load(open(args.data_path))
    train_metadata, remaining_metadata = train_test_split(metadata, test_size=0.2, random_state=0)
    validation_metadata, test_metadata = train_test_split(remaining_metadata, test_size=0.5, random_state=0)

    # Create datasets
    train_data = GraphDataset(train_metadata, atom_types=loaded_args.atom_types, bond_types=loaded_args.bond_types,
                              max_path_length=loaded_args.max_shortest_path_length)
    val_data = GraphDataset(validation_metadata, atom_types=loaded_args.atom_types, bond_types=loaded_args.bond_types,
                            max_path_length=loaded_args.max_shortest_path_length)
    test_data = GraphDataset(test_metadata, atom_types=loaded_args.atom_types, bond_types=loaded_args.bond_types,
                             max_path_length=loaded_args.max_shortest_path_length)

    train_data_length, val_data_length, test_data_length = len(train_data), len(val_data), len(test_data)
    debug('train size = {:,} | val size = {:,} | test size = {:,}'.format(
        train_data_length,
        val_data_length,
        test_data_length)
    )

    # Convert to iterator
    train_data = DataLoader(train_data, args.batch_size)
    val_data = DataLoader(val_data, args.batch_size)
    test_data = DataLoader(test_data, args.batch_size)

    # Print model details
    debug(model)
    debug('Number of parameters = {:,}'.format(param_count(model)))

    if args.cuda:
        debug('Moving model to cuda')
        model = model.cuda()

    # Loss function
    loss_func = torch.nn.MSELoss()

    # Loss function used to compute individual errors
    loss_func_aux = torch.nn.MSELoss(reduction='none')

    # Load uid-smiles dictionary
    uid_dict = pickle.load(open(args.uid_path, "rb"))
    binary_dict = pickle.load(open(args.binary_dict_path, "rb"))

    # Select dataset to evaluate
    if args.dataset == "test":
        data = test_data
    elif args.dataset == "train":
        data = train_data
    else:
        data = val_data

    # Extract predictions, errors, and true values in the dataset
    true_distances = []
    shortest_paths = []
    bond_types = []
    triplet_types = []
    carbon_carbon_ring_types = []
    carbon_carbon_non_ring_types = []
    carbon_carbon_non_ring_molecules = []
    carbon_carbon_chain_types = []
    carbon_carbon_no_chain_types = []
    carbon_carbon_aromatic_types = []
    carbon_carbon_non_aromatic_types = []
    carbon_carbon_quadruplet_types = []
    bond_type_dict = dict()
    triplet_dict = dict()
    carbon_carbon_ring_dict = dict()
    carbon_carbon_non_ring_dict = dict()
    carbon_carbon_chain_dict = dict()
    carbon_carbon_no_chain_dict = dict()
    carbon_carbon_aromatic_dict = dict()
    carbon_carbon_non_aromatic_dict = dict()
    carbon_carbon_quadruplet_dict = dict()
    # noinspection PyUnresolvedReferences
    stereo_dict = {rdkit.Chem.rdchem.BondStereo.STEREONONE: "STEREONONE",
                   rdkit.Chem.rdchem.BondStereo.STEREOANY: "STEREOANY",
                   rdkit.Chem.rdchem.BondStereo.STEREOZ: "STEREOZ",
                   rdkit.Chem.rdchem.BondStereo.STEREOE: "STEREOE",
                   rdkit.Chem.rdchem.BondStereo.STEREOCIS: "STEREOCIS",
                   rdkit.Chem.rdchem.BondStereo.STEREOTRANS: "STEREOTRANS"}
    # noinspection PyUnresolvedReferences
    chiral_dict = {rdkit.Chem.rdchem.ChiralType.CHI_UNSPECIFIED: "UNSPECIFIED",
                   rdkit.Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW: "CHI_TETRAHEDRAL_CW",
                   rdkit.Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW: "CHI_TETRAHEDRAL_CCW",
                   rdkit.Chem.rdchem.ChiralType.CHI_OTHER: "CHI_OTHER"}
    with torch.no_grad():
        # Initialize lists and counters for mean + std predictions
        if loaded_args.final_output_size == 2:
            mean_predictions = []
            std_predictions = []
            mean_errors = []
            std_errors = []
            true_means = []
            true_stds = []
            mean_error_sum, std_error_sum, batch_count = 0, 0, 0
        # Initialize lists and counters for just mean predictions
        else:
            errors = []
            predictions = []
            error_sum, batch_count = 0, 0

        model.eval()
        for batch in tqdm(data, total=len(data)):
            # Move batch to cuda
            batch.x = batch.x.cuda()
            batch.edge_attr = batch.edge_attr.cuda()

            # Generate predictions for mean + std
            if loaded_args.final_output_size == 2:
                targets = batch.y.cuda()
            # Generate predictions for mean
            else:
                targets = batch.y.cuda()[:, 0].unsqueeze(1)
            preds = model(batch)

            # Extract prediction, error, and true value information for mean + std
            if loaded_args.final_output_size == 2:
                # Warnings for negative mean or std predictions
                if sum(preds[:, 1] < 0).item() > 0:
                    debug('Negative std prediction detected!')
                if sum(preds[:, 0] < 0).item() > 0:
                    debug('Negative mean prediction detected!')

                # Extract mean + std predictions
                tensor_to_list(preds[:, 0].cpu().numpy(), mean_predictions)
                tensor_to_list(preds[:, 1].cpu().numpy(), std_predictions)

                # Extract mean + std errors
                tensor_to_list(torch.sqrt_(loss_func_aux(preds[:, 0], targets[:, 0])).cpu().numpy(), mean_errors)
                tensor_to_list(torch.sqrt_(loss_func_aux(preds[:, 1], targets[:, 1])).cpu().numpy(), std_errors)

                # Extract mean + std true values
                tensor_to_list(targets[:, 0].cpu().numpy(), true_means)
                tensor_to_list(targets[:, 1].cpu().numpy(), true_stds)

                # Compute shortest path length for each edge and append to relevant lists
                path_and_bond_extraction(batch, uid_dict, binary_dict, shortest_paths, bond_types, triplet_types,
                                         carbon_carbon_ring_types, carbon_carbon_non_ring_types,
                                         carbon_carbon_chain_types, carbon_carbon_no_chain_types,
                                         carbon_carbon_aromatic_types, carbon_carbon_non_aromatic_types,
                                         carbon_carbon_quadruplet_types, carbon_carbon_non_ring_molecules)

                # Compute RMSE for means and stds
                mean_error_sum += torch.sqrt_(loss_func(preds[:, 0], targets[:, 0])).item()
                std_error_sum += torch.sqrt_(loss_func(preds[:, 1], targets[:, 1])).item()
                batch_count += 1

            # Extract prediction, error, and true value information for mean
            else:

                # Extract predictions
                tensor_to_list(preds.cpu().numpy().squeeze(1), predictions)

                # Extract errors
                tensor_to_list(torch.sqrt_(loss_func_aux(preds, targets)).cpu().numpy().squeeze(1), errors)

                # Extract true values
                tensor_to_list(targets.cpu().numpy().squeeze(1), true_distances)

                # Compute shortest path length for each edge and append to relevant lists
                path_and_bond_extraction(batch, uid_dict, binary_dict, shortest_paths, bond_types, triplet_types,
                                         carbon_carbon_ring_types, carbon_carbon_non_ring_types,
                                         carbon_carbon_chain_types, carbon_carbon_no_chain_types,
                                         carbon_carbon_aromatic_types, carbon_carbon_non_aromatic_types,
                                         carbon_carbon_quadruplet_types, carbon_carbon_non_ring_molecules)

                # Compute RMSE loss
                error_sum += torch.sqrt_(loss_func(preds, targets)).item()
                batch_count += 1

    if loaded_args.final_output_size == 2:
        mean_error_avg = mean_error_sum / batch_count
        std_error_avg = std_error_sum / batch_count
        debug(f'Test mean error avg = {mean_error_avg:.4e}')
        debug(f'Test std error avg = {std_error_avg:.4e}')

        # Convert to numpy
        mean_errors = np.array(mean_errors)
        std_errors = np.array(std_errors)
        true_means = np.array(true_means)
        true_stds = np.array(true_stds)
        shortest_paths = np.array(shortest_paths)
        mean_predictions = np.array(mean_predictions)
        std_predictions = np.array(std_predictions)

        create_bond_dictionary(bond_types, bond_type_dict, mean_errors, std_errors, true_means, true_stds,
                               mean_predictions, std_predictions)
        create_bond_dictionary(triplet_types, triplet_dict, mean_errors, std_errors, true_means, true_stds,
                               mean_predictions, std_predictions)
        create_bond_dictionary(carbon_carbon_ring_types, carbon_carbon_ring_dict, mean_errors, std_errors, true_means,
                               true_stds, mean_predictions, std_predictions)
        create_bond_dictionary(carbon_carbon_non_ring_types, carbon_carbon_non_ring_dict, mean_errors, std_errors,
                               true_means, true_stds, mean_predictions, std_predictions)
        create_bond_dictionary(carbon_carbon_chain_types, carbon_carbon_chain_dict, mean_errors, std_errors,
                               true_means, true_stds, mean_predictions, std_predictions)
        create_bond_dictionary(carbon_carbon_no_chain_types, carbon_carbon_no_chain_dict, mean_errors, std_errors,
                               true_means, true_stds, mean_predictions, std_predictions)
        create_bond_dictionary(carbon_carbon_aromatic_types, carbon_carbon_aromatic_dict, mean_errors, std_errors,
                               true_means, true_stds, mean_predictions, std_predictions)
        create_bond_dictionary(carbon_carbon_non_aromatic_types, carbon_carbon_non_aromatic_dict, mean_errors,
                               std_errors, true_means, true_stds, mean_predictions, std_predictions)
        create_bond_dictionary(carbon_carbon_quadruplet_types, carbon_carbon_quadruplet_dict, mean_errors,
                               std_errors, true_means, true_stds, mean_predictions, std_predictions)

        for i in range(len(carbon_carbon_non_ring_molecules)):
            if carbon_carbon_non_ring_molecules[i] is not None:
                carbon_carbon_non_ring_molecules[i] += [mean_errors[i], true_means[i]]

        carbon_carbon_non_ring_molecules = [x for x in carbon_carbon_non_ring_molecules if x is not None]

        lo_chirality = []
        lo_stereo = []
        hi_chirality = []
        hi_stereo = []
        lo_true_chirality = []
        lo_true_stereo = []
        hi_true_chirality = []
        hi_true_stereo = []

        carbon_carbon_non_ring_molecules_path_3_errors = []
        for i in range(len(carbon_carbon_non_ring_molecules)):
            mol = carbon_carbon_non_ring_molecules[i][0]
            atom_a_idx = carbon_carbon_non_ring_molecules[i][1]
            atom_b_idx = carbon_carbon_non_ring_molecules[i][2]
            error = carbon_carbon_non_ring_molecules[i][5]
            if len(rdmolops.GetShortestPath(mol, int(atom_a_idx), int(atom_b_idx))) - 1 == 3:
                carbon_carbon_non_ring_molecules_path_3_errors.append(error)

        carbon_carbon_non_ring_molecules_path_3_errors = sorted(carbon_carbon_non_ring_molecules_path_3_errors)
        bottom_cut = carbon_carbon_non_ring_molecules_path_3_errors[
                     :int(np.ceil(len(carbon_carbon_non_ring_molecules_path_3_errors) / 5.))][-1]
        top_cut = carbon_carbon_non_ring_molecules_path_3_errors[
                  -int(np.ceil(len(carbon_carbon_non_ring_molecules_path_3_errors) / 5.)):][0]

        for i in range(len(carbon_carbon_non_ring_molecules)):
            mol = carbon_carbon_non_ring_molecules[i][0]
            atom_a_idx = carbon_carbon_non_ring_molecules[i][1]
            atom_b_idx = carbon_carbon_non_ring_molecules[i][2]
            uid = carbon_carbon_non_ring_molecules[i][3]
            smiles = carbon_carbon_non_ring_molecules[i][4]
            error = carbon_carbon_non_ring_molecules[i][5]
            true = carbon_carbon_non_ring_molecules[i][6]

            d = rdMolDraw2D.MolDraw2DCairo(500, 500)
            # noinspection PyArgumentList
            d.drawOptions().addStereoAnnotation = True
            # noinspection PyArgumentList
            d.drawOptions().addAtomIndices = True
            rdMolDraw2D.PrepareAndDrawMolecule(d, mol, highlightAtoms=[int(atom_a_idx), int(atom_b_idx)], legend=smiles)
            if len(rdmolops.GetShortestPath(mol, int(atom_a_idx), int(atom_b_idx))) - 1 == 3:
                if error < bottom_cut or error > top_cut:
                    if error < bottom_cut:
                        title_prefix = "lo_error"
                        lo_chirality.append(chiral_dict[mol.GetAtomWithIdx(int(atom_a_idx)).GetChiralTag()])
                        lo_chirality.append(chiral_dict[mol.GetAtomWithIdx(int(atom_b_idx)).GetChiralTag()])
                        for bond in mol.GetBonds():
                            if int(atom_a_idx) or int(atom_b_idx) in bond.GetStereoAtoms():
                                lo_stereo.append(stereo_dict[bond.GetStereo()])
                    else:
                        title_prefix = "hi_error"
                        hi_chirality.append(chiral_dict[mol.GetAtomWithIdx(int(atom_a_idx)).GetChiralTag()])
                        hi_chirality.append(chiral_dict[mol.GetAtomWithIdx(int(atom_b_idx)).GetChiralTag()])
                        for bond in mol.GetBonds():
                            if int(atom_a_idx) or int(atom_b_idx) in bond.GetStereoAtoms():
                                hi_stereo.append(stereo_dict[bond.GetStereo()])
                    with open(os.path.join(args.save_dir, str(uid) + "-" + title_prefix + '.png'), 'wb') as f:
                        # noinspection PyArgumentList
                        f.write(d.GetDrawingText())
                if error > top_cut:
                    if 3. < true < 3.1 or 3.9 < true < 4.0:
                        if 3. < true < 3.1:
                            title_prefix = "lo_true"
                            lo_true_chirality.append(chiral_dict[mol.GetAtomWithIdx(int(atom_a_idx)).GetChiralTag()])
                            lo_true_chirality.append(chiral_dict[mol.GetAtomWithIdx(int(atom_b_idx)).GetChiralTag()])
                            for bond in mol.GetBonds():
                                if int(atom_a_idx) or int(atom_b_idx) in bond.GetStereoAtoms():
                                    lo_true_stereo.append(stereo_dict[bond.GetStereo()])
                        else:
                            title_prefix = "hi_true"
                            hi_true_chirality.append(chiral_dict[mol.GetAtomWithIdx(int(atom_a_idx)).GetChiralTag()])
                            hi_true_chirality.append(chiral_dict[mol.GetAtomWithIdx(int(atom_b_idx)).GetChiralTag()])
                            for bond in mol.GetBonds():
                                if int(atom_a_idx) or int(atom_b_idx) in bond.GetStereoAtoms():
                                    hi_true_stereo.append(stereo_dict[bond.GetStereo()])
                        with open(os.path.join(args.save_dir, str(uid) + "-" + title_prefix + '.png'), 'wb') as f:
                            # noinspection PyArgumentList
                            f.write(d.GetDrawingText())

        # Plotting
        dictionaries_list = [bond_type_dict, triplet_dict, carbon_carbon_ring_dict, carbon_carbon_non_ring_dict,
                             carbon_carbon_chain_dict, carbon_carbon_no_chain_dict, carbon_carbon_aromatic_dict,
                             carbon_carbon_non_aromatic_dict, carbon_carbon_quadruplet_dict]
        true_list = [true_means, true_stds]
        errors_list = [mean_errors, std_errors]
        predictions_list = [mean_predictions, std_predictions]
        string_list = ["mean", "std"]
        x_lim_list = [(0.0, 10.0), (0.0, 1.7)]
        y_lim_1_list = [(0.0, 4.0), (0.0, 1.7)]
        y_lim_2_list = [(0.0, 10.0), (0.0, 1.7)]

        for i in range(2):
            simple_plot(array_x=true_list[i], array_y=errors_list[i], x_label="True", y_label="Error",
                        save_path=os.path.join(args.save_dir, string_list[i] + "-error-vs-distance"))

            simple_plot(array_x=np.log(true_list[i]), array_y=np.log(errors_list[i]), x_label="Log True",
                        y_label="Log Error",
                        save_path=os.path.join(args.save_dir, string_list[i] + "-log-error-vs-log-distance"))

            simple_plot(array_x=shortest_paths, array_y=errors_list[i], x_label="Shortest Path Length",
                        y_label="Error",
                        save_path=os.path.join(args.save_dir, string_list[i] + "-error-vs-path-length"),
                        y_lim=(0.0, 4.0))

            simple_plot(array_x=true_list[i], array_y=shortest_paths, x_label="True Distance",
                        y_label="Shortest Path Length",
                        save_path=os.path.join(args.save_dir, string_list[i] + "-path-length-vs-distance"))

            simple_plot(array_x=true_list[i], array_y=predictions_list[i], x_label="True Distance",
                        y_label="Prediction",
                        save_path=os.path.join(args.save_dir, string_list[i] + "-prediction-vs-distance"))

            # Plot error vs true distance for each bond type
            for j in range(len(dictionaries_list)):
                for key in dictionaries_list[j]:
                    double_axis_plot(array_x=dictionaries_list[j][key][:, 2 + i],
                                     array_y_1=dictionaries_list[j][key][:, 0 + i],
                                     array_y_2=dictionaries_list[j][key][:, 4 + i],
                                     title="Error and Predicted vs True Distance: " + key + " " + string_list[i],
                                     x_label="True Distance", y_label_1="Error", y_label_2="Predicted",
                                     save_path=os.path.join(args.save_dir, key + "-" + string_list[i] +
                                                            "-error-vs-distance-joint"), x_lim=x_lim_list[i],
                                     y_lim_1=y_lim_1_list[i], y_lim_2=y_lim_2_list[i])

                    simple_plot(array_x=dictionaries_list[j][key][:, 2 + i],
                                array_y=dictionaries_list[j][key][:, 0 + i],
                                x_label="True Distance", y_label="Error",
                                save_path=os.path.join(args.save_dir, key + "-" + string_list[i] +
                                                       "-error-vs-distance"),
                                x_lim=x_lim_list[i], y_lim=y_lim_1_list[i], color='r')

                    simple_plot(array_x=dictionaries_list[j][key][:, 2 + i],
                                array_y=dictionaries_list[j][key][:, 4 + i],
                                x_label="True Distance", y_label="Predicted",
                                save_path=os.path.join(args.save_dir, key + "-" + string_list[i] +
                                                       "-prediction-vs-distance"),
                                x_lim=x_lim_list[i], y_lim=y_lim_2_list[i])

    else:
        loss_avg = error_sum / batch_count
        debug("Test error avg = {:.4e}".format(loss_avg))

        # Convert to numpy
        errors = np.array(errors)
        true_distances = np.array(true_distances)
        shortest_paths = np.array(shortest_paths)
        predictions = np.array(predictions)

        # Create dictionaries
        create_bond_dictionary(bond_types, bond_type_dict, errors, true_distances, predictions)
        create_bond_dictionary(triplet_types, triplet_dict, errors, true_distances, predictions)
        create_bond_dictionary(carbon_carbon_ring_types, carbon_carbon_ring_dict, errors, true_distances, predictions)
        create_bond_dictionary(carbon_carbon_non_ring_types, carbon_carbon_non_ring_dict, errors, true_distances,
                               predictions)
        create_bond_dictionary(carbon_carbon_chain_types, carbon_carbon_chain_dict, errors, true_distances,
                               predictions)
        create_bond_dictionary(carbon_carbon_no_chain_types, carbon_carbon_no_chain_dict, errors, true_distances,
                               predictions)
        create_bond_dictionary(carbon_carbon_aromatic_types, carbon_carbon_aromatic_dict, errors, true_distances,
                               predictions)
        create_bond_dictionary(carbon_carbon_non_aromatic_types, carbon_carbon_non_aromatic_dict, errors,
                               true_distances, predictions)
        create_bond_dictionary(carbon_carbon_quadruplet_types, carbon_carbon_quadruplet_dict, errors,
                               true_distances, predictions)

        for i in range(len(carbon_carbon_non_ring_molecules)):
            if carbon_carbon_non_ring_molecules[i] is not None:
                carbon_carbon_non_ring_molecules[i] += [errors[i], true_distances[i]]

        carbon_carbon_non_ring_molecules = [x for x in carbon_carbon_non_ring_molecules if x is not None]

        lo_chirality = []
        lo_stereo = []
        hi_chirality = []
        hi_stereo = []
        lo_true_chirality = []
        lo_true_stereo = []
        hi_true_chirality = []
        hi_true_stereo = []

        carbon_carbon_non_ring_molecules_path_3_errors = []
        for i in range(len(carbon_carbon_non_ring_molecules)):
            mol = carbon_carbon_non_ring_molecules[i][0]
            atom_a_idx = carbon_carbon_non_ring_molecules[i][1]
            atom_b_idx = carbon_carbon_non_ring_molecules[i][2]
            error = carbon_carbon_non_ring_molecules[i][5]
            if len(rdmolops.GetShortestPath(mol, int(atom_a_idx), int(atom_b_idx))) - 1 == 3:
                carbon_carbon_non_ring_molecules_path_3_errors.append(error)

        carbon_carbon_non_ring_molecules_path_3_errors = sorted(carbon_carbon_non_ring_molecules_path_3_errors)
        bottom_cut = carbon_carbon_non_ring_molecules_path_3_errors[
                     :int(np.ceil(len(carbon_carbon_non_ring_molecules_path_3_errors)/5.))][-1]
        top_cut = carbon_carbon_non_ring_molecules_path_3_errors[
                     -int(np.ceil(len(carbon_carbon_non_ring_molecules_path_3_errors)/5.)):][0]

        for i in range(len(carbon_carbon_non_ring_molecules)):
            mol = carbon_carbon_non_ring_molecules[i][0]
            atom_a_idx = carbon_carbon_non_ring_molecules[i][1]
            atom_b_idx = carbon_carbon_non_ring_molecules[i][2]
            uid = carbon_carbon_non_ring_molecules[i][3]
            smiles = carbon_carbon_non_ring_molecules[i][4]
            error = carbon_carbon_non_ring_molecules[i][5]
            true = carbon_carbon_non_ring_molecules[i][6]

            d = rdMolDraw2D.MolDraw2DCairo(500, 500)
            # noinspection PyArgumentList
            d.drawOptions().addStereoAnnotation = True
            # noinspection PyArgumentList
            d.drawOptions().addAtomIndices = True
            rdMolDraw2D.PrepareAndDrawMolecule(d, mol, highlightAtoms=[int(atom_a_idx), int(atom_b_idx)], legend=smiles)
            if len(rdmolops.GetShortestPath(mol, int(atom_a_idx), int(atom_b_idx))) - 1 == 3:
                if error < bottom_cut or error > top_cut:
                    if error < bottom_cut:
                        title_prefix = "lo_error"
                        lo_chirality.append(chiral_dict[mol.GetAtomWithIdx(int(atom_a_idx)).GetChiralTag()])
                        lo_chirality.append(chiral_dict[mol.GetAtomWithIdx(int(atom_b_idx)).GetChiralTag()])
                        for bond in mol.GetBonds():
                            if int(atom_a_idx) or int(atom_b_idx) in bond.GetStereoAtoms():
                                lo_stereo.append(stereo_dict[bond.GetStereo()])
                    else:
                        title_prefix = "hi_error"
                        hi_chirality.append(chiral_dict[mol.GetAtomWithIdx(int(atom_a_idx)).GetChiralTag()])
                        hi_chirality.append(chiral_dict[mol.GetAtomWithIdx(int(atom_b_idx)).GetChiralTag()])
                        for bond in mol.GetBonds():
                            if int(atom_a_idx) or int(atom_b_idx) in bond.GetStereoAtoms():
                                hi_stereo.append(stereo_dict[bond.GetStereo()])
                    with open(os.path.join(args.save_dir, str(uid) + "-" + title_prefix + '.png'), 'wb') as f:
                        # noinspection PyArgumentList
                        f.write(d.GetDrawingText())
                if error > top_cut:
                    if 3. < true < 3.1 or 3.9 < true < 4.0:
                        if 3. < true < 3.1:
                            title_prefix = "lo_true"
                            lo_true_chirality.append(chiral_dict[mol.GetAtomWithIdx(int(atom_a_idx)).GetChiralTag()])
                            lo_true_chirality.append(chiral_dict[mol.GetAtomWithIdx(int(atom_b_idx)).GetChiralTag()])
                            for bond in mol.GetBonds():
                                if int(atom_a_idx) or int(atom_b_idx) in bond.GetStereoAtoms():
                                    lo_true_stereo.append(stereo_dict[bond.GetStereo()])
                        else:
                            title_prefix = "hi_true"
                            hi_true_chirality.append(chiral_dict[mol.GetAtomWithIdx(int(atom_a_idx)).GetChiralTag()])
                            hi_true_chirality.append(chiral_dict[mol.GetAtomWithIdx(int(atom_b_idx)).GetChiralTag()])
                            for bond in mol.GetBonds():
                                if int(atom_a_idx) or int(atom_b_idx) in bond.GetStereoAtoms():
                                    hi_true_stereo.append(stereo_dict[bond.GetStereo()])
                        with open(os.path.join(args.save_dir, str(uid) + "-" + title_prefix + '.png'), 'wb') as f:
                            # noinspection PyArgumentList
                            f.write(d.GetDrawingText())

        pair_bar_plots(list_1=lo_chirality, list_2=hi_chirality, save_name_1="lo_chirality",
                       save_name_2="hi_chirality", args=args, y_lim=(0.0, 1.0))

        pair_bar_plots(list_1=lo_stereo, list_2=hi_stereo, save_name_1="lo_stereo",
                       save_name_2="hi_stereo", args=args, y_lim=(0.0, 1.0))

        pair_bar_plots(list_1=lo_true_chirality, list_2=hi_true_chirality, save_name_1="lo_true_chirality",
                       save_name_2="hi_true_chirality", args=args, y_lim=(0.0, 1.0))

        pair_bar_plots(list_1=lo_true_stereo, list_2=hi_true_stereo, save_name_1="lo_true_stereo",
                       save_name_2="hi_true_stereo", args=args, y_lim=(0.0, 1.0))

        dictionaries_list = [bond_type_dict, triplet_dict, carbon_carbon_ring_dict, carbon_carbon_non_ring_dict,
                             carbon_carbon_chain_dict, carbon_carbon_no_chain_dict, carbon_carbon_aromatic_dict,
                             carbon_carbon_non_aromatic_dict, carbon_carbon_quadruplet_dict]

        # Plotting
        simple_plot(array_x=true_distances, array_y=errors, x_label="True", y_label="Error",
                    save_path=os.path.join(args.save_dir, "error-vs-distance"),
                    y_lim=(0.0, 4.0))

        simple_plot(array_x=np.log(true_distances), array_y=np.log(errors), x_label="Log True", y_label="Log Error",
                    save_path=os.path.join(args.save_dir, "log-error-vs-log-distance"), x_lim=(0.0, 2.5),
                    y_lim=(-15.0, 2.0))

        simple_plot(array_x=shortest_paths, array_y=errors, x_label="Shortest Path Length", y_label="Error",
                    save_path=os.path.join(args.save_dir, "error-vs-path-length"), y_lim=(0.0, 4.0))

        simple_plot(array_x=true_distances, array_y=shortest_paths, x_label="True Distance",
                    y_label="Shortest Path Length",
                    save_path=os.path.join(args.save_dir, "path-length-vs-distance"))

        simple_plot(array_x=true_distances, array_y=predictions, x_label="True Distance", y_label="Prediction",
                    save_path=os.path.join(args.save_dir, "prediction-vs-distance"))

        # Plot error vs true distance for each bond type
        for j in range(len(dictionaries_list)):
            for key in dictionaries_list[j]:
                double_axis_plot(array_x=dictionaries_list[j][key][:, 1], array_y_1=dictionaries_list[j][key][:, 0],
                                 array_y_2=dictionaries_list[j][key][:, 2],
                                 title="Error and Predicted vs True Distance: " + key, x_label="True Distance",
                                 y_label_1="Error", y_label_2="Predicted",
                                 save_path=os.path.join(args.save_dir, key + "-error-vs-distance-joint"),
                                 x_lim=(0.0, 10.0), y_lim_1=(0.0, 4.0), y_lim_2=(0.0, 10.0))

                simple_plot(array_x=dictionaries_list[j][key][:, 1], array_y=dictionaries_list[j][key][:, 0],
                            x_label="True Distance",
                            y_label="Error", save_path=os.path.join(args.save_dir, key + "-error-vs-distance"),
                            x_lim=(0.0, 10.0), y_lim=(0.0, 4.0), color="r")

                simple_plot(array_x=dictionaries_list[j][key][:, 1], array_y=dictionaries_list[j][key][:, 2],
                            x_label="True Distance",
                            y_label="Predicted", save_path=os.path.join(args.save_dir, key + "-prediction-vs-distance"),
                            x_lim=(0.0, 10.0), y_lim=(0.0, 10.0))

        # Save errors as numpy array
        np.save(os.path.join(args.save_dir, "errors"), errors)
