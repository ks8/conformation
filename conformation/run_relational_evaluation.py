""" Run relational network evaluation. """
import itertools
import json
from logging import Logger
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from typing import Dict, List, Tuple
from typing_extensions import Literal

from rdkit import Chem
from rdkit.Chem import rdmolops
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
    checkpoint_path: str  # Directory of checkpoint to load saved model
    save_dir: str  # Directory for logger
    batch_size: int = 10  # Batch size
    dataset: Literal["train", "val", "test"] = "test"  # Which dataset to evaluate
    cuda: bool = False  # Cuda availability
    distance_analysis: bool = False  # Whether or not to print selective bond/edge information
    distance_analysis_lo: float = 1.0  # Lower distance analysis bound
    distance_analysis_hi: float = 2.0  # Upper distance analysis bound


def simple_plot(array_x: np.ndarray, array_y: np.ndarray, title: str, x_label: str, y_label: str, save_path: str,
                style: str = 'bo', size: float = 0.5, x_lim: Tuple = None, y_lim: Tuple = None) -> None:
    """
    Plot one array against another.
    :param array_x: Data measured on the x-axis.
    :param array_y: Data measured on the y-axis.
    :param title: Plot title.
    :param x_label: x-axis label.
    :param y_label: y-axis label.
    :param save_path: Name to save figure as.
    :param style: Plot style.
    :param size: Marker size.
    :param x_lim: x-axis limits.
    :param y_lim: y-axis limits.
    :return: None.
    """
    plt.plot(array_x, array_y, style, markersize=size)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if x_lim is not None:
        plt.xlim(x_lim)
    if y_lim is not None:
        plt.ylim(y_lim)
    plt.savefig(save_path)
    plt.clf()


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


def path_and_bond_extraction(batch: Batch, uid_dict: Dict, shortest_paths: List, bond_types: List,
                             triplet_types: List) -> None:
    """
    Compute shortest path, bond type information for each edge and add to relevant lists.
    :param batch: Data batch.
    :param uid_dict: uid-smiles dictionary.
    :param shortest_paths: List containing shortest path lengths.
    :param bond_types: List containing bond types (pars of atoms, not actual molecular bond types) for edges.
    :param triplet_types: List containing bond types for "bonds" with shortest path length 2.
    :return: None.
    """
    for i in range(batch.uid.shape[0]):
        # noinspection PyUnresolvedReferences
        uid = batch.uid[i].item()
        smiles = uid_dict[uid]
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
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
    bond_type_dict = dict()
    triplet_dict = dict()
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
                path_and_bond_extraction(batch, uid_dict, shortest_paths, bond_types, triplet_types)

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
                path_and_bond_extraction(batch, uid_dict, shortest_paths, bond_types, triplet_types)

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

        # Load items into bond type dictionary
        for i in range(len(bond_types)):
            if bond_types[i] in bond_type_dict:
                bond_type_dict[bond_types[i]].append([mean_errors[i], std_errors[i], true_means[i], true_stds[i],
                                                      mean_predictions[i], std_predictions[i]])
            else:
                bond_type_dict[bond_types[i]] = [[mean_errors[i], std_errors[i], true_means[i], true_stds[i],
                                                  mean_predictions[i], std_predictions[i]]]

        for i in range(len(triplet_types)):
            if triplet_types[i] is not None:
                if triplet_types[i] in triplet_dict:
                    triplet_dict[triplet_types[i]].append([mean_errors[i], std_errors[i], true_means[i],
                                                           true_stds[i], mean_predictions[i], std_predictions[i]])
                else:
                    triplet_dict[triplet_types[i]] = [[mean_errors[i], std_errors[i], true_means[i], true_stds[i],
                                                       mean_predictions[i], std_predictions[i]]]

        for key in bond_type_dict:
            bond_type_dict[key] = np.array(bond_type_dict[key])

        for key in triplet_dict:
            triplet_dict[key] = np.array(triplet_dict[key])

        # Plotting
        true_list = [true_means, true_stds]
        errors_list = [mean_errors, std_errors]
        predictions_list = [mean_predictions, std_predictions]
        string_list = ["mean", "std"]
        x_lim_list = [(0.0, 10.0), (0.0, 1.7)]
        y_lim_1_list = [(0.0, 4.0), (0.0, 1.7)]
        y_lim_2_list = [(0.0, 10.0), (0.0, 1.7)]

        for i in range(2):
            simple_plot(array_x=true_list[i], array_y=errors_list[i],
                        title="Error vs True Distance: " + string_list[i], x_label="True", y_label="Error",
                        save_path=os.path.join(args.save_dir, string_list[i] + "-error-vs-distance"))

            simple_plot(array_x=np.log(true_list[i]), array_y=np.log(errors_list[i]),
                        title="Log Error vs Log True Distance: " + string_list[i], x_label="Log True",
                        y_label="Log Error",
                        save_path=os.path.join(args.save_dir, string_list[i] + "-log-error-vs-log-distance"))

            simple_plot(array_x=shortest_paths, array_y=errors_list[i],
                        title="Error vs Shortest Path Length: " + string_list[i], x_label="Shortest Path Length",
                        y_label="Error",
                        save_path=os.path.join(args.save_dir, string_list[i] + "-error-vs-path-length"),
                        y_lim=(0.0, 4.0))

            simple_plot(array_x=true_list[i], array_y=shortest_paths,
                        title="True Distance vs Shortest Path Length: " + string_list[i], x_label="True Distance",
                        y_label="Shortest Path Length",
                        save_path=os.path.join(args.save_dir, string_list[i] + "-path-length-vs-distance"))

            simple_plot(array_x=true_list[i], array_y=predictions_list[i],
                        title="Prediction vs True Distance: " + string_list[i], x_label="True Distance",
                        y_label="Prediction",
                        save_path=os.path.join(args.save_dir, string_list[i] + "-prediction-vs-distance"))

            # Plot error vs true distance for each bond type
            for key in bond_type_dict:
                double_axis_plot(array_x=bond_type_dict[key][:, 2 + i], array_y_1=bond_type_dict[key][:, 0 + i],
                                 array_y_2=bond_type_dict[key][:, 4 + i],
                                 title="Error and Predicted vs True Distance: " + key + " " + string_list[i],
                                 x_label="True Distance", y_label_1="Error", y_label_2="Predicted",
                                 save_path=os.path.join(args.save_dir, string_list[i] +
                                                        "-error-vs-distance-" + key), x_lim=x_lim_list[i],
                                 y_lim_1=y_lim_1_list[i], y_lim_2=y_lim_2_list[i])

            for key in triplet_dict:
                double_axis_plot(array_x=triplet_dict[key][:, 2 + i], array_y_1=triplet_dict[key][:, 0 + i],
                                 array_y_2=triplet_dict[key][:, 4 + i],
                                 title="Error and Predicted vs True Distance: " + key + " " + string_list[i],
                                 x_label="True Distance", y_label_1="Error", y_label_2="Predicted",
                                 save_path=os.path.join(args.save_dir, string_list[i] +
                                                        "-error-vs-distance-" + key), x_lim=x_lim_list[i],
                                 y_lim_1=y_lim_1_list[i], y_lim_2=y_lim_2_list[i])

    else:
        loss_avg = error_sum / batch_count
        debug("Test error avg = {:.4e}".format(loss_avg))

        # Convert to numpy
        errors = np.array(errors)
        true_distances = np.array(true_distances)
        shortest_paths = np.array(shortest_paths)
        predictions = np.array(predictions)

        # Load items into bond type dictionary
        for i in range(len(bond_types)):
            if bond_types[i] in bond_type_dict:
                bond_type_dict[bond_types[i]].append([errors[i], true_distances[i], predictions[i]])
            else:
                bond_type_dict[bond_types[i]] = [[errors[i], true_distances[i], predictions[i]]]

        for i in range(len(triplet_types)):
            if triplet_types[i] is not None:
                if triplet_types[i] in triplet_dict:
                    triplet_dict[triplet_types[i]].append([errors[i], true_distances[i], predictions[i]])
                else:
                    triplet_dict[triplet_types[i]] = [[errors[i], true_distances[i], predictions[i]]]

        for key in bond_type_dict:
            bond_type_dict[key] = np.array(bond_type_dict[key])

        for key in triplet_dict:
            triplet_dict[key] = np.array(triplet_dict[key])

        # Plotting
        simple_plot(array_x=true_distances, array_y=errors, title="Error vs True Distance",
                    x_label="True", y_label="Error", save_path=os.path.join(args.save_dir, "error-vs-distance"),
                    y_lim=(0.0, 4.0))

        simple_plot(array_x=np.log(true_distances), array_y=np.log(errors),
                    title="Log Error vs Log True Distance", x_label="Log True", y_label="Log Error",
                    save_path=os.path.join(args.save_dir, "log-error-vs-log-distance"), x_lim=(0.0, 2.5),
                    y_lim=(-15.0, 2.0))

        simple_plot(array_x=shortest_paths, array_y=errors, title="Error vs Shortest Path Length",
                    x_label="Shortest Path Length", y_label="Error",
                    save_path=os.path.join(args.save_dir, "error-vs-path-length"), y_lim=(0.0, 4.0))

        simple_plot(array_x=true_distances, array_y=shortest_paths,
                    title="Shortest Path Length vs True Distance", x_label="True Distance",
                    y_label="Shortest Path Length",
                    save_path=os.path.join(args.save_dir, "path-length-vs-distance"))

        simple_plot(array_x=true_distances, array_y=predictions, title="Prediction vs True Distance",
                    x_label="True Distance", y_label="Prediction",
                    save_path=os.path.join(args.save_dir, "prediction-vs-distance"))

        # Plot error vs true distance for each bond type
        for key in bond_type_dict:
            double_axis_plot(array_x=bond_type_dict[key][:, 1], array_y_1=bond_type_dict[key][:, 0],
                             array_y_2=bond_type_dict[key][:, 2],
                             title="Error and Predicted vs True Distance: " + key, x_label="True Distance",
                             y_label_1="Error", y_label_2="Predicted",
                             save_path=os.path.join(args.save_dir, "error-vs-distance-" + key), x_lim=(0.0, 10.0),
                             y_lim_1=(0.0, 4.0), y_lim_2=(0.0, 10.0))

        # Plot error vs true distance for each triplet type
        for key in triplet_dict:
            double_axis_plot(array_x=triplet_dict[key][:, 1], array_y_1=triplet_dict[key][:, 0],
                             array_y_2=triplet_dict[key][:, 2],
                             title="Error and Predicted vs True Distance: " + key, x_label="True Distance",
                             y_label_1="Error", y_label_2="Predicted",
                             save_path=os.path.join(args.save_dir, "error-vs-distance-" + key), x_lim=(0.0, 10.0),
                             y_lim_1=(0.0, 4.0), y_lim_2=(0.0, 10.0))

        # Save errors as numpy array
        np.save(os.path.join(args.save_dir, "errors"), errors)
