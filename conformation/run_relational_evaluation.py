""" Run relational network evaluation. """
import itertools
import json
from logging import Logger
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from typing import Tuple
from typing_extensions import Literal

from rdkit import Chem
from rdkit.Chem import rdmolops
from sklearn.model_selection import train_test_split
# noinspection PyPackageRequirements
from tap import Tap
import torch
from tqdm import tqdm

from conformation.dataloader import DataLoader
from conformation.dataset import GraphDataset
from conformation.relational import RelationalNetwork
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
    Plot one array against another.
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


def run_relational_evaluation(args: Args, logger: Logger) -> None:
    """
    Run evaluation of relational neural network.
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

    # Load metadata
    debug("loading data")
    metadata = json.load(open(args.data_path))
    train_metadata, remaining_metadata = train_test_split(metadata, test_size=0.2, random_state=0)
    validation_metadata, test_metadata = train_test_split(remaining_metadata, test_size=0.5, random_state=0)

    # Create datasets
    train_data = GraphDataset(train_metadata)
    val_data = GraphDataset(validation_metadata)
    test_data = GraphDataset(test_metadata)

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

    # Load/build model
    debug('Loading model from {}'.format(args.checkpoint_path))
    state = torch.load(args.checkpoint_path, map_location=lambda storage, loc: storage)
    loaded_args = RelationalTrainArgs().from_dict(state['args'])
    loaded_state_dict = state['state_dict']
    model = RelationalNetwork(loaded_args.hidden_size, loaded_args.num_layers, loaded_args.num_edge_features,
                              loaded_args.num_vertex_features, loaded_args.final_linear_size,
                              loaded_args.final_output_size)
    model.load_state_dict(loaded_state_dict)

    # Print model details
    debug(model)
    debug('Number of parameters = {:,}'.format(param_count(model)))

    if args.cuda:
        debug('Moving model to cuda')
        model = model.cuda()

    # Loss function
    loss_func = torch.nn.MSELoss()

    # Error function
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

    errors = []
    true_distances = []
    shortest_paths = []
    predictions = []
    mean_predictions = []
    std_predictions = []
    mean_errors = []
    std_errors = []
    true_means = []
    true_stds = []
    bond_types = []
    triplet_types = []
    bond_type_dict = dict()
    triplet_dict = dict()
    with torch.no_grad():
        if loaded_args.final_output_size == 2:
            mean_error_sum, std_error_sum, batch_count = 0, 0, 0
        else:
            loss_sum, batch_count = 0, 0
        model.eval()
        for batch in tqdm(data, total=len(data)):
            batch.x = batch.x.cuda()
            batch.edge_attr = batch.edge_attr.cuda()
            if loaded_args.final_output_size == 2:
                targets = batch.y.cuda()
            else:
                targets = batch.y.cuda()[:, 0].unsqueeze(1)
            preds = model(batch)

            if loaded_args.final_output_size == 2:
                if sum(preds[:, 1] < 0).item() > 0:
                    debug('Negative std prediction detected!')
                if sum(preds[:, 0] < 0).item() > 0:
                    debug('Negative mean prediction detected!')

                mean_preds = preds[:, 0].cpu().numpy()
                for i in range(mean_preds.shape[0]):
                    mean_predictions.append(mean_preds[i])

                std_preds = preds[:, 1].cpu().numpy()
                for i in range(std_preds.shape[0]):
                    std_predictions.append(std_preds[i])

                mean_error_aux = torch.sqrt_(loss_func_aux(preds[:, 0], targets[:, 0]))
                mean_error_aux = mean_error_aux.cpu().numpy()
                for i in range(mean_error_aux.shape[0]):
                    mean_errors.append(mean_error_aux[i])

                std_error_aux = torch.sqrt_(loss_func_aux(preds[:, 1], targets[:, 1]))
                std_error_aux = std_error_aux.cpu().numpy()
                for i in range(std_error_aux.shape[0]):
                    std_errors.append(std_error_aux[i])

                mean_targets_aux = targets[:, 0].cpu().numpy()
                for i in range(mean_targets_aux.shape[0]):
                    edge_distance = mean_targets_aux[i]
                    true_means.append(edge_distance)

                std_targets_aux = targets[:, 1].cpu().numpy()
                for i in range(std_targets_aux.shape[0]):
                    edge_std = std_targets_aux[i]
                    true_stds.append(edge_std)

                    # Compute shortest path length for each edge and append to shortest_paths list, and also
                    # compute the bond type and append to bond_types list
                for i in range(batch.uid.shape[0]):
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

                # Compute RMSE loss for means and stds
                mean_loss, std_loss = loss_func(preds[:, 0], targets[:, 0]), loss_func(preds[:, 1], targets[:, 1])
                mean_error, std_error = torch.sqrt_(mean_loss), torch.sqrt_(std_loss)
                mean_error_sum += mean_error.item()
                std_error_sum += std_error.item()
                batch_count += 1

            else:

                preds_numpy = preds.cpu().numpy()
                for i in range(preds_numpy.shape[0]):
                    predictions.append(preds[i][0])

                # Compute error, target (true distance), and shortest path for each edge
                # Compute error for each edge and append to errors list
                loss_aux = torch.sqrt_(loss_func_aux(preds, targets))
                loss_aux = loss_aux.cpu().numpy()
                for i in range(loss_aux.shape[0]):
                    errors.append(loss_aux[i][0])

                # Compute true distance for each edge and append to true_distances list
                if args.distance_analysis:
                    candidates = []
                targets_aux = targets.cpu().numpy()
                for i in range(targets_aux.shape[0]):
                    edge_distance = targets_aux[i][0]
                    true_distances.append(edge_distance)
                    if args.distance_analysis:
                        if args.distance_analysis_lo < edge_distance < args.distance_analysis_hi:
                            candidates.append(i)

                # Compute shortest path length for each edge and append to shortest_paths list, and also
                # compute the bond type and append to bond_types list
                if args.distance_analysis:
                    j = 0
                for i in range(batch.uid.shape[0]):
                    uid = batch.uid[i].item()
                    smiles = uid_dict[uid]
                    mol = Chem.MolFromSmiles(smiles)
                    mol = Chem.AddHs(mol)
                    for m, n in itertools.combinations(list(np.arange(mol.GetNumAtoms())), 2):
                        if args.distance_analysis:
                            if j in candidates:
                                debug(str(mol.GetAtoms()[int(m)].GetSymbol()) + " " +
                                      str(mol.GetAtoms()[int(n)].GetSymbol()) + " " +
                                      str(len(rdmolops.GetShortestPath(mol, int(m), int(n))) - 1))
                            j += 1
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

                # Compute RMSE loss
                loss = loss_func(preds, targets)
                loss = torch.sqrt_(loss)
                loss_sum += loss.item()
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
            # Plot error vs true distance
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
            loss_avg = loss_sum / batch_count
            debug("Test loss avg = {:.4e}".format(loss_avg))

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
            # Plot error vs true distance
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



