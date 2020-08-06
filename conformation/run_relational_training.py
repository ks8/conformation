""" Run relational network training. """
import json
from logging import Logger
import os
from typing import Tuple

from sklearn.model_selection import train_test_split
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm, trange

from conformation.dataloader import DataLoader
from conformation.dataset import GraphDataset
from conformation.relational import RelationalNetwork
from conformation.relational_utils import load_relational_checkpoint
from conformation.train_args_relational import Args
from conformation.utils import param_count


def train(model: nn.Module, optimizer: Adam, data: DataLoader, args: Args, logger: Logger, n_iter: int,
          loss_func: torch.nn.MSELoss, summary_writer: SummaryWriter) -> int:
    """
    Function for training a relational network.
    :param model: Neural network.
    :param optimizer: Adam optimizer.
    :param data: DataLoader.
    :param args: System arguments.
    :param logger: System logger.
    :param n_iter: Total number of iterations.
    :param loss_func: MSE loss function.
    :param summary_writer: TensorboardX summary writer.
    :return: total number of iterations.
    """
    # Set up logger
    debug, info = logger.debug, logger.info

    # Run training
    model.train()
    loss_sum, batch_count = 0, 0
    for batch in tqdm(data, total=len(data)):
        # Move batch to cuda
        batch.x = batch.x.cuda()
        batch.edge_attr = batch.edge_attr.cuda()

        # Zero gradients
        model.zero_grad()

        # Extract targets
        if args.std:
            targets = batch.y.cuda()
        else:
            targets = batch.y.cuda()[:, 0].unsqueeze(1)

        # Generate predictions
        preds = model(batch)

        # Compute loss
        if args.std:
            loss = loss_func(preds[:, 0], targets[:, 0]) + args.alpha * loss_func(preds[:, 1], targets[:, 1])
        else:
            loss = loss_func(preds, targets)
        loss_sum += loss.item()
        batch_count += 1
        n_iter += batch.num_graphs

        # Backpropagation
        loss.backward()
        optimizer.step()

        # Logging
        if (n_iter // args.batch_size) % args.log_frequency == 0:
            loss_avg = loss_sum / batch_count
            loss_sum, batch_count = 0, 0
            debug("Train loss avg = {:.4e}".format(loss_avg))
            summary_writer.add_scalar("Avg Train Loss", loss_avg, n_iter)

    return n_iter


def evaluate(model: nn.Module, data: DataLoader, args: Args, loss_func: torch.nn.MSELoss) -> Tuple[float, float, float]:
    """
    Function for training a relational network.
    :param model: Neural network.
    :param data: DataLoader.
    :param args: System arguments.
    :param loss_func: MSE loss function.
    :return: total number of iterations.
    """
    # Run evaluation
    with torch.no_grad():
        error_avg, mean_error_avg, std_error_avg, batch_count = 0, 0, 0, 0
        model.eval()
        for batch in tqdm(data, total=len(data)):
            # Move batch to cuda
            batch.x = batch.x.cuda()
            batch.edge_attr = batch.edge_attr.cuda()

            # Extract targets
            if args.std:
                targets = batch.y.cuda()
            else:
                targets = batch.y.cuda()[:, 0].unsqueeze(1)

            # Generate predictions
            preds = model(batch)

            # Compute error
            if args.std:
                error_avg += torch.sqrt_(loss_func(preds[:, 0], targets[:, 0]) +
                                         args.alpha * loss_func(preds[:, 1], targets[:, 1])).item()
                mean_error_avg += torch.sqrt_(loss_func(preds[:, 0], targets[:, 0])).item()
                std_error_avg += torch.sqrt_(loss_func(preds[:, 1], targets[:, 1])).item()
            else:
                error_avg += torch.sqrt_(loss_func(preds, targets)).item()
            batch_count += 1

        # Compute average error
        error_avg /= batch_count
        if args.std:
            mean_error_avg /= batch_count
            std_error_avg /= batch_count

    return error_avg, mean_error_avg, std_error_avg


def run_relational_training(args: Args, logger: Logger) -> None:
    """
    Run training of relational neural network.
    :param args: System arguments.
    :param logger: Logging.
    :return: None.
    """

    # Save directories
    os.makedirs(os.path.join(args.save_dir, "checkpoints"))

    # Set up logger
    debug, info = logger.debug, logger.info

    # Print args
    debug(args)

    # Check cuda availability
    args.cuda = torch.cuda.is_available()

    # Load data
    debug("loading data")
    metadata = json.load(open(args.data_path))
    train_metadata, remaining_metadata = train_test_split(metadata, test_size=0.2, random_state=0)
    validation_metadata, test_metadata = train_test_split(remaining_metadata, test_size=0.5, random_state=0)

    # Convert to dataset
    train_data = GraphDataset(train_metadata, atom_types=args.atom_types, bond_types=args.bond_types,
                              max_path_length=args.max_shortest_path_length, atomic_num=args.atomic_num,
                              partial_charge=args.partial_charge, mmff_atom_types_one_hot=args.mmff_atom_types_one_hot,
                              valence_types=args.valence_types, valence=args.valence, aromatic=args.aromatic,
                              hybridization=args.hybridization, assign_stereo=args.assign_stereo,
                              charge_types=args.charge_types, formal_charge=args.formal_charge,
                              r_covalent=args.r_covalent, r_vanderwals=args.r_vanderwals,
                              default_valence=args.default_valence, max_ring_size=args.max_ring_size, rings=args.rings,
                              chirality=args.chirality, mmff94_atom_types=args.mmff94_atom_types,
                              hybridization_types=args.hybridization_types, chi_types=args.chi_types)
    val_data = GraphDataset(validation_metadata, atom_types=args.atom_types, bond_types=args.bond_types,
                            max_path_length=args.max_shortest_path_length, atomic_num=args.atomic_num,
                            partial_charge=args.partial_charge, mmff_atom_types_one_hot=args.mmff_atom_types_one_hot,
                            valence_types=args.valence_types, valence=args.valence, aromatic=args.aromatic,
                            hybridization=args.hybridization, assign_stereo=args.assign_stereo,
                            charge_types=args.charge_types, formal_charge=args.formal_charge,
                            r_covalent=args.r_covalent, r_vanderwals=args.r_vanderwals,
                            default_valence=args.default_valence, max_ring_size=args.max_ring_size, rings=args.rings,
                            chirality=args.chirality, mmff94_atom_types=args.mmff94_atom_types,
                            hybridization_types=args.hybridization_types, chi_types=args.chi_types)
    test_data = GraphDataset(test_metadata, atom_types=args.atom_types, bond_types=args.bond_types,
                             max_path_length=args.max_shortest_path_length, atomic_num=args.atomic_num,
                             partial_charge=args.partial_charge, mmff_atom_types_one_hot=args.mmff_atom_types_one_hot,
                             valence_types=args.valence_types, valence=args.valence, aromatic=args.aromatic,
                             hybridization=args.hybridization, assign_stereo=args.assign_stereo,
                             charge_types=args.charge_types, formal_charge=args.formal_charge,
                             r_covalent=args.r_covalent, r_vanderwals=args.r_vanderwals,
                             default_valence=args.default_valence, max_ring_size=args.max_ring_size, rings=args.rings,
                             chirality=args.chirality, mmff94_atom_types=args.mmff94_atom_types,
                             hybridization_types=args.hybridization_types, chi_types=args.chi_types)

    train_data_length, val_data_length, test_data_length = len(train_data), len(val_data), len(test_data)
    debug(f'train size = {train_data_length:,} | val size = {val_data_length:,} | test size = {test_data_length:,}'
          )

    # Convert to iterators
    train_data = DataLoader(train_data, args.batch_size)
    val_data = DataLoader(val_data, args.batch_size)
    test_data = DataLoader(test_data, args.batch_size)

    # Load/build model
    if args.checkpoint_path is not None:
        debug('Loading model from {}'.format(args.checkpoint_path))
        model, _ = load_relational_checkpoint(args.checkpoint_path, Args())
    else:
        args.num_edge_features = len(args.bond_types) + args.max_shortest_path_length + 1
        args.num_vertex_features = 0
        if args.atomic_num:
            args.num_vertex_features += len(args.atom_types)
        if args.valence:
            args.num_vertex_features += len(args.valence_types)
        if args.partial_charge:
            args.num_vertex_features += 1
        if args.mmff_atom_types_one_hot:
            args.num_vertex_features += len(args.mmff94_atom_types)
        if args.aromatic:
            args.num_vertex_features += 1
        if args.hybridization:
            args.num_vertex_features += len(args.hybridization_types)
        if args.formal_charge:
            args.num_vertex_features += len(args.charge_types)
        if args.r_covalent:
            args.num_vertex_features += 1
        if args.r_vanderwals:
            args.num_vertex_features += 1
        if args.default_valence:
            args.num_vertex_features += len(args.valence_types)
        if args.rings:
            args.num_vertex_features += args.max_ring_size - 2
        if args.chirality:
            args.num_vertex_features += len(args.chi_types)

        debug('Building model')
        model = RelationalNetwork(args.hidden_size, args.num_layers, args.num_edge_features, args.num_vertex_features,
                                  args.final_linear_size, args.final_output_size)

    # Print model info
    debug(model)
    debug('Number of parameters = {:,}'.format(param_count(model)))

    # Move model to cuda if available
    if args.cuda:
        print('Moving model to cuda')
        model = model.cuda()

    # Loss func and optimizer
    loss_func = torch.nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=1e-4)

    # Run training
    summary_writer = SummaryWriter(logdir=args.save_dir)
    best_epoch, n_iter = 0, 0
    best_error = best_mean_error = best_std_error = float('inf')
    for epoch in trange(args.num_epochs):
        n_iter = train(model, optimizer, train_data, args, logger, n_iter, loss_func, summary_writer)
        state = {
            'args': args.as_dict(),
            'state_dict': model.state_dict()
        }
        torch.save(state, os.path.join(args.save_dir, "checkpoints", 'model-' + str(epoch) + '.pt'))
        val_error_avg, val_mean_error_avg, val_std_error_avg = evaluate(model, val_data, args, loss_func)
        debug(f"Epoch {epoch} validation error avg = {val_error_avg:.4e}")
        summary_writer.add_scalar("Validation Average Error", val_error_avg, epoch)
        if args.std:
            debug(f"Epoch {epoch} validation mean error avg = {val_mean_error_avg:.4e}")
            summary_writer.add_scalar("Validation Mean Average Error", val_mean_error_avg, epoch)
            debug(f"Epoch {epoch} validation std error avg = {val_std_error_avg:.4e}")
            summary_writer.add_scalar("Validation Std Average Error", val_std_error_avg, epoch)

        if val_error_avg < best_error:
            torch.save(state, os.path.join(args.save_dir, "checkpoints", 'best.pt'))
            best_error = val_error_avg
            best_epoch = epoch
            if args.std:
                best_mean_error = val_mean_error_avg
                best_std_error = val_std_error_avg

    # Print best validation error(s)
    debug(f"Best epoch: {best_epoch} with validation error avg = {best_error:.4e}")
    if args.std:
        debug(f"Best validation mean error avg = {best_mean_error:.4e}")
        debug(f"Best validation std error avg = {best_std_error:.4e}")

    # Run test evaluation
    model, _ = load_relational_checkpoint(os.path.join(args.save_dir, "checkpoints", 'best.pt'), Args())
    if args.cuda:
        print('Moving model to cuda')
        model = model.cuda()
    test_error_avg, test_mean_error_avg, test_std_error_avg = evaluate(model, test_data, args, loss_func)
    debug(f"Test error avg = {test_error_avg:.4e}")
    if args.std:
        debug(f"Test mean error avg = {test_mean_error_avg:.4e}")
        debug(f"Test std error avg = {test_std_error_avg:.4e}")
