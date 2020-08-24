""" Run chem prop network training. """
import json
from logging import Logger
import numpy as np
import os
from typing import Tuple

from sklearn.model_selection import train_test_split
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm, trange

from chemprop.args import TrainArgs
from chemprop.data import MoleculeDataLoader, MoleculeDatapoint, MoleculeDataset
from chemprop.models import MoleculeModel
from conformation.dataloader import DataLoader
from conformation.dataloader_jonas import DataLoader as DataLoaderImproved
from conformation.dataset import GraphDataset
from conformation.relational import RelationalNetwork
from conformation.relational_jonas import MPMLPOutNet
from conformation.relational_utils import load_relational_checkpoint
from conformation.train_args_relational import Args
from conformation.utils import param_count


def train(model: nn.Module, optimizer: Adam, data: MoleculeDataLoader, args: Args, logger: Logger, n_iter: int,
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
        batch: MoleculeDataset
        mol_batch, target_batch = batch.batch_graph(), batch.targets()
        target_batch = np.concatenate(target_batch)
        targets = torch.from_numpy(target_batch)

        # Zero gradients
        model.zero_grad()

        # Predictions
        preds = model(mol_batch)

        # Targets
        if args.std:
            targets = targets.to(preds)
        else:
            targets = targets.to(preds)[:, 0].unsqueeze(1)

        # Compute loss
        if args.std:
            loss = loss_func(preds[:, 0], targets[:, 0]) + args.alpha * loss_func(preds[:, 1], targets[:, 1])
        else:
            loss = loss_func(preds, targets)

        loss_sum += loss.item()
        batch_count += 1
        if args.improved_architecture:
            n_iter += targets.shape[0]
        else:
            n_iter += len(batch)

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


def run_chemprop_training(args: Args, logger: Logger) -> None:
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

    # Extract smiles and targets
    # noinspection PyTypeChecker
    data = [MoleculeDatapoint(smiles=x['smiles'], targets=np.load(x['target'])) for x in metadata]

    # Data plist
    train_data, remaining_data = train_test_split(data, test_size=0.2, random_state=0)
    val_data, test_data = train_test_split(remaining_data, test_size=0.5, random_state=0)

    # Datasets
    train_data = MoleculeDataset(train_data)
    val_data = MoleculeDataset(val_data)
    test_data = MoleculeDataset(test_data)

    train_data_length, val_data_length, test_data_length = len(train_data), len(val_data), len(test_data)
    debug(f'train size = {train_data_length:,} | val size = {val_data_length:,} | test size = {test_data_length:,}'
          )

    # Data loaders
    train_data = MoleculeDataLoader(dataset=train_data, batch_size=args.batch_size, num_workers=args.num_workers,
                                    shuffle=True)
    val_data = MoleculeDataLoader(dataset=val_data, batch_size=args.batch_size, num_workers=args.num_workers)
    test_data = MoleculeDataLoader(dataset=test_data, batch_size=args.batch_size, num_workers=args.num_workers)

    # Load model
    train_args = TrainArgs().parse_args(['--data_path', 'poop', '--dataset_type', 'regression', '--depth',
                                         str(args.num_layers)])
    if args.std:
        train_args.task_names = ['mean', 'std']
    else:
        train_args.task_names = ['mean']
    model = MoleculeModel(train_args)

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
        # state = {
        #     'args': args.as_dict(),
        #     'state_dict': model.state_dict()
        # }
        # torch.save(state, os.path.join(args.save_dir, "checkpoints", 'model-' + str(epoch) + '.pt'))
        # val_error_avg, val_mean_error_avg, val_std_error_avg = evaluate(model, val_data, args, loss_func)
        # debug(f"Epoch {epoch} validation error avg = {val_error_avg:.4e}")
        # summary_writer.add_scalar("Validation Average Error", val_error_avg, epoch)
        # if args.std:
        #     debug(f"Epoch {epoch} validation mean error avg = {val_mean_error_avg:.4e}")
        #     summary_writer.add_scalar("Validation Mean Average Error", val_mean_error_avg, epoch)
        #     debug(f"Epoch {epoch} validation std error avg = {val_std_error_avg:.4e}")
        #     summary_writer.add_scalar("Validation Std Average Error", val_std_error_avg, epoch)
        #
        # if val_error_avg < best_error:
        #     torch.save(state, os.path.join(args.save_dir, "checkpoints", 'best.pt'))
        #     best_error = val_error_avg
        #     best_epoch = epoch
        #     if args.std:
        #         best_mean_error = val_mean_error_avg
        #         best_std_error = val_std_error_avg

    # # Print best validation error(s)
    # debug(f"Best epoch: {best_epoch} with validation error avg = {best_error:.4e}")
    # if args.std:
    #     debug(f"Best validation mean error avg = {best_mean_error:.4e}")
    #     debug(f"Best validation std error avg = {best_std_error:.4e}")
    #
    # # Run test evaluation
    # model, _ = load_relational_checkpoint(os.path.join(args.save_dir, "checkpoints", 'best.pt'), Args())
    # if args.cuda:
    #     print('Moving model to cuda')
    #     model = model.cuda()
    # test_error_avg, test_mean_error_avg, test_std_error_avg = evaluate(model, test_data, args, loss_func)
    # debug(f"Test error avg = {test_error_avg:.4e}")
    # if args.std:
    #     debug(f"Test mean error avg = {test_mean_error_avg:.4e}")
    #     debug(f"Test std error avg = {test_std_error_avg:.4e}")
