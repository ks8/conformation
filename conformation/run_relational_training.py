""" Run relational network training. """
from logging import Logger
import json
import os

from sklearn.model_selection import train_test_split
from tensorboardX import SummaryWriter
# noinspection PyPackageRequirements
from tap import Tap
import torch
from torch.optim import Adam
from tqdm import tqdm, trange

from conformation.dataloader import DataLoader
from conformation.dataset import GraphDataset
from conformation.relational import RelationalNetwork
from conformation.utils import param_count


class Args(Tap):
    """
    System arguments.
    """
    data_path: str  # Path to metadata file
    num_epochs: int  # Number of training epochs
    batch_size: int = 10  # Batch size
    lr: float = 1e-4  # Learning rate
    hidden_size: int = 256  # Hidden size
    num_layers: int = 10  # Number of layers
    num_edge_features: int = 16  # Number of edge features
    final_linear_size: int = 1024  # Size of last linear layer
    final_output_size: int = 1  # Size of output layer
    num_vertex_features: int = 5  # Number of vertex features
    cuda: bool = False  # Cuda availability
    checkpoint_path: str = None  # Directory of checkpoint to load saved model
    save_dir: str  # Save directory
    log_frequency: int = 10  # Log frequency
    std: bool = False  # Whether or not to additionally train on atomic pairwise distance standard deviation
    alpha: float = 20.0  # How much to weight positive std prediction losses
    beta: float = 2000.0  # How much to weight negative std prediction losses
    # TODO: add option for atom types and bond types instead of num features...


def train(model: RelationalNetwork, optimizer: Adam, data: DataLoader, args: Args, logger: Logger, n_iter: int,
          loss_func: torch.nn.MSELoss, loss_func_aux: torch.nn.MSELoss, summary_writer: SummaryWriter) -> int:
    """
    Function for training a relational network.
    :param model: Neural network.
    :param optimizer: Adam optimizer.
    :param data: DataLoader.
    :param args: System arguments.
    :param logger: System logger.
    :param n_iter: Total number of iterations.
    :param loss_func: MSE loss function.
    :param loss_func_aux: MSE loss function for std predictions.
    :param summary_writer: TensorboardX summary writer.
    :return: total number of iterations.
    """
    # Set up logger
    debug, info = logger.debug, logger.info

    model.train()
    loss_sum, batch_count = 0, 0
    for batch in tqdm(data, total=len(data)):
        batch.x = batch.x.cuda()
        batch.edge_attr = batch.edge_attr.cuda()

        model.zero_grad()
        if args.std:
            targets = batch.y.cuda()
        else:
            targets = batch.y.cuda()[:, 0].unsqueeze(1)
        # noinspection PyCallingNonCallable
        preds = model(batch)
        if args.std:
            # loss = loss_func(preds[:, 0], targets[:, 0]) + args.alpha*loss_func(preds[:, 1], targets[:, 1])
            std_loss_weights = args.alpha*(preds[:, 1] > 0) + args.beta*(preds[:, 1] < 0)
            mean_loss = loss_func(preds[:, 0], targets[:, 0])
            std_loss = torch.mean(loss_func_aux(preds[:, 1], targets[:, 1])*std_loss_weights)
            loss = mean_loss + std_loss
        else:
            loss = loss_func(preds, targets)
        loss_sum += loss.item()
        batch_count += 1
        n_iter += batch.num_graphs

        loss.backward()
        optimizer.step()

        if (n_iter // args.batch_size) % args.log_frequency == 0:
            loss_avg = loss_sum / batch_count
            loss_sum, batch_count = 0, 0
            debug("Train loss avg = {:.4e}".format(loss_avg))
            summary_writer.add_scalar("Avg Train Loss", loss_avg, n_iter)

    return n_iter


def evaluate(model: RelationalNetwork, data: DataLoader, args: Args, loss_func: torch.nn.MSELoss,
             loss_func_aux: torch.nn.MSELoss) -> float:
    """
    Function for training a relational network.
    :param model: Neural network.
    :param data: DataLoader.
    :param args: System arguments.
    :param loss_func: MSE loss function.
    :param loss_func_aux: MSE loss function for std predictions.
    :return: total number of iterations.
    """
    with torch.no_grad():
        loss_sum, batch_count = 0, 0
        model.eval()
        for batch in tqdm(data, total=len(data)):
            batch.x = batch.x.cuda()
            batch.edge_attr = batch.edge_attr.cuda()
            if args.std:
                targets = batch.y.cuda()
            else:
                targets = batch.y.cuda()[:, 0].unsqueeze(1)
            preds = model(batch)
            if args.std:
                # loss = loss_func(preds[:, 0], targets[:, 0]) + args.alpha*loss_func(preds[:, 1], targets[:, 1])
                std_loss_weights = args.alpha * (preds[:, 1] > 0) + args.beta * (preds[:, 1] < 0)
                mean_loss = loss_func(preds[:, 0], targets[:, 0])
                std_loss = torch.mean(loss_func_aux(preds[:, 1], targets[:, 1]) * std_loss_weights)
                loss = mean_loss + std_loss
            else:
                loss = loss_func(preds, targets)
            loss = torch.sqrt_(loss)
            loss_sum += loss.item()
            batch_count += 1
        loss_avg = loss_sum / batch_count

    return loss_avg


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

    debug(args)

    args.cuda = torch.cuda.is_available()

    metadata = json.load(open(args.data_path))
    train_metadata, remaining_metadata = train_test_split(metadata, test_size=0.2, random_state=0)
    validation_metadata, test_metadata = train_test_split(remaining_metadata, test_size=0.5, random_state=0)

    debug("loading data")
    train_data = GraphDataset(train_metadata)
    val_data = GraphDataset(validation_metadata)
    test_data = GraphDataset(test_metadata)

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
        # Load model and args
        state = torch.load(args.checkpoint_path, map_location=lambda storage, loc: storage)
        loaded_args = Args().from_dict(state['args'])
        loaded_state_dict = state['state_dict']

        model = RelationalNetwork(loaded_args.hidden_size, loaded_args.num_layers, loaded_args.num_edge_features,
                                  loaded_args.num_vertex_features, loaded_args.final_linear_size,
                                  loaded_args.final_output_size)
        model.load_state_dict(loaded_state_dict)
    else:
        debug('Building model')
        model = RelationalNetwork(args.hidden_size, args.num_layers, args.num_edge_features, args.num_vertex_features,
                                  args.final_linear_size, args.final_output_size)

    debug(model)
    debug('Number of parameters = {:,}'.format(param_count(model)))

    if args.cuda:
        print('Moving model to cuda')
        model = model.cuda()

    # Loss func and optimizer
    loss_func = torch.nn.MSELoss()
    loss_func_aux = torch.nn.MSELoss(reduction='none')
    optimizer = Adam(model.parameters(), lr=1e-4)

    summary_writer = SummaryWriter(logdir=args.save_dir)
    best_epoch, n_iter = 0, 0
    best_metric_eval = float('inf')
    for epoch in trange(args.num_epochs):
        n_iter = train(model, optimizer, train_data, args, logger, n_iter, loss_func, loss_func_aux, summary_writer)
        state = {
            'args': args.as_dict(),
            'state_dict': model.state_dict()
        }
        torch.save(state, os.path.join(args.save_dir, "checkpoints", 'model-' + str(epoch) + '.pt'))
        val_metric_avg = evaluate(model, val_data, args, loss_func, loss_func_aux)
        debug(f"Epoch {epoch} validation error avg = {val_metric_avg:.4e}")
        summary_writer.add_scalar("Validation Average Error", val_metric_avg, epoch)

        if val_metric_avg < best_metric_eval:
            torch.save(state, os.path.join(args.save_dir, "checkpoints", 'best.pt'))
            best_metric_eval = val_metric_avg
            best_epoch = epoch

    debug(f"Best epoch: {best_epoch} with validation error avg = {best_metric_eval:.4e}")

    # Load model and args
    state = torch.load(os.path.join(args.save_dir, "checkpoints", 'best.pt'), map_location=lambda storage, loc: storage)
    loaded_args = Args().from_dict(state['args'])
    loaded_state_dict = state['state_dict']

    model = RelationalNetwork(loaded_args.hidden_size, loaded_args.num_layers, loaded_args.num_edge_features,
                              loaded_args.num_vertex_features, loaded_args.final_linear_size,
                              loaded_args.final_output_size)
    model.load_state_dict(loaded_state_dict)
    if args.cuda:
        print('Moving model to cuda')
        model = model.cuda()
    test_metric_avg = evaluate(model, test_data, args, loss_func, loss_func_aux)
    debug(f"Test error avg = {test_metric_avg:.4e}")
