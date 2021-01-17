""" Run density evaluation for NF trained on non-molecule data. """
import numpy as np
import os

# noinspection PyPackageRequirements
from tap import Tap
import torch
from tqdm import tqdm

# noinspection PyUnresolvedReferences
from torch.utils.data import DataLoader
from conformation.dataset import TestDataset
from conformation.utils import load_checkpoint, param_count, density_func, density_func_cnf


class Args(Tap):
    """
    System arguments.
    """
    checkpoint_path: str  # Path to saved model checkpoint file
    evaluation_points_path: str  # Path to numpy file containing points to evaluate the density of
    condition_path: str = None  # Path to torch file of the condition for conditional normalizing flow
    batch_size: int = 500  # Batch size for loading data
    num_data_loader_workers: int = 2  # Number of workers for PyTorch DataLoader (0 is just the main process, > 0
    # specifies the number of subprocesses).
    cuda: bool = False  # Whether or not to use cuda
    save_dir: str = None  # Save directory


def run_basic_nf_density_evaluation(args: Args) -> None:
    """
    Run density evaluation.
    :param args: System args.
    :return: None.
    """
    os.makedirs(args.save_dir)

    args.cuda = torch.cuda.is_available()
    print(args)

    # Load density evaluation points
    evaluation_points = torch.from_numpy(np.load(args.evaluation_points_path)).type(torch.float32)

    print('Loading model from {}'.format(args.checkpoint_path))
    model = load_checkpoint(args.checkpoint_path, args.cuda)

    print(model)
    print('Number of parameters = {:,}'.format(param_count(model)))

    if args.cuda:
        # noinspection PyUnresolvedReferences
        print('Moving model to cuda')
        model = model.cuda()

    print("Starting NF density evaluation...")
    with torch.no_grad():
        model.eval()
        if model.conditional_base or model.conditional_concat:
            condition = torch.load(args.condition_path)
            data = TestDataset(evaluation_points, condition)
        else:
            data = TestDataset(evaluation_points)
        data = DataLoader(data, batch_size=args.batch_size, num_workers=args.num_data_loader_workers)
        densities = None
        for batch in tqdm(data, total=len(data)):
            if args.cuda:
                # noinspection PyUnresolvedReferences
                if model.conditional_base or model.conditional_concat:
                    batch = (batch[0].cuda(), batch[1].cuda())
                else:
                    batch = batch.cuda()
            if model.conditional_base:
                z, log_jacobians, means = model(batch[0], batch[1])
                density = density_func_cnf(z, log_jacobians, means, args.cuda)
                density = density.cpu().numpy()
            elif model.conditional_concat:
                z, log_jacobians = model(batch[0], batch[1])
                density = density_func(z, log_jacobians, model.base_dist)
                density = density.cpu().numpy()
            else:
                z, log_jacobians = model(batch)
                density = density_func(z, log_jacobians, model.base_dist)
                density = density.cpu().numpy()
            if densities is None:
                densities = density
            else:
                densities = np.concatenate([densities, density])
        np.save(os.path.join(args.save_dir, "densities.npy"), densities)
