""" Run density evaluation for NF trained on non-molecule data. """
import numpy as np
import os

# noinspection PyPackageRequirements
from tap import Tap
import torch
from tqdm import tqdm

from conformation.utils import load_checkpoint, param_count, density_func, density_func_cnf


class Args(Tap):
    """
    System arguments.
    """
    checkpoint_path: str  # Path to saved model checkpoint file
    evaluation_points_path: str  # Path to numpy file containing points to evaluate the density of
    condition_path: str = None  # Path to condition numpy file for conditional normalizing flow
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
    if args.cuda:
        evaluation_points = evaluation_points.cuda()

    # Load model
    if args.checkpoint_path is not None:
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
                condition = np.load(args.condition_path)
                condition = torch.from_numpy(condition)
                condition = condition.type(torch.float32)
                if args.cuda:
                    condition = condition.cuda()
                densities = []
                for i in tqdm(range(len(evaluation_points))):
                    if model.conditional_base:
                        z, log_jacobians, means = model(evaluation_points[i].unsqueeze(0), condition.unsqueeze(0))
                        density = density_func_cnf(z, log_jacobians, means, args.cuda).item()
                    elif model.conditional_concat:
                        z, log_jacobians = model(evaluation_points[i].unsqueeze(0), condition.unsqueeze(0))
                        density = density_func(z, log_jacobians, model.base_dist).item()
                    else:
                        z, log_jacobians = model(evaluation_points[i].unsqueeze(0))
                        density = density_func(z, log_jacobians, model.base_dist).item()
                    densities.append(density)
            densities = np.array(densities)
            np.save(os.path.join(args.save_dir, "densities.npy"), densities)

    else:
        print('Must specify a model to load.')
        exit()
