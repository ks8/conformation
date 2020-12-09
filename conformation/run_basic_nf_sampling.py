""" Run sampling for NF trained on non-molecule data. """
from logging import Logger
import math
import numpy as np
import os
import scipy.stats
import time
from typing_extensions import Literal

# noinspection PyPackageRequirements
from tap import Tap
import torch
from tqdm import tqdm

from conformation.utils import load_checkpoint, param_count
from conformation.funnel_sampler import funnel_pdf


class Args(Tap):
    """
    System arguments.
    """
    checkpoint_path: str  # Path to saved model checkpoint file
    mcmc: bool = False  # Whether or not to do MCMC-driven sampling
    num_layers: int = 10  # Number of RealNVP layers
    num_samples: int = 1000  # Number of samples
    base_dim: int = 10  # Dimension of the base distribution
    proposal_std: float = 0.1  # Isotropic MCMC proposal std
    target_distribution: Literal["funnel"] = "funnel"  # Target distribution for MCMC sampling
    subsample_frequency: int = 100  # Subsample frequency
    log_frequency: int = 1000  # Log frequency
    cuda: bool = False  # Whether or not to use cuda
    gpu_device: int = 0  # Which GPU to use (0 or 1)
    save_dir: str = None  # Save directory


def run_basic_nf_sampling(args: Args, logger: Logger) -> None:
    """
    Perform neural network training.
    :param args: System parameters.
    :param logger: System logger.
    :return: None.
    """
    # Set up logger
    debug, info = logger.debug, logger.info

    args.cuda = torch.cuda.is_available()
    print(args)

    # Load model
    if args.checkpoint_path is not None:
        debug('Loading model from {}'.format(args.checkpoint_path))
        model = load_checkpoint(args.checkpoint_path, args.cuda, args.gpu_device)

        debug(model)
        debug('Number of parameters = {:,}'.format(param_count(model)))

        if args.cuda:
            # noinspection PyUnresolvedReferences
            with torch.cuda.device(args.gpu_device):
                debug('Moving model to cuda')
                model = model.cuda()

        with torch.no_grad():
            model.eval()

            if args.mcmc:
                debug("Starting MCMC search...")

                # Specify the target distribution pdf function
                if args.target_distribution == "funnel":
                    target_pdf = funnel_pdf

                # Samples list
                samples = []

                # Define the base distribution
                rv = scipy.stats.multivariate_normal(mean=np.zeros(args.base_dim), cov=np.ones(args.base_dim))

                # Generate an initial sample from the base space
                current_base_sample = np.array([rv.rvs(1)])
                z = torch.from_numpy(current_base_sample).type(torch.float32)
                if args.cuda:
                    # noinspection PyUnresolvedReferences
                    with torch.cuda.device(args.gpu_device):
                        z = z.cuda()

                # Transform the sample to one from the target space and compute the list of absolute value of Jacobian
                # determinants
                current_target_sample, log_det = model.forward_pass_with_log_abs_det_jacobian(z)

                # Compute the product of the determinants
                log_det = [math.exp(i) for i in log_det]

                # Compute the probability under the target distribution
                current_target_sample = current_target_sample.cpu().numpy()
                current_target_probability = target_pdf(current_target_sample[0]) * np.prod(log_det)

                samples.append(current_target_sample[0])

                debug(f'Running MC steps...')
                num_accepted = 0
                start_time = time.time()
                for step in tqdm(range(args.num_samples)):
                    # Generate an isotropic proposal in the base space
                    proposed_base_sample = np.array([current_base_sample[0] +
                                                     np.random.normal(0, args.proposal_std, args.base_dim)])
                    z = torch.from_numpy(proposed_base_sample).type(torch.float32)
                    if args.cuda:
                        # noinspection PyUnresolvedReferences
                        with torch.cuda.device(args.gpu_device):
                            z = z.cuda()

                    # Transform the sample to one from the target space and compute the list of absolute value of
                    # Jacobian determinants
                    proposed_target_sample, log_det = model.forward_pass_with_log_abs_det_jacobian(z)

                    # Compute the product of the determinants
                    log_det = [math.exp(i) for i in log_det]

                    # Compute the probability under the target distribution
                    proposed_target_sample = proposed_target_sample.cpu().numpy()
                    proposed_target_probability = target_pdf(proposed_target_sample[0]) * np.prod(log_det)

                    # Apply Metropolis-Hastings acceptance criterion
                    prob_ratio = proposed_target_probability / current_target_probability
                    mu = np.random.uniform(0, 1)
                    if mu <= prob_ratio:
                        current_base_sample = proposed_base_sample
                        current_target_sample = proposed_target_sample
                        current_target_probability = proposed_target_probability

                        num_accepted += 1

                    if step % args.subsample_frequency == 0:
                        samples.append(current_target_sample[0])

                    if step % args.log_frequency == 0:
                        if num_accepted == 0:
                            acceptance_percentage = 0.0
                        else:
                            acceptance_percentage = float(num_accepted) / float(step + 1) * 100.0
                        debug(f'Steps completed: {step}, acceptance percentage: {acceptance_percentage}')
                end_time = time.time()
                debug(f'Total Time (s): {end_time - start_time}')
                debug(f'% Moves Accepted: {num_accepted / args.num_samples}')

                # Save samples
                samples = np.array(samples)
                np.save(os.path.join(args.save_dir, "samples.npy"), samples)

            else:
                debug("Starting NF sampling...")
                samples = []
                for _ in tqdm(range(args.num_samples)):
                    gen_sample = model.sample(args.num_layers)
                    samples.append(gen_sample.cpu().numpy())
                samples = np.array(samples)
                np.save(os.path.join(args.save_dir, "samples.npy"), samples)

    else:
        print('Must specify a model to load.')
        exit()
