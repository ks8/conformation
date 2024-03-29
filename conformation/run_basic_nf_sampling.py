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
from conformation.funnel_sampler import funnel_pdf, perturbed_funnel_pdf
from conformation.gmm_sampler import gmm_pdf


class Args(Tap):
    """
    System arguments.
    """
    checkpoint_path: str  # Path to saved model checkpoint file
    mcmc: bool = False  # Whether or not to do MCMC-driven sampling
    condition_path: str = None  # Path to torch file of condition for conditional normalizing flow
    num_samples: int = 1000  # Number of samples
    proposal_std: float = 0.1  # Isotropic MCMC proposal std
    target_distribution: Literal["funnel", "perturbed_funnel", "gmm"] = "funnel"  # Target for MCMC sampling
    subsample_frequency: int = 100  # Subsample frequency
    log_frequency: int = 1000  # Log frequency
    cuda: bool = False  # Whether or not to use cuda
    batch_size: int = 500  # Batch size for generating samples
    num_data_loader_workers: int = 2  # Number of workers for PyTorch DataLoader (0 is just the main process, > 0
    # specifies the number of subprocesses).
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
    debug(args)

    debug('Loading model from {}'.format(args.checkpoint_path))
    model = load_checkpoint(args.checkpoint_path, args.cuda)

    input_dim = model.input_dim

    debug(model)
    debug('Number of parameters = {:,}'.format(param_count(model)))

    if args.cuda:
        # noinspection PyUnresolvedReferences
        debug('Moving model to cuda')
        model = model.cuda()

    if args.mcmc:
        debug("Starting MCMC search...")

        with torch.no_grad():
            model.eval()

            # Specify the target distribution pdf function
            if args.target_distribution == "funnel":
                target_pdf = funnel_pdf
            elif args.target_distribution == "perturbed_funnel":
                target_pdf = perturbed_funnel_pdf
            elif args.target_distribution == "gmm":
                target_pdf = gmm_pdf

            # Samples list
            samples = []

            # Define the base distribution
            if model.conditional_base:
                condition = torch.load(args.condition_path)
                if args.cuda:
                    condition = condition.cuda()
                u = model.output_layer(model.linear_layer(condition))
                u = u.cpu().numpy()
                rv = scipy.stats.multivariate_normal(mean=u, cov=np.ones(input_dim))
            else:
                rv = scipy.stats.multivariate_normal(mean=np.zeros(input_dim), cov=np.ones(input_dim))

            # Generate an initial sample from the base space
            current_base_sample = np.array([rv.rvs(1)])
            z = torch.from_numpy(current_base_sample).type(torch.float32)
            if args.cuda:
                z = z.cuda()

            # Transform the sample to one from the target space and compute the list of absolute value of Jacobian
            # determinants
            if model.conditional_concat:
                condition = torch.load(args.condition_path)
                if args.cuda:
                    condition = condition.cuda()
                current_target_sample, log_det = \
                    model.forward_pass_with_log_abs_det_jacobian(z, condition.unsqueeze(0))
            else:
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
                                                 np.random.normal(0, args.proposal_std, input_dim)])
                z = torch.from_numpy(proposed_base_sample).type(torch.float32)
                if args.cuda:
                    z = z.cuda()

                # Transform the sample to one from the target space and compute the list of absolute value of
                # Jacobian determinants
                if model.conditional_concat:
                    proposed_target_sample, log_det = \
                        model.forward_pass_with_log_abs_det_jacobian(z, condition.unsqueeze(0))
                else:
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
            debug(f'% Moves Accepted: {num_accepted / args.num_samples * 100.}')

            # Save samples
            samples = np.array(samples)
            np.save(os.path.join(args.save_dir, "samples.npy"), samples)

    else:
        debug("Starting NF sampling...")
        with torch.no_grad():
            model.eval()
            if model.conditional_base or model.conditional_concat:
                condition = torch.load(args.condition_path)
                if args.cuda:
                    condition = condition.cuda()
                samples = model.sample(args.num_samples, args.cuda, condition, batch_size=args.batch_size,
                                       num_data_loader_workers=args.num_data_loader_workers)
            else:
                samples = model.sample(args.num_samples, args.cuda, batch_size=args.batch_size,
                                       num_data_loader_workers=args.num_data_loader_workers)
            np.save(os.path.join(args.save_dir, "samples.npy"), samples)
