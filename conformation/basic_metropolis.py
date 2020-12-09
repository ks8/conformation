""" Run basic Metropolis-Hastings sampling. """
from logging import Logger
import numpy as np
import os
import time
from typing_extensions import Literal

# noinspection PyPackageRequirements
from tap import Tap
from tqdm import tqdm

from conformation.funnel_sampler import funnel_pdf, funnel_sample


class Args(Tap):
    """
    System arguments.
    """
    num_samples: int = 1000  # Number of samples
    proposal_std: float = 0.1  # Isotropic MCMC proposal std
    target_distribution: Literal["funnel"] = "funnel"  # Target distribution for MCMC sampling
    num_funnel_x_vars: int = 9  # Number of x variables for funnel
    subsample_frequency: int = 100  # Subsample frequency
    log_frequency: int = 1000  # Log frequency
    save_dir: str = None  # Save directory


def basic_metropolis(args: Args, logger: Logger) -> None:
    """
    Perform Metropolis-Hastings sampling.
    :param args: System parameters.
    :param logger: System logger.
    :return: None.
    """
    # Set up logger
    debug, info = logger.debug, logger.info

    print(args)

    debug("Starting MCMC search...")
    # Specify the target distribution pdf function
    if args.target_distribution == "funnel":
        target_pdf = funnel_pdf
        target_sample = funnel_sample

    # Samples list
    samples = []

    # Generate an initial sample from the base space
    # noinspection PyUnboundLocalVariable
    current_sample = target_sample(args.num_funnel_x_vars)
    # noinspection PyUnboundLocalVariable
    current_probability = target_pdf(current_sample)

    samples.append(current_sample)

    debug(f'Running MC steps...')
    num_accepted = 0
    start_time = time.time()
    for step in tqdm(range(args.num_samples)):
        # Generate an isotropic proposal in the base space
        proposed_sample = current_sample + np.random.normal(0, args.proposal_std, current_sample.shape[0])
        proposed_probability = target_pdf(proposed_sample)

        # Apply Metropolis-Hastings acceptance criterion
        prob_ratio = proposed_probability / current_probability
        mu = np.random.uniform(0, 1)
        if mu <= prob_ratio:
            current_sample = proposed_sample
            current_probability = proposed_probability

            num_accepted += 1

        if step % args.subsample_frequency == 0:
            samples.append(current_sample)

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
