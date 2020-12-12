""" Run basic HMC sampling. """
from logging import Logger
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import time
from typing import Tuple
from typing_extensions import Literal

# noinspection PyPackageRequirements
from jax import grad
# noinspection PyPackageRequirements
import jax.numpy as jnp
# noinspection PyPackageRequirements
from tap import Tap
from tqdm import tqdm


class Args(Tap):
    """
    System arguments.
    """
    num_steps: int = 100  # Number of HMC steps
    target_distribution: Literal["normal", "funnel"] = "normal"  # Target distribution
    num_x_vars_funnel: int = 9  # Number of x variables for funnel distribution
    epsilon: float = 0.25  # Leapfrog step size
    L: int = 20  # Number of leapfrog steps
    subsample_frequency: int = 1  # Subsample frequency
    log_frequency: int = 10  # Log frequency
    save_dir: str = None  # Save directory


def neg_log_pdf_normal(q: jnp.ndarray) -> float:
    """
    Function for computing the negative log of the bivariate normal pdf specified in Neal's basic HMC example.
    :param q: Array of points.
    :return: Negative log of the pdf at the input point.
    """
    cov = jnp.array([[1.0, 0.98], [0.98, 1.0]])
    cov_inv = jnp.linalg.inv(cov)
    neg_log_pdf = -1.*jnp.log(jnp.exp(
        -0.5*jnp.matmul(q, jnp.matmul(cov_inv, q)))/(jnp.sqrt((2.*jnp.pi)**2)*jnp.linalg.det(cov)))
    return neg_log_pdf


def normal_pdf(x: float, mean: float, std: float) -> float:
    """
    Compute PDF value of normal distribution.
    :param x: Input value.
    :param mean: Mean of distribution.
    :param std: Standard deviation of distribution.
    :return: PDF value.
    """
    return (1./(std*jnp.sqrt(2*jnp.pi)))*jnp.exp(-0.5*((x - mean)/std)**2)


def neg_log_pdf_funnel(x: jnp.ndarray) -> float:
    """
    Compute negative log of the pdf value of x under the funnel distribution.
    :param x: Sample to evaluate.
    :return: Negative log value.
    """
    neg_log_pdf = normal_pdf(x[0], 0, 3)
    for i in range(1, x.shape[0]):
        neg_log_pdf *= normal_pdf(x[i], 0, jnp.exp(x[0] / 2))
    neg_log_pdf = -1.*jnp.log(neg_log_pdf)
    return neg_log_pdf


def funnel_sample() -> np.ndarray:
    """
    Sample from the funnel distribution.
    :return:
    """
    sample = []
    y = np.random.normal(0, 3)
    sample.append(y)
    for _ in range(1):
        sample.append(np.random.normal(0, math.exp(y / 2)))
    sample = np.array(sample)

    return sample


def hmc_step(current_q: np.ndarray, energy_function, gradient_function, epsilon: float, L: int) -> \
        Tuple[bool, np.ndarray]:
    """
    Run a single Hamiltonian Monte Carlo step.
    :param current_q: Current point.
    :param energy_function: Function that gives the negative log of the pdf.
    :param gradient_function: Gradient of the energy function.
    :param epsilon: Leapfrog step size.
    :param L: Number of leapfrog steps.
    :return: Whether or not trial move is accepted and the updated position.
    """
    # Set the current position variables
    q = current_q

    # Generate random momentum values
    p = np.random.multivariate_normal(np.zeros(len(q)), np.identity(len(q)))
    current_p = p

    # Make a half step for momentum at the beginning
    p = p - epsilon * gradient_function(jnp.array(q)) / 2.

    # Alternate full steps for position and momentum
    for i in range(L):
        # Make a full step for the position
        q = q + epsilon * p

        # Make a full step for the momentum, except at the end of the trajectory
        if i != L - 1:
            p = p - epsilon * gradient_function(jnp.array(q))

    # Make a half step for momentum at the end
    p = p - epsilon * gradient_function(jnp.array(q)) / 2.

    # Negate the momentum at the end of the trajectory to make the proposal symmetric
    p *= -1.0

    # Evaluate potential and kinetic energies at start and end of the trajectory
    current_u = energy_function(current_q)
    current_k = np.dot(current_p, current_p) / 2.
    proposed_u = energy_function(q)
    proposed_k = np.dot(p, p) / 2.

    # Apply the Metropolis-Hastings criterion
    prob_ratio = math.exp(current_u - proposed_u + current_k - proposed_k)
    mu = random.uniform(0, 1)
    accepted = False
    if mu <= prob_ratio:
        current_q = q
        accepted = True

    return accepted, current_q


def basic_hmc(args: Args, logger: Logger) -> None:
    """
    Perform basic HMC sampling given a specified target distribution.
    :param args: System parameters.
    :param logger: System logger.
    :return: None.
    """
    # Set up logger
    debug, info = logger.debug, logger.info

    debug("Starting HMC search...")
    # # Specify the target distribution pdf function
    if args.target_distribution == "normal":
        energy_function = neg_log_pdf_normal
        gradient_function = grad(energy_function)
        current_q = np.random.multivariate_normal(np.zeros(2), np.array([[1.0, 0.98], [0.98, 1.0]]))
    elif args.target_distribution == "funnel":
        debug("Coming soon...")
        energy_function = neg_log_pdf_funnel
        gradient_function = grad(energy_function)
        current_q = funnel_sample()
    else:
        exit()

    # noinspection PyUnboundLocalVariable
    samples = [current_q]

    debug(f'Running HMC steps...')
    start_time = time.time()
    num_accepted = 0
    for step in tqdm(range(args.num_steps)):
        # noinspection PyUnboundLocalVariable
        accepted, current_q = hmc_step(current_q, energy_function, gradient_function, args.epsilon, args.L)
        if accepted:
            num_accepted += 1

        if step % args.subsample_frequency == 0:
            samples.append(current_q)

        if step % args.log_frequency == 0:
            debug(f'% Moves accepted: {float(num_accepted) / float(step + 1) * 100.0}')
    end_time = time.time()
    debug(f'Total Time(s): {end_time - start_time}')
    debug(f'% Moves accepted: {float(num_accepted) / float(args.num_steps) * 100.0}')

    # Save samples
    samples = np.array(samples)
    np.save(os.path.join(args.save_dir, "samples.npy"), samples)

    samples = np.array(samples)
    plt.plot(samples[:, 1], samples[:, 0], 'bo', markersize=2)
    plt.savefig("test_joint")
    plt.clf()
    plt.plot(samples[:, 0], 'bo')
    plt.savefig("test_marginal")
    plt.clf()
