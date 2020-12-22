""" Samples from Neal's funnel distribution """
import math
import numpy as np
import os
import scipy.stats

# noinspection PyPackageRequirements
from tap import Tap
from tqdm import tqdm


class Args(Tap):
    """
    System arguments.
    """
    num_samples: int = 10000  # Number of samples to generate
    num_x_vars: int = 9  # Number of x variables in the distribution
    save_dir: str  # Path to directory containing output files


def funnel_pdf(x: np.ndarray) -> float:
    """
    Compute PDF value of x under the funnel distribution.
    :param x: Sample to evaluate.
    :return: PDF value.
    """
    pdf = scipy.stats.norm(0, 3).pdf(x[0])
    for i in range(1, x.shape[0]):
        pdf *= scipy.stats.norm(0, math.exp(x[0] / 2)).pdf(x[i])
    return pdf


def perturbed_funnel_pdf(x: np.ndarray) -> float:
    """
    Compute PDF value of x under the funnel distribution.
    :param x: Sample to evaluate.
    :return: PDF value.
    """
    pdf = scipy.stats.norm(0, 1).pdf(x[0])
    for i in range(1, x.shape[0]):
        pdf *= scipy.stats.norm(0, math.exp(x[0])).pdf(x[i])
    return pdf


def funnel_sample(num_x_vars: int) -> np.ndarray:
    """
    Sample from the funnel distribution.
    :param num_x_vars: Number of x variables in the distribution.
    :return:
    """
    sample = []
    y = np.random.normal(0, 3)
    sample.append(y)
    for _ in range(num_x_vars):
        sample.append(np.random.normal(0, math.exp(y / 2)))
    sample = np.array(sample)

    return sample


def perturbed_funnel_sample(num_x_vars: int) -> np.ndarray:
    """
    Sample from the funnel distribution.
    :param num_x_vars: Number of x variables in the distribution.
    :return:
    """
    sample = []
    y = np.random.normal(0, 1)
    sample.append(y)
    for _ in range(num_x_vars):
        sample.append(np.random.normal(0, math.exp(y)))
    sample = np.array(sample)

    return sample


def funnel_sampler(args: Args):
    """
    Sampling from Neal's funnel distribution.
    :param args: System args.
    :return: None.
    """
    os.makedirs(args.save_dir)

    for i in tqdm(range(args.num_samples)):
        sample = funnel_sample(args.num_x_vars)
        np.save(os.path.join(args.save_dir, "funnel_samples_" + str(i) + ".npy"), sample)
