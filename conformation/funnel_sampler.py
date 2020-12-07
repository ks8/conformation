""" Samples from Neal's funnel distribution """
import math
import numpy as np
import os

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


def funnel_sampler(args: Args):
    """
    Samples from Neal's funnel distribution.
    :param args: System args.
    :return:
    """
    os.makedirs(args.save_dir)

    for i in tqdm(range(args.num_samples)):
        s = []
        y = np.random.normal(0, 3)
        s.append(y)
        for _ in range(args.num_x_vars):
            s.append(np.random.normal(0, math.exp(y/2)))
        np.save(os.path.join(args.save_dir, "funnel_samples_" + str(i) + ".npy"), s)
