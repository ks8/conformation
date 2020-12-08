""" Run basic NF sampling. """
import math
import numpy as np
import os
import scipy.stats

# noinspection PyPackageRequirements
from tap import Tap
import torch
from tqdm import tqdm

from conformation.utils import load_checkpoint, param_count


class Args(Tap):
    """
    System arguments.
    """
    checkpoint_path: str  # Path to saved model checkpoint file
    mcmc: bool = False  # Whether or not to do MCMC-driven sampling
    num_layers: int = 10  # Number of RealNVP layers
    num_samples: int = 1000  # Number of samples to attempt
    proposal_std: float = 0.1  # Isotropic MCMC proposal std
    subsample_frequency: int = 100  # Subsample frequency
    save_dir: str = None  # Save directory
    cuda: bool = False  # Whether or not to use cuda
    gpu_device: int = 0  # Which GPU to use (0 or 1)


def funnel_pdf(x: np.ndarray) -> float:
    """
    Compute PDF value of x under the funnel distribution.
    :param x: Sample to evaluate.
    :return: PDF value.
    """
    pdf = scipy.stats.multivariate_normal(0, 3).pdf(x[0])
    for i in range(1, x.shape[0]):
        pdf *= scipy.stats.multivariate_normal(0, math.exp(x[0] / 2)).pdf(x[i])
    return pdf


def run_basic_nf_sampling(args: Args) -> None:
    """
    Perform neural network training.
    :param args: System parameters.
    :return: None.
    """

    os.makedirs(args.save_dir)
    args.cuda = torch.cuda.is_available()

    print(args)

    # Load model
    if args.checkpoint_path is not None:
        print('Loading model from {}'.format(args.checkpoint_path))
        model = load_checkpoint(args.checkpoint_path, args.cuda, args.gpu_device)

        print(model)
        print('Number of parameters = {:,}'.format(param_count(model)))

        if args.cuda:
            # noinspection PyUnresolvedReferences
            with torch.cuda.device(args.gpu_device):
                print('Moving model to cuda')
                model = model.cuda()

        with torch.no_grad():
            model.eval()

            if args.mcmc:
                num_accepted = 0
                samples = []
                rv = scipy.stats.multivariate_normal(mean=np.zeros(10), cov=np.ones(10))
                current_base_sample = np.array([rv.rvs(1)])
                z = torch.from_numpy(current_base_sample)
                z = z.type(torch.float32)
                z = z.cuda()
                current_target_sample, log_det = model.forward_pass_with_log_abs_det_jacobian(z)
                log_det = [math.exp(i) for i in log_det]
                current_target_sample = current_target_sample.cpu().numpy()
                current_target_probability = funnel_pdf(current_target_sample[0]) * np.prod(log_det)
                samples.append(current_target_sample[0])
                for step in tqdm(range(args.num_samples)):
                    # proposed_base_sample = np.array([current_base_sample[0] +
                    #                                  np.random.normal(0, args.proposal_std, 10)])
                    proposed_base_sample = np.array([scipy.stats.multivariate_normal(mean=np.zeros(10),
                                                                                     cov=np.ones(10)).rvs(1)])

                    z = torch.from_numpy(proposed_base_sample)
                    z = z.type(torch.float32)
                    z = z.cuda()
                    proposed_target_sample, log_det = model.forward_pass_with_log_abs_det_jacobian(z)
                    log_det = [math.exp(i) for i in log_det]
                    proposed_target_sample = proposed_target_sample.cpu().numpy()
                    proposed_target_probability = funnel_pdf(proposed_target_sample[0]) * np.prod(log_det)

                    # prob_ratio = proposed_target_probability / current_target_probability
                    prob_ratio = 1
                    mu = np.random.uniform(0, 1)
                    if mu <= prob_ratio:
                        current_base_sample = proposed_base_sample
                        current_target_sample = proposed_target_sample
                        current_target_probability = proposed_target_probability

                        num_accepted += 1

                    if step % args.subsample_frequency == 0:
                        samples.append(current_target_sample[0])

                samples = np.array(samples)
                import matplotlib.pyplot as plt
                plt.plot(samples[:, 1], samples[:, 0], 'bo', markersize=2)
                plt.xlim((-20, 20))
                plt.ylim((-10, 10))
                plt.savefig("test5.png")
                np.save(os.path.join(args.save_dir, "samples.npy"), samples)
                print(num_accepted / args.num_samples)

            else:
                for j in tqdm(range(args.num_samples)):
                    gen_sample = model.sample(args.num_layers)
                    gen_sample = gen_sample.cpu().numpy()
                    # noinspection PyTypeChecker
                    np.save(os.path.join(args.save_dir, "sample_" + str(j)), gen_sample)

    else:
        print('Must specify a model to load.')
        exit()
