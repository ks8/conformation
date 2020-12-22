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
    hmc: bool = False  # Whether or not to do HMC-driven sampling
    conditional: bool = False  # Whether or not to use a conditional normalizing flow
    conditional_concat: bool = False  # Whether or not to use conditional concat NF
    condition_path: str = None  # Path to condition numpy file for conditional normalizing flow
    epsilon: float = 0.25  # Leapfrog step size
    L: int = 20  # Number of leapfrog steps
    num_layers: int = 10  # Number of RealNVP layers
    num_samples: int = 1000  # Number of samples
    base_dim: int = 10  # Dimension of the base distribution
    proposal_std: float = 0.1  # Isotropic MCMC proposal std
    target_distribution: Literal["funnel", "perturbed_funnel", "gmm"] = "funnel"  # Target for MCMC sampling
    subsample_frequency: int = 100  # Subsample frequency
    log_frequency: int = 1000  # Log frequency
    cuda: bool = False  # Whether or not to use cuda
    gpu_device: int = 0  # Which GPU to use (0 or 1)
    save_dir: str = None  # Save directory


def normal_pdf_pytorch(x: float, mean: float, std: float) -> torch.Tensor:
    """
    Compute PDF value of normal distribution.
    :param x: Input value.
    :param mean: Mean of distribution.
    :param std: Standard deviation of distribution.
    :return: PDF value.
    """
    return (1./(std*torch.sqrt(torch.tensor([2*math.pi]))))*torch.exp(torch.tensor([-0.5*((x - mean)/std)**2]))


def neg_log_pdf_funnel_pytorch(x: torch.tensor) -> torch.Tensor:
    """
    Compute negative log of the pdf value of x under the funnel distribution.
    :param x: Sample to evaluate.
    :return: Negative log value.
    """
    neg_log_pdf = normal_pdf_pytorch(x[0], 0, 3)
    for i in range(1, x.size()[0]):
        neg_log_pdf *= normal_pdf_pytorch(x[i], 0, torch.exp(torch.tensor([x[0] / 2])).item())
    neg_log_pdf = -1.*torch.log(neg_log_pdf)
    return neg_log_pdf


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
                device = torch.device(args.gpu_device)
        else:
            device = torch.device('cpu')

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
                if args.conditional:
                    condition = np.load(args.condition_path)
                    condition = torch.from_numpy(condition)
                    condition = condition.type(torch.float32)
                    condition = condition.cuda()
                    u = model.output_layer(model.linear_layer(condition))
                    u = u.cpu().numpy()
                    rv = scipy.stats.multivariate_normal(mean=u, cov=np.ones(args.base_dim))
                else:
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
                if args.conditional_concat:
                    condition = np.load(args.condition_path)
                    condition = torch.from_numpy(condition)
                    condition = condition.type(torch.float32)
                    condition = condition.cuda()
                    current_target_sample, log_det = model.forward_pass_with_log_abs_det_jacobian(z, condition.unsqueeze(0))
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
                                                     np.random.normal(0, args.proposal_std, args.base_dim)])
                    z = torch.from_numpy(proposed_base_sample).type(torch.float32)
                    if args.cuda:
                        # noinspection PyUnresolvedReferences
                        with torch.cuda.device(args.gpu_device):
                            z = z.cuda()

                    # Transform the sample to one from the target space and compute the list of absolute value of
                    # Jacobian determinants
                    if args.conditional_concat:
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
                debug(f'% Moves Accepted: {num_accepted / args.num_samples}')

                # Save samples
                samples = np.array(samples)
                np.save(os.path.join(args.save_dir, "samples.npy"), samples)

        elif args.hmc:
            debug("Starting HMC sampling...")
            model.eval()

            for param in model.parameters():
                param.requires_grad = False

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

            # Compute the energy under the target distribution
            current_target_sample = current_target_sample.detach().cpu().numpy()
            current_target_energy = -1.*np.log(funnel_pdf(current_target_sample[0]) * np.prod(log_det))

            samples.append(current_target_sample[0])

            for step in tqdm(range(args.num_samples)):
                model.zero_grad()
                # Set the current position variables
                q = current_base_sample[0]

                # Generate random momentum values
                p = np.random.multivariate_normal(np.zeros(len(q)), np.identity(len(q)))
                current_p = p

                z = torch.from_numpy(current_base_sample).type(torch.float32)
                if args.cuda:
                    # noinspection PyUnresolvedReferences
                    with torch.cuda.device(args.gpu_device):
                        z = z.cuda()
                z.requires_grad = True

                # Transform the sample to one from the target space and compute the list of absolute value of
                # Jacobian determinants
                current_target_sample, log_det = model.forward_pass_with_log_abs_det_jacobian(z)
                log_det_prod = log_det[0]
                for i in range(1, len(log_det)):
                    log_det_prod = torch.dot(log_det_prod, log_det[i]).unsqueeze(0)

                test = neg_log_pdf_funnel_pytorch(current_target_sample[0])
                test.backward()
                print(z.grad)
                exit()

                # current_target_sample, log_det = model.forward_pass_with_log_abs_det_jacobian(z)
                # a = torch.prod(torch.tensor(log_det))
                # a.backward()
                # exit()
                # current_target_energy = neg_log_pdf_funnel_pytorch(current_target_sample[0])*torch.prod(torch.tensor(log_det))
                # current_target_energy.backward()


            #
            #     # Make a half step for momentum at the beginning
            #     p = p - args.epsilon * gradient_function(jnp.array(q)) / 2.
            #
            #     # Alternate full steps for position and momentum
            #     for i in range(L):
            #         # Make a full step for the position
            #         q = q + epsilon * p
            #
            #         # Make a full step for the momentum, except at the end of the trajectory
            #         if i != L - 1:
            #             p = p - epsilon * gradient_function(jnp.array(q))
            #
            #     # Make a half step for momentum at the end
            #     p = p - epsilon * gradient_function(jnp.array(q)) / 2.
            #
            #     # Negate the momentum at the end of the trajectory to make the proposal symmetric
            #     p *= -1.0
            #
            #     # Evaluate potential and kinetic energies at start and end of the trajectory
            #     current_u = energy_function(current_q)
            #     current_k = np.dot(current_p, current_p) / 2.
            #     proposed_u = energy_function(q)
            #     proposed_k = np.dot(p, p) / 2.
            #
            #     # Apply the Metropolis-Hastings criterion
            #     prob_ratio = math.exp(current_u - proposed_u + current_k - proposed_k)
            #     mu = random.uniform(0, 1)
            #     accepted = False
            #     if mu <= prob_ratio:
            #         current_q = q
            #         accepted = True

        else:
            debug("Starting NF sampling...")
            with torch.no_grad():
                model.eval()
                samples = []
                for _ in tqdm(range(args.num_samples)):
                    if args.conditional or args.conditional_concat:
                        gen_sample = model.sample(args.num_layers, args.condition_path, device, args.conditional_concat)
                    else:
                        gen_sample = model.sample(args.num_layers)
                    samples.append(gen_sample.cpu().numpy())
                samples = np.array(samples)
                np.save(os.path.join(args.save_dir, "samples.npy"), samples)

    else:
        print('Must specify a model to load.')
        exit()
