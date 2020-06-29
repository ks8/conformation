

def loss_func(z, log_jacobians, base_dist):
    """
    Loss function that computes the mean log probability of training example by computing the log probability of its
    corresponding latent variable and the sum of the log abs det jacobians of the normalizing flow transformations.
    :param z:
    :param log_jacobians:
    :param base_dist:
    :return: Average loss
    """

    return -(base_dist.log_prob(z) - sum(log_jacobians)).mean()
