import torch
import torch.nn as nn


class AffineTransform(nn.Module):
    """
    Performs a 2D affine transformation.
    """

    def __init__(self):
        super(AffineTransform, self).__init__()
        self.loc = nn.Parameter(torch.randn(1, 2, requires_grad=True))  # Two parameters for shifting
        self.scale = nn.Parameter(torch.randn(1, 2, requires_grad=True))  # Two parameters for scaling

    def forward(self, x):
        return self.loc + self.scale * x

    def inverse(self, y):
        return (y - self.loc) / self.scale

    def log_abs_det_jacobian(self, y):
        x = self._inverse(y)
        shape = x.shape
        scale = self.scale
        result = torch.abs(scale).log()  # dy/dx = scale for a given coordinate
        return torch.sum(result.expand(shape), axis=1)


class RealNVP(nn.Module):
    """
    Performs a single layer of the RealNVP flow.
    """

    def __init__(self, nets, nett, mask, prior):
        super(RealNVP, self).__init__()
        self.prior = prior
        self.mask = nn.Parameter(mask, requires_grad=False)
        self.t = nett
        self.s = nets

    def forward(self, z):
        """

        :param z:
        :return:
        """
        x = z
        x_ = x * self.mask
        s = self.s(x_) * (1 - self.mask)
        t = self.t(x_) * (1 - self.mask)
        x = x_ + (1 - self.mask) * (x * torch.exp(s) + t)
        return x

    def inverse(self, x):
        """

        :param x:
        :return:
        """
        log_det_j, z = x.new_zeros(x.shape[0]), x
        z_ = self.mask * z
        s = self.s(z_) * (1 - self.mask)
        t = self.t(z_) * (1 - self.mask)
        z = (1 - self.mask) * (z - t) * torch.exp(-s) + z_
        return z

    def log_abs_det_jacobian(self, x):
        """

        :param x:
        :return:
        """
        log_det_j, z = x.new_zeros(x.shape[0]), x
        z_ = self.mask * z
        s = self.s(z_) * (1 - self.mask)
        log_det_j += s.sum(dim=1)
        return log_det_j


class NormalizingFlowModel(nn.Module):
    """
    Normalizing flow class. The forward function computes the cumulative log absolute value of the determinant of the
    jacobians of the transformations, and the sample function samples from the flow starting at the base distribution.
    """

    def __init__(self, base_dist, biject):
        super(NormalizingFlowModel, self).__init__()
        self.biject = biject  # List of transformations, each of which is nn.Module
        self.base_dist = base_dist  # Base distribution
        self.bijectors = nn.ModuleList(self.biject)
        self.log_det = []

    def forward(self, x):
        self.log_det = []  # Accumulate the log abs det jacobians of the transformations
        for b in range(len(self.bijectors) - 1, -1, -1):
            self.log_det.append(self.bijectors[b].log_abs_det_jacobian(x))
            x = self.bijectors[b].inverse(x)
        return x, self.log_det

    def sample(self, sample_layers):
        x = self.base_dist.sample()
        for b in range(sample_layers):
            x = self.bijectors[b](x)  # Process a sample through the flow
        return x
