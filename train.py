# File for stuff
import numpy as np
from utils import loss
import os
import torch


def train(model, optimizer, batch_size, num_epochs, num_batch_iterations, input_folder):
    """
    Function for training a normalizing flow model.
    :param model: nn.Module neural network
    :param optimizer: PyTorch optimizer
    :param batch_size: Batch size
    :param num_epochs: Number of training epochs
    :param num_batch_iterations:
    :param input_folder:
    :return:
    """
    model.train()

    for _ in range(num_epochs):
        loss_sum = 0
        for t in range(num_batch_iterations):
            model.zero_grad()
            x_samples = [[] for _ in range(batch_size)]
            for i, j in enumerate(np.random.randint(0, 9990, batch_size)):
                distmat = np.loadtxt(os.path.join(input_folder, "distmat-" + str(j) + "-ethane.txt"))
                num_atoms = distmat.shape[0]
                for m in range(num_atoms):
                    for n in range(1, num_atoms):
                        if n > m:
                            x_samples[i].append(distmat[m][n])
            x_samples = torch.from_numpy(np.array(x_samples))
            x_samples = x_samples.type(torch.float32)
            x_samples = x_samples.cuda()
            z, log_jacobians = model(x_samples)
            test = loss(z, log_jacobians, model.base_dist)
            loss_sum += test.item()

            test.backward()
            optimizer.step()

            print('iter %s:' % t, 'loss = %.3f' % test)

