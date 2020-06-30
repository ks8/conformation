import torch
import numpy as np
import os


def evaluate(model, args):
    """
    Generate samples from trained normalizing flow.
    :param model: PyTorch model.
    :param args: System parameters.
    :return: None.
    """
    os.makedirs(os.path.join(args.save_dir, "samples"))
    with torch.no_grad():
        model.eval()
        num_atoms = args.num_atoms
        for j in range(args.num_test_samples):
            sample = model.sample(args.num_layers)
            distmat = torch.zeros([num_atoms, num_atoms])
            indices = []
            for m in range(num_atoms):
                for n in range(1, num_atoms):
                    if n > m:
                        indices.append((m, n))
            for i in range(len(sample)):
                distmat[indices[i][0], indices[i][1]] = sample[i].item()
                distmat[indices[i][1], indices[i][0]] = distmat[indices[i][0], indices[i][1]]
            np.savetxt(os.path.join(args.save_dir, "samples", "distmat-" + str(j) + ".txt"), distmat)
