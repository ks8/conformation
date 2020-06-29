import torch
import numpy as np
import os


def evaluate(model, num_test_samples, sample_layers):
    """
    Function for plotting samples of the trained distribution versus the target distribution.
    model: Trained NormalizingFlowModel
    num_test_samples: Number of samples to generate
    sample_layers: How many transformations to apply from the model in order
    """
    os.makedirs("nn-test")
    os.makedirs(os.path.join("nn-test", "distmat"))
    with torch.no_grad():
        model.eval()
        num_atoms = 8
        for j in range(num_test_samples):
            sample = model.sample(sample_layers)
            distmat = torch.zeros([num_atoms, num_atoms])
            indices = []
            for m in range(num_atoms):
                for n in range(1, num_atoms):
                    if n > m:
                        indices.append((m, n))
            for i in range(len(sample)):
                distmat[indices[i][0], indices[i][1]] = sample[i].item()/10.0
                distmat[indices[i][1], indices[i][0]] = distmat[indices[i][0], indices[i][1]]
            np.savetxt(os.path.join("nn-test", "distmat", "distmat-" + str(j) + ".txt"), distmat)



