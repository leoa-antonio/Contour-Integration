import numpy as np
import torch
from torch.utils.data import Dataset
from stimuli import generate_stimulus

class ContourIntegrationDataset(Dataset):
    '''
    PyTorch dataset for procedurally generated contour-integration stimuli.

    Stimuli are generated with fixed step_length and value of beta. Labels are evenly split
    between contour present (1) and no countour present (0). Each image is reproducible from
    a seed and a label.
    '''

    def __init__(self, n_images, beta, step_length, seed=0):
        self.n_images = n_images
        self.beta = beta
        self.step_length = step_length
        self.seed = seed

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        label = idx % 2

        rng = np.random.default_rng(self.seed + idx)

        stimulus = generate_stimulus(
            rng=rng,
            beta=self.beta,
            step_length=self.step_length,
            contour_present=bool(label)
        )

        stimulus = torch.from_numpy(stimulus).float()
        stimulus = torch.unsqueeze(stimulus, 0)

        label = torch.tensor(label, dtype=torch.long)

        return stimulus, label
