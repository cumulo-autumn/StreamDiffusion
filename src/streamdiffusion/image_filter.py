import random

import torch


class SimilarImageFilter:
    def __init__(self, threshold: float = 0.98):
        self.threshold = threshold
        self.prev_tensor = None
        self.cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)

    def __call__(self, x: torch.Tensor):
        if self.prev_tensor is None:
            self.prev_tensor = x.detach().clone()
            return x
        else:
            cos_sim = self.cos(self.prev_tensor.reshape(-1), x.reshape(-1)).item()
            sample = random.uniform(0, 1)
            if self.threshold >= 1:
                skip_prob = 0
            else:
                skip_prob = max(0, 1 - (1 - cos_sim) / (1 - self.threshold))

            # not skip frame
            if skip_prob < sample:
                self.prev_tensor = x.detach().clone()
                return x
            # skip frame
            else:
                return None

    def set_threshold(self, threshold: float):
        self.threshold = threshold
