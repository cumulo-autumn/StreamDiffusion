import random

import torch


class SimilarImageFilter:
    def __init__(self, threshold: float = 0.95, threshold_scale: float = 7):
        self.threshold_scale = threshold_scale
        self.threshold = threshold
        self.prev_tensor = None
        self.cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)

    def __call__(self, x: torch.Tensor):
        if self.prev_tensor is None:
            self.prev_tensor = x.detach().clone()
            return x, 0
        else:
            output = self.cos(self.prev_tensor.reshape(-1), x.reshape(-1))
            output = ((output + 1) / 2) ** self.threshold_scale
            self.prev_tensor = x.detach().clone()
            if output.item() > self.threshold:
                sample = abs(random.gauss(0, 1))
                skip_prob = 1 - (1 - output.item()) / (1 - self.threshold)
                if skip_prob < sample:
                    return x, skip_prob
                return None, skip_prob
            else:
                return x, 0

    def set_threshold(self, threshold: float):
        self.threshold = threshold
