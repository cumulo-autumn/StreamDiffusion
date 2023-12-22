from typing import Optional
import random

import torch


class SimilarImageFilter:
    def __init__(self, threshold: float = 0.98, max_skip_frame: float = 10) -> None:
        self.threshold = threshold
        self.prev_tensor = None
        self.cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
        self.max_skip_frame = max_skip_frame
        self.skip_count = 0

    def __call__(self, x: torch.Tensor) -> Optional[torch.Tensor]:
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
                if self.skip_count > self.max_skip_frame:
                    self.skip_count = 0
                    self.prev_tensor = x.detach().clone()
                    return x
                else:
                    self.skip_count += 1
                    return None

    def set_threshold(self, threshold: float) -> None:
        self.threshold = threshold
    
    def set_max_skip_frame(self, max_skip_frame: float) -> None:
        self.max_skip_frame = max_skip_frame
