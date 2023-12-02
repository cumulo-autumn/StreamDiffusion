import torch


class SimilarImageFilter:
    def __init__(self, threshold: float = 0.95):
        self.threshold = threshold
        self.prev_tensor = None

    def __call__(self, x: torch.Tensor):
        if self.prev_tensor is None:
            self.prev_tensor = x.detach().clone()
            return x
        else:
            cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
            output = cos(self.prev_tensor.reshape(-1), x.reshape(-1))
            self.prev_tensor = x.detach().clone()
            if output.item() > self.threshold:
                return None
            else:
                return x

    def set_threshold(self, threshold: float):
        self.threshold = threshold
