
import torchmetrics
import torch

class RootMeanSquaredError(torchmetrics.Metric):
    def __init__(self):
        super().__init__()
        self.add_state("sum_squared_error", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("num_samples", default=torch.tensor(0), dist_reduce_fx="sum")
    
    def update(self, predicted, target):
        squared_error = (predicted - target) ** 2
        self.sum_squared_error += torch.sum(squared_error)
        self.num_samples += squared_error.numel()

    def compute(self):
        return torch.sqrt(self.sum_squared_error / self.num_samples)
