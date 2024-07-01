import torch
import torch.nn as nn
import warnings
import copy

warnings.filterwarnings("ignore")
RUN_EXAMPLES = True


def show_example(fn, args=[]):
    if __name__ == "__main__" and RUN_EXAMPLES:
        return fn(*args)


def execute_example(fn, args=[]):
    if __name__ == "__main__" and RUN_EXAMPLES:
        fn(*args)


class DummyOptimizer(torch.optim.Optimizer):
    def __init__(self):
        self.param_groups = [{"lr": 0}]

    def step(self):
        return None

    def zero_grad(self, set_to_none=False):
        return None


class DummyScheduler:
    def step(self):
        return None


def clones(module, N):
    """Produces N identical layers."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
