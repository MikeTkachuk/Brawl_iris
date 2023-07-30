from typing import Dict, Any
import torch


class Compose:
    def __init__(self, optimizers: Dict[str, torch.optim.Optimizer]):
        self.optimizers = optimizers

    def step(self, closure=None):
        for optimizer in self.optimizers.values():
            optimizer.step()

    def zero_grad(self):
        for optimizer in self.optimizers.values():
            optimizer.zero_grad()

    def reset(self, name, optimizer):
        self.optimizers[name] = optimizer

    def state_dict(self):
        return {name: optimizer.state_dict() for name, optimizer in self.optimizers.items()}

    def load_state_dict(self, state_dict):
        for name, state in state_dict.items():
            self.optimizers[name].load_state_dict(state)


class ComposeScheduler:
    def __init__(self, optimizer: Compose, schedulers: Dict[str, Any]):
        assert set(optimizer.optimizers) == set(schedulers)
        self.schedulers = {k: sch(optimizer.optimizers[k]) for k, sch in schedulers.items()}
        self.optimizer = optimizer

    def step(self, epoch=None):
        for sch in self.schedulers.values():
            sch.step(epoch)

    def reset(self, name, scheduler, preserve_step=True):
        last_step = self.schedulers[name].last_epoch
        self.schedulers[name] = scheduler(self.optimizer.optimizers[name])
        if preserve_step:
            self.schedulers[name].last_epoch = last_step

    def get_last_lr(self):
        return [s.get_last_lr() for s in self.schedulers.values()]
