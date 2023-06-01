import numpy as np

class Optimizer:
    def __init__(self, params):
        self.params = [p for p in params if p.requires_grad == True]
    def zero_grad(self):
        for param in self.params:
            param.grad = None

class SGD(Optimizer):
    def __init__(self, params, lr=0.001):
        super(SGD, self).__init__(params)
        self.lr = lr
    def step(self):
        for p in self.params:
            p -= p.grad * self.lr

