import numpy as np
from tensor import Tensor

class Optimizer:
  def __init__(self, params):
    self.grads  = {}
    self.params = [p for p in params if p.requires_grad]
  def zero_grad(self):
    for param in self.params: 
      param.grad = None

class SGD(Optimizer):
  def __init__(self, params, lr=0.001):
    super().__init__(params)
    self.lr = lr

  def step(self):
    for p in self.params:
      p -= p.grad * self.lr

class Adam(Optimizer):
  def __init__(self, params, lr=0.001, b1=0.9, b2=0.999, eps=1e-8):
    super().__init__(params)
    self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
    self.t = 0

    #### Momentum 
    self.grads["m"] = [ Tensor.zeros(p.shape,requires_grad=False) for p in self.params ]
    self.grads["v"] = [ Tensor.zeros(p.shape,requires_grad=False) for p in self.params ]

  def step(self):
    self.t += 1
    a = self.lr * ((1.0 - self.b2**self.t)**0.5) / (1.0 - self.b1**self.t)

    for i, p in enumerate(self.params):
      self.grads["m"][i] = self.b1 * self.grads["m"][i] + (1.0 - self.b1) * p.grad
      self.grads["v"][i] = self.b2 * self.grads["v"][i] + (1.0 - self.b2) * p.grad * p.grad
      p -= a * self.grads["m"][i].div(self.grads["v"][i].sqrt() + self.eps)
