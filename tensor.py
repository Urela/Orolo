
from __future__ import annotations
from mdarray import MDArray, ManipulationOps, ElementwiseOps, ReduceOps, Machine
import functools
import numpy as np
from math import prod

class Tensor:
  def __init__(self, data, machine=Machine.DEFAULT, requires_grad=True):
    if isinstance(data, (list, int, float, tuple)):
        data = np.array(data, dtype=np.float32)
    if isinstance(data, (np.uint8, np.float32,  np.float16)):
        data = np.array(data, dtype=np.float32)
    if isinstance(data, np.ndarray): 
      data = MDArray.load( data )
    if isinstance(data, MDArray):
      self.data = data
    else: raise Exception(f"Can't create Tensor from {data} of type {type(data)}")

    self.grad = None
    self.machine = machine
    self.ctx : Optional[Function] = None # used for the autograd graph construction
    self.requires_grad : Optional[bool] = requires_grad
  # --------- Operator Overloading --------- 
  def __repr__(self): return f"<Tensor\n {self.data} \nwith grad\n {self.grad}>"

  def __getitem__(self, index): return Tensor( self.data[index] )

  def __setitem__(self,index,value): self.data[index] = value

  # --------- Properties --------- 
  def compute(self): return self.data.compute()
  @property
  def shape(self): return self.data.shape

  def numpy(self): return np.array(self.data.compute())

  def assign(self, x): 
    if not isinstance(x, Tensor): 
      x = Tensor(x)
    assert self.shape == x.shape
    self.data = x.data
    return x

  def zero_grad(self):
    self.grad = None
    self._ctx = None
  # --------- backwards --------- 
  def topo_sort(self, node:Tensor, visited:set, nodes:list):
    if node not in visited:
      visited.add(node)
      if node.ctx is not None:
        for p in node.ctx.parents: self.topo_sort(p,visited, nodes)
        nodes.append(node)
    return nodes

  def backward(self):
    nodes = self.topo_sort(self, set(), []) #returns our topo sort DAG
    # intial grads as we are explicitly creating gradients
    self.grad = Tensor.ones(self.shape, machine=Machine.DEFAULT, requires_grad=False)

    # looop through our sorted graph and update gradients
    for node in nodes[::-1]:
      if not any(p.requires_grad for p in node.ctx.parents): continue
      assert (node.grad is not None)

      #collect grads created by each operation when applied to a tensor
      grads = node.ctx.backward(node.grad.data)
      if len(node.ctx.parents) == 1:
        grads = [grads]

      # udpated gradients
      for parent, grad in zip(node.ctx.parents, grads):
        if grad is None: continue
        parent.grad = Tensor(grad) if parent.grad is None else (parent.grad + Tensor(grad))
    pass

  # --------- Data initialization types --------- 
  @classmethod
  def ones(Tensor, *shape, **kwargs): 
    return Tensor(np.ones(*shape, dtype=np.float32), **kwargs)

  @classmethod
  def zeros(Tensor, *shape, **kwargs): 
    return Tensor(np.zeros(*shape, dtype=np.float32), **kwargs)

  @classmethod
  def rand(Tensor, *shape, **kwargs): 
    return Tensor(np.random.randn(*shape).astype(np.float32), **kwargs)

  @classmethod 
  def uniform(Tensor, *shape, **kwargs):
    return Tensor((np.random.uniform(-1.,1., size=shape)/np.sqrt(np.prod(shape))).astype(np.float32), **kwargs)

  # --------- operations --------- 
  #https://github.com/geohot/tinygrad/
  def transpose(self, order=(1,0)): return self.permute(order=order)
  def reshape(self, shape): return self._reshape(new_shape=shape)
  def expand(self, shape): return self._expand(shape=shape)

  def conv2d(self, other, bias=None, **kwargs): 
    ret = self._conv2d(other, **kwargs)
    return ret if bias is None else ret.add(bias.reshape(shape=[1,-1,1,1]))

  def matmul(a:Tensor, b:Tensor): #TODO
    # NOTE: we use a 1x1 conv2d to do the matmul. mxk @ kxn = (1,k,m,1).conv2d(n,k,1,1)

    #determine the number of matrices in each tensor
    bs, groups = prod(a.shape[0:-2]), prod(b.shape[0:-2]) #batch size, groups of matrices, (channels and kernals)
    cin, cout = b.shape[-2], b.shape[-1]

    #ret_shape_flat = tuple(list(a.shape[0:-2]) + [cout,-1]) # why shape created like this???
    ret_shape_t = tuple([*a.shape[:-2], cout,-1]) # why shape created like this???
    if len(a.shape) > 1: aorder = tuple(list(range(len(a.shape[:-2])))+[len(a.shape)-1, len(a.shape[:-2])])
    else: aorder, ret_shape_t = (0,), (cout,)

    border = tuple(list(range(len(b.shape[0:-2])))+[len(b.shape)-1,len(b.shape)-2])

    
    # what ??
    ca = a.transpose(order=aorder).reshape(shape=(bs//groups, cin*groups, -1,1))
    cb = b.transpose(order=border).reshape(shape=(groups*cout, cin, 1,1))
    return ca.conv2d(cb, groups=groups).reshape(shape=ret_shape_t).transpose(order=aorder)

  def _reduce_axis(self, axis):
    if axis is None: axis = range(len(self.shape))
    if isinstance(axis, int): axis = [axis]
    axis = tuple([ x if x >= 0 else x+len(self.shape) for x in axis]) #?? if negative value flip other way
    shape = [self.shape[i] for i in range(len(self.shape)) if i not in axis]  #####  the axis we are reducing by is removed so create a new shape
    shape = [1] if shape == [] else shape  # looks prettier this way, I assume
    return axis, shape

  def sum(self, axis=None, keepdim=False): 
    axis, out_shape = self._reduce_axis(axis) 
    ret = self._sum(axis=axis)
    return ret if keepdim or ret.shape == out_shape else ret.reshape(shape=out_shape)
  
  def max(self, axis=None, keepdim=False): 
    axis, out_shape = self._reduce_axis(axis) 
    ret = self._max(axis=axis)
    return ret if keepdim or ret.shape == out_shape else ret.reshape(shape=out_shape)

  def mean(self, axis=None, keepdim=False): 
    out = self.sum(axis=axis, keepdim=keepdim)
    return out *(prod(out.shape) / prod(self.shape))


  def _softmax(self): 
    max_x = self - self.max(axis=len(self.shape)-1, keepdim=True)
    exp_x = max_x.exp()
    return max_x, exp_x, exp_x.sum(axis=len(self.shape)-1, keepdim=True)

  def softmax(self):
    max_x, exp_x, ss =  self._softmax()
    return exp_x.div(ss)

  def logsoftmax(self):
    max_x, exp_x, ss =  self._softmax()
    return max_x - ss.log()

  # ***** broadcasted binary ops *****

  @staticmethod
  def broadcasted(fxn, x, y):
    tt = [arg for arg in [x,y] if isinstance(arg, Tensor)][0]  # this is the prototype tensor
    if not isinstance(x, Tensor): x = Tensor([x], machine=tt.machine, requires_grad=False) 
    if not isinstance(y, Tensor): y = Tensor([y], machine=tt.machine, requires_grad=False) 

    n_dims = max(len(x.shape), len(y.shape))
    if len(x.shape) != n_dims: x = x.reshape(list(x.shape) + [1]*(n_dims-len(x.shape)))
    if len(y.shape) != n_dims: y = y.reshape(list(y.shape) + [1]*(n_dims-len(y.shape)))

    shape_ret = tuple([max(sx, sy) for sx,sy in zip(x.shape, y.shape)])
    if x.shape != shape_ret: x = x.expand(shape_ret)
    if y.shape != shape_ret: y = y.expand(shape_ret)
    return fxn(x, y)

  # TODO: are these the only ones that can take number arguments?
  def add(self, x): return Tensor.broadcasted(Tensor._add, self, x)
  def sub(self, x): return Tensor.broadcasted(Tensor._sub, self, x)
  def mul(self, x): return Tensor.broadcasted(Tensor._mul, self, x)
  def pow(self, x): return Tensor.broadcasted(Tensor._pow, self, x)
  def sqrt(self): return self.pow(0.5)



class Function:  # operation as data
  def __init__(self, *tensors:Tensor):
    self.parents = tensors
    self.buffer: List[Tensor] = [] # saved tensors
    self.needs_input_grad = [p.requires_grad for p in self.parents]
    self.requires_grad = True if any(self.needs_input_grad) else (None if any(x is None for x in self.needs_input_grad) else False)

  def forward(self, *args, **kwargs): 
    raise NotImplementedError(f"forward() is not implementd for {self}")

  def backward(self, *args, **kwargs): 
    raise NotImplementedError(f"backward() is not implementd for {self}")

  @classmethod
  def apply(cls, *x:Tensor, **kwargs):
    tensors = [t if isinstance(t, Tensor) else Tensor(t) for t in x] # assert, all input as tensors
    ctx = cls(*tensors)

    ret = Tensor(ctx.forward(*[t.data for t in tensors], **kwargs), Machine.DEFAULT, ctx.requires_grad)
    ret.ctx = ctx if ctx.requires_grad else None # store context/Function for autograd 
    return ret
  
# ------ register the operations ------ 
''' set up each operation as attribute, this allows to register new operation at a later date'''

from ops import *


if __name__ == "__main__":
  #A = Tensor.uniform(1,3,3)
  #B = Tensor.uniform(1,3,3)

  #C = A.add(B)
  #C = C.mul(B)
  #C = C.relu() 
  #C = C.transpose((2,1,0))  # permute
  #C = C.reshape((1,1,9))  
  #C = C.sum((1,1,2))  
  #C.backward()
  #C = C.compute()

  DEBUG = True
  a = np.random.uniform(size=(2,2))
  b = np.random.uniform(size=(2,2))
  A,B = Tensor(a), Tensor(b)
  A = A.add(B)
  A = A.sub(B)
  print(type(A-B))
  print(type(A+B))
  print(type(A.__isub__(B)))
  print(type(A.__iadd__(B)))
  print(type(A.sub(B)))
  print(type(A.add(B)))

  A = A.mul(B)
  C = A.matmul(B)

  #print('A', A.compute(), '\nB', B.compute(), '\nret',C.compute())

  C.backward()
  C = C.compute()
