from __future__ import annotations
from mdarray import MDArray, fixArgsType
from math import prod
import numpy as np

class Tensor:
  def __init__(self, data, device=None, requires_grad=True):
    if isinstance(data, (tuple,list,set)):
      if isinstance(data[0], (int, float)): data = np.array(data).astype(np.float32) 
      if isinstance(data[0], MDArray): data = np.array([x.evaluate().compute() for x in data ]).astype(np.float32)
    if isinstance(data, (int, float)):data = np.array(data).astype(np.float32) 
    if isinstance(data, np.ndarray): 
      data = MDArray.load(data.astype(np.float32), device)
    if isinstance(data, MDArray): self.data = data
    else: raise Exception(f"Can't create Tensor from {data} of type {type(data)}")

    self.grad: Optional[Tensor] = None
    self._ctx: Optional[Function] = None # used for the autograd graph construction
    self.requires_grad: Optional[bool] = requires_grad

  def backward(self):
    # generate the topological sort of the compute graph       
    def topo_sort(node:Tensor, visited:set, nodes:List[Tensor]):
      visited.add(node)
      if node._ctx is not None: 
        [topo_sort(node, visited, nodes) for node in node._ctx.parents if node not in visited]
        nodes.append(node)
      return nodes

    # this is "implicit gradient creation"
    self.grad = Tensor.ones(*self.shape, requires_grad=False)
    for node in reversed(topo_sort(self, set(), [])):
      if not any(p.requires_grad for p in node._ctx.parents): continue
      assert node.grad is not None, f'error with grad'

      # get grads created by the operation applied to the tensor
      grads = node._ctx.backward(node.grad.data) #get grads created by the operation to the tensor
      grads = [Tensor(grad,  requires_grad=False) if grad is not None else None for grad in ([grads] if len(node._ctx.parents) ==1 else grads)]
      
      # update gradients
      for parent, grad in zip(node._ctx.parents, grads):
        if grad is not None and parent.requires_grad:
          assert grad.shape == parent.shape, f"grad shape must match tensor shape in {self._ctx!r}, {grad.shape!r} != {parent.shape!r} {parent._ctx}"
          parent.grad = grad if parent.grad is None else (parent.grad + grad)
      del node._ctx #prune

  def __repr__(self): return f"<Tensor\n {self.data} \nwith grad\n {self.grad}>"
  # --------- Properties --------- 
  @property
  def shape(self): return self.data.shape
  @property
  def dtype(self): return self.data.shape
  def numpy(self): return self.data.compute()

  def assign(self, x): self.data = x.data
  def zero_grad(self): self.grad, self._ctx = None, None

  # ---------  Data initialization types --------- 

  @staticmethod
  def zeros(*shape, **kwargs) -> Tensor: return Tensor(np.zeros(*shape, dtype=np.float32), **kwargs)
  @staticmethod
  def rand(*shape, **kwargs) -> Tensor: return Tensor(np.random.randn(*shape).astype(np.float32), **kwargs)
  @staticmethod
  def uniform(*shape, **kwargs) -> Tensor: return Tensor((np.random.uniform(-1.,1., size=shape)/np.sqrt(np.prod(shape))).astype(np.float32), **kwargs)
  @staticmethod
  def ones(*shape, **kwargs) -> Tensor: return Tensor(np.ones(shape, dtype=np.float32), **kwargs)

  '''
  def matmul(A:Tensor, B:Tensor) -> Tensor:
    A = A.reshape(*A.shape[0:-1], 1, A.shape[-1]) # convert all matrix to vectors based on rows
    B = B.reshape(*B.shape[0:-2], 1, B.shape[-2], B.shape[-1]).transpose(-2,-1) # convert all matrix to vectors based on cols, then transpose
    return (A*B).sum(axis=-1).reshape(*A.shape[0:-2], -1)

  def _reduce(self, fxn:Type[Function], axis:Optional[Union[int, Tuple[int, ...]]]=None, keepdim=False):
    _axis: List[int] = tuple(range(len(self.shape))) if axis is None else ([axis] if isinstance(axis, int) else list(axis)) # type correction
    _axis = [x if x>=0 else x+len(self.shape) for x in _axis]  # clearn up by wrapping
    new_shape = [self.shape[i] for i in range(len(self.shape)) if i not in _axis]
    ret = fxn.apply(self, new_shape=tuple(1 if i in _axis else self.shape[i] for i in range(len(self.shape))))
    return ret if keepdim else ret.reshape(shape=[1] if new_shape == [] else new_shape )

  def sum(self, axis=None, keepdim=False): return self._reduce(ops.Sum(), axis, keepdim)
  def max(self, axis=None, keepdim=False): return self._reduce(ops.Max(), axis, keepdim)
  def mean(self, axis=None, keepdim=False): out = self.sum(axis, keepdim); return out * (prod(out.shape)/prod(self.shape))

  def log(self): return ops.Log.apply(self)
  def exp(self): return ops.Exp.apply(self)
  #def relu(self) -> Tensor: return self._broadcast(ops.Maximum, 0) 

  def expand(self, shape,  *args) -> Tensor: return ops.Expand.apply(self, shape=tuple(x if d!=-1 else s for d,x in zip(self.shape, fixArgsType(shape, *args))))
  def permute(self, shape, *args) -> Tensor: return ops.Permute.apply(self, order= fixArgsType(shape, *args))
  def reshape(self, shape, *args) -> Tensor:
    new_shape = fixArgsType(shape, *args)
    assert len(new_shape)>0 and all(d!=0 for d in new_shape), f'Zeros are not allowed in shape {new_shape}'# we can't remove dimensions
    return ops.Reshape.apply(self, shape= tuple(-prod(self.shape)//prod(new_shape) if d==-1 else d for d in new_shape))
  
  def transpose(self, ax1=1, ax2=0) -> Tensor:
    order = list(range(len(self.shape)))
    order[ax1], order[ax2] = order[ax2], order[ax1]
    return self.permute(order)
  '''

# we don't need to every instance of the Tensor to know all operations, just the operations that lead to it
class Function:
  def __init__(self, *tensors:Tensor):
    self.parents = tensors
    self.saved_tensors : List[Tensor] = []
    self.membuffer = []

    self.needs_input_grad = [p.requires_grad for p in self.parents]
    self.requires_grad = True if any(self.needs_input_grad) else (None if any(x is None for x in self.needs_input_grad) else False)

  def save_for_backward(self, *x):
    self.saved_tensors.extend(x)

  def forward(self, *args, **kwargs): raise NotImplementedError("forward() is not implementd")
  def backward(self, *args, **kwargs): raise NotImplementedError("backward() is not implementd")

  @classmethod
  def apply(fxn:Type[Function], *real_srcs:Tensor, **kwargs) -> Tensor:
    srcs = [Tensor(t) if not isinstance(t, Tensor) else t for t in real_srcs] # input type correction
    ctx = fxn(*srcs)
    ret = Tensor( ctx.forward(*[t.data for t in srcs], **kwargs), requires_grad=ctx.requires_grad)
    if ctx.requires_grad: ret._ctx = ctx    # used by autograd engine
    return ret

from ops import *
