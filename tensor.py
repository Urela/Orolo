import functools
import numpy as np

"""
 TODO: 
 - [ ] move away from numpy backend, consider LAPACK 
 - [ ] Add support for GPUS like Triton
"""

class Tensor:
    def __init__(self, data, device=None, requires_grad=True):
        if isinstance(data, (list, int, float)):
            data = np.array(data, dtype=np.float32)
        if isinstance(data, (np.uint8, np.float32,  np.float16)):
            data = np.array(data, dtype=np.float32)
        if isinstance(data, np.ndarray): 
            self.data = np.array(data, dtype=np.float32)
        else: raise Exception(f"Can't create Tensor from {data} of type {type(data)}")

        self.grad = None
        self._ctx : Optional[Function] = None # used for the autograd graph construction
        self.requires_grad : Optional[bool] = requires_grad

    def backward(self): 
        def topo_sort(node: 'Tensor', visited: set, nodes: []):
            if node not in visited:
                visited.add(node)
                if node._ctx is not None:
                    [topo_sort(p, visited, nodes) for p in node._ctx.parents]
                    nodes.append(node)
            return nodes
        nodes = topo_sort(self, set(), []) # generate the topological sort of the compute graph       

        # initial grads - ones as we are impliciting creating gradients
        self.grad = Tensor.ones(*self.shape, requires_grad=False)

        # loop through sorted graph and update gradients
        for node in reversed(nodes):
            if not any(x.requires_grad for x in node._ctx.parents): continue
            assert(node.grad is not None)

            # get grads created by the operation applied to the tensor
            grads = node._ctx.backward(node.grad.data) 
            if len(node._ctx.parents) == 1:
                grads = [grads]

            # update gradients
            for parent, grad in zip(node._ctx.parents, grads):
                if grad is None: continue
                parent.grad = Tensor(grad) if parent.grad is None else (parent.grad + Tensor(grad))
        pass
        
    def __repr__(self): return f"<Tensor\n {self.data} \nwith grad\n {self.grad}>"
    # --------- Properties --------- 
    @property
    def shape(self): return self.data.shape

    @property
    def dtype(self): return self.data.shape

    def numpy(self): return self.data 

    def assign(self, x): self.data = x.data

    def zero_grad(self):
        self.grad = None
        self._ctx = None

    # ---------  Data initialization types --------- 
    @classmethod
    def ones(cls, *shape, **kwargs): return cls(np.ones(shape, dtype=np.float32), **kwargs)

    @classmethod
    def zeros(cls, *shape, **kwargs): return cls(np.zeros(shape, dtype=np.float32), **kwargs)

    @classmethod
    def rand(cls, *shape, **kwargs): return cls(np.random.randn(*shape).astype(np.float32), **kwargs)

    @classmethod 
    def uniform(cls, *shape, **kwargs):
        return cls((np.random.uniform(-1.,1., size=shape)/np.sqrt(np.prod(shape))).astype(np.float32), **kwargs)

# we don't need to every instance of the Tensor to know all operations, just the operations that lead to it
class Function:
    def __init__(self, *tensors:Tensor):
        self.parents = tensors
        self.saved_tensors : List[Tensor] = []
        self.needs_input_grad = [p.requires_grad for p in self.parents]
        self.requires_grad = True if any(self.needs_input_grad) else (None if any(x is None for x in self.needs_input_grad) else False)

    def save_for_backward(self, *x):
        self.saved_tensors.extend(x)

    def forward(self, *args, **kwargs):   
        raise NotImplementedError("forward() is not implementd")

    def backward(self, *args, **kwargs): 
        raise NotImplementedError("backward() is not implementd")

    @classmethod
    def apply(cls, *args, **kwargs):
        x = [t if isinstance(t, Tensor) else Tensor(t) for t in args] # convert arg to Tensor
        ctx = cls(*x) # some Function/operation i.e Add, Sub, MatMul,...
        ret = Tensor(ctx.forward(*[t.data for t in x], **kwargs), requires_grad=ctx.requires_grad)
        # store function for autograd engine
        if ctx.requires_grad:
            ret._ctx = ctx  
        return ret

from ops import *
from mlops import *
