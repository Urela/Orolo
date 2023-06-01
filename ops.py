import numpy as np
from tensor import *
# --------------- basic (math) operations --------------- 

class Add(Function):
    def forward(self, x:np.ndarray, y:np.ndarray) -> np.ndarray:
        self.save_for_backward(x, y)
        return np.add(x, y, dtype=np.float32)

    def backward(self, grad_output):
        return grad_output, grad_output

class Sub(Function):
    def forward(self, x:np.ndarray, y:np.ndarray) -> np.ndarray:
        self.save_for_backward(x, y)
        return np.subtract(x, y, dtype=np.float32)

    def backward(self, grad_output): 
        return grad_output, -grad_output


class Mul(Function):
    def forward(self, x:np.ndarray, y:np.ndarray) -> np.ndarray:
        self.save_for_backward(x, y)
        return np.multiply(x, y, dtype=np.float32)

    def backward(self, grad): 
        x,y = self.saved_tensors
        return y*grad, x*grad

class MatMul(Function):
    def forward(self, x:np.ndarray, y:np.ndarray) -> np.ndarray:
        self.save_for_backward(x, y)
        return np.matmul(x, y, dtype=np.float32)

    def backward(self, grad):
        x, y = self.saved_tensors
        grad_x = np.matmul(grad, y.T, dtype=np.float32)
        grad_y = np.matmul(x.T, grad, dtype=np.float32)
        return grad_x, grad_y

class Pow(Function):
    def forward(self, x:np.ndarray, y:np.ndarray) -> np.ndarray:
        out = np.power(x,y, np.float32)
        self.save_for_backward(x, y, out)
        return out

    # TODO
    #def backward(self, grad_output):
    #    x,y,out = self.saved_tensors
    #    grad_x = grad_output * y * (out/x)
    #    grad_y = grad_output * np.log(x) * out
    #    return grad_x, grad_y

class Truediv(Function):
    def forward(self, x:np.ndarray, y:np.ndarray) -> np.ndarray:
        return np.true_divide(x, y, np.float32)

# ------ operations ------ 
class Sum(Function):
    def forward(self, x:np.ndarray, y:np.ndarray) -> np.ndarray:
        self.save_for_backward(x, y)
        return np.sum(x, y, np.float32)

    def backward(self, grad):
        x, = self.saved_tensors
        grad = grad+np.zeros_like(x, np.float32)
        return grad

class Mean(Function):
    def forward(self, x:np.ndarray) -> np.ndarray:
        self.save_for_backward(x)
        return np.mean(x)

    def backward(self, grad):
        x, = self.saved_tensors
        grad = grad.mean()+np.zeros_like(x, dtype=np.float32)
        return grad
    

# ------ register the operations ------ 
''' set up each operation as attribute, this allows to register new operation at a later date'''

def register(name, fxn):
    func = lambda *args, **kwargs: fxn.apply(*args, **kwargs)
    setattr(Tensor, "_"+name if getattr(Tensor, name, None) is not None else name, func)

# register the operators as operation
def register_op(name, fxn):
    setattr(Tensor, f"__{name}__", fxn)
    setattr(Tensor, f"__i{name}__", lambda self,x: self.assign(fxn(self,x)))
    setattr(Tensor, f"__r{name}__", lambda self,x: fxn(x,self))

# register operations
for name,fxn in {'add':Add, 'sub':Sub, 'mul':Mul, 'pow':Pow, 'matmul' :MatMul, 'truediv':Truediv}.items():
    register(name, fxn)
    register_op(name, getattr(Tensor, name))

# register operations
#register('sum', Sum)
register('mean', Mean)
