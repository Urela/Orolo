import numpy as np
from tensor import *

class ReLU(Function):
    ## commented out pytorch inspired implemention for faster implentation
    def forward(self, x:np.ndarray) -> np.ndarray:
        self.save_for_backward(x)
        #return np.where(x > 0, x, 0)
        return x * (x > 0)


    def backward(self, dout):
        x, = self.saved_tensors
        #return np.where(x > 0, dout, 0)
        return dout * (x > 0)


class LogSoftmax(Function):
    def forward(self, x:np.ndarray) -> np.ndarray:
        c = np.max(x, axis=1)
        exp_x = np.exp(x - c.reshape((-1,1))) 
        logsumexp = c + np.log(exp_x.sum(axis=1))
        output = x - logsumexp.reshape((-1, 1))
        self.save_for_backward(output)
        return output 

    def backward(self, dout):
        # assume we are using NLL loss
        output, = self.saved_tensors
        return dout - np.exp(output)*(dout.sum(axis=1).reshape((-1, 1)))


# ------ register the operations ------ 
''' set up each operation as attribute, this allows to register new operation at a later date'''

def register(name, fxn):
    func = lambda *args, **kwargs: fxn.apply(*args, **kwargs)
    setattr(Tensor, "_"+name if getattr(Tensor, name, None) is not None else name, func)



register("relu", ReLU)
register("logsoftmax", LogSoftmax)

if __name__ == '__main__': 
    fxn = LogSoftmax()
    A = np.array([10,11,2,2,3,5])
    print(A,'\n', fxn.forward(A))
    print(sum(fxn.forward(A)))
