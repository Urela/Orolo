from mdarray import MDArray, function, OPERATOR 
from typing import Union, Tuple
from tensor import Function, Tensor


# https://stackoverflow.com/questions/3382352/equivalent-of-numpy-argsort-in-basic-python
def argsort(x): return sorted(range(len(x)), key=x.__getitem__)
# ******************** Pointwise Ops ******************** 
class Add(Function):
  def forward(ctx, a:MDArray ,b:MDArray) -> MDArray:
    return function(OPERATOR.ADD, (a,b))

  def backward(ctx, dout:MDArray) -> MDArray:
    return dout, dout

class Sub(Function):
  def forward(ctx, a:MDArray ,b:MDArray) -> MDArray:
    return function(OPERATOR.SUB, (a,b))

  def backward(ctx, dout:MDArray) -> MDArray:
    return dout, function(OPERATOR.NEG, (dout,))

class Mul(Function):
  def forward(ctx, a:MDArray ,b:MDArray) -> MDArray:
    ctx.membuffer.extend([a,b])
    return function(OPERATOR.MUL, (a,b))

  def backward(ctx, dout:MDArray) -> MDArray:
    a,b = ctx.membuffer
    return function(OPERATOR.MUL, (b,dout)), function(OPERATOR.MUL, (a,dout))

class Div(Function):
  def forward(ctx, a:MDArray ,b:MDArray) -> MDArray:
    ctx.membuffer.extend([a,b])
    return function(OPERATOR.DIV, (a,b))

  def backward(ctx, dout:MDArray) -> MDArray:
    a,b = ctx.membuffer
    Dout = function(OPERATOR.NEG, (dout,))
    ret  = function(OPERATOR.MUL, (Dout,a))
    B = function(OPERATOR.MUL, (b,b))
    return function(OPERATOR.DIV, (dout,b)), function(OPERATOR.DIV, (A,B))

class Pow(Function):
  def forward(ctx, a:MDArray ,b:MDArray) -> MDArray:
    ret = function(OPERATOR.POW, (a,b))
    ctx.membuffer.extend([a,b,ret])
    return ret

  def backward(ctx, dout:MDArray) -> MDArray:
    raise NotImplementedError(f'method is not implmented for {ctx}')

# ******************** Reduction Ops ******************** 

class Max(Function):
  def forward(ctx, a:MDArray, new_shape):
    ret = function(OPERATOR.MAX, (a,), new_shape)
    ctx.membuffer.extend([a, ret])
    return ret

  def backward(ctx, dout:MDArray) -> MDArray:
    a, ret = ctx.membuffer
    # 1s in locations where the max was chosen (can be two locations)
    max_is_1s = function(OPERATOR.CMPEQ, (a, function(OPERATOR.EXPAND, (ret,), a.shape)))

    # sum of locations, averaged
    div = function(OPERATOR.EXPAND, tuple([function(OPERATOR.SUM, tuple([max_is_1s]),dout.shape)]), a.shape)
    max_is_amount = function(OPERATOR.DIV, (max_is_1s, div))

    grad_output_expanded = function(OPERATOR.EXPAND, (dout,), a.shape)
    return function(OPERATOR.MUL, (max_is_amount, grad_output_expanded))


class Sum(Function):
  def forward(ctx, a:MDArray, new_shape):
    ctx.membuffer.extend(a.shape)
    return function(OPERATOR.SUM, (a,), new_shape)

  def backward(ctx, dout:MDArray) -> MDArray: 
    input_shape = ctx.membuffer
    return function(OPERATOR.EXPAND, (dout,), input_shape )

# ******************** Transformations Ops ******************** 
class Expand(Function): 
  def forward(ctx, a:MDArray, shape):
    ctx.membuffer.extend([a.shape])
    return function(OPERATOR.EXPAND, (a,), shape)

  def backward(ctx, dout:MDArray) -> MDArray: 
    input_shape, = ctx.membuffer
    return function(OPERATOR.SUM, (dout,), input_shape)

class Reshape(Function): 
  def forward(ctx, a:MDArray, shape):
    ctx.membuffer.extend([a.shape])
    return function(OPERATOR.RESHAPE, (a,), shape)

  def backward(ctx, dout:MDArray) -> MDArray: 
    input_shape, = ctx.membuffer
    return function(OPERATOR.RESHAPE, (dout,), input_shape)

class Permute(Function): 
  def forward(ctx, a:MDArray, order=(1,0)):
    ctx.membuffer.extend([order])
    return function(OPERATOR.PERMUTE, (a,), order)

  def backward(ctx, dout:MDArray) -> MDArray: 
    norder = argsort(*ctx.membuffer)
    return function(OPERATOR.PERMUTE, (dout,), norder)

class Maximum(Function):
  def forward(ctx, a:MDArray, b:MDArray):
    ret = function(OPERATOR.RELU, (a,b))
    ctx.membuffer.extend([ret, b])
    return ret

  def backward(ctx, dout:MDArray) -> MDArray: 
    ret, b = ctx.membuffer
    mask = function(OPERATOR.CMPEQ, (b,ret))
    #return function(OPERATOR.MUL, (dout,mask)), function(OPERATOR.MUL, (dout, function(OPERATOR.NOT, (mask,))) )
    return function(OPERATOR.MUL, (dout,mask)), function(OPERATOR.MUL, (dout,mask))

class Log(Function):
  def forward(ctx, a:MDArray):
    ctx.membuffer.extend([a])
    return function(OPERATOR.LOG, (a,))

  def backward(ctx, dout:MDArray) -> MDArray: 
    a, = ctx.membuffer
    return function(OPERATOR.DIV, (dout,a))

class Exp(Function):
  def forward(ctx, a:MDArray):
    ret = function(OPERATOR.EXP, (a,))
    ctx.membuffer.extend([ret])
    return ret.compute()

  def backward(ctx, dout:MDArray) -> MDArray: 
    ret, = ctx.membuffer
    return function(OPERATOR.DIV, (ret,dout))


import numpy 
class LogSoftmax(Function):
  def forward(self, x):
    x = x.compute()
    c = numpy.max(x, axis=1)
    exp_x = numpy.exp(x - c.reshape((-1,1))) 
    logsumexp = c + numpy.log(exp_x.sum(axis=1))
    output = x - logsumexp.reshape((-1, 1))
    self.membuffer.extend([output])
    return MDArray.load( output )

  def backward(self, dout):
    # assume we are using NLL loss
    dout = dout.compute()
    output, = self.membuffer
    return MDArray.load( dout - numpy.exp(output)*(dout.sum(axis=1).reshape((-1, 1))) )

class Matmul(Function):
  def forward(self, x, y):
    x, y = x.compute(), y.compute()
    self.membuffer.extend([x,y])
    return MDArray.load(numpy.matmul(x, y, dtype=numpy.float32))

  def backward(self, dout):
    dout = dout.compute()
    x, y = self.membuffer
    grad_x = numpy.matmul(dout, y.T, dtype=numpy.float32)
    grad_y = numpy.matmul(x.T, dout, dtype=numpy.float32)
    return MDArray.load( grad_x ), MDArray.load( grad_y )

class Mean(Function):
  def forward(self, x):
    x = x.compute()
    self.save_for_backward(x)
    return MDArray.load(numpy.mean(x))

  def backward(self, dout):
    x, = self.saved_tensors
    dout = dout.compute()
    return MDArray.load( dout.mean()+numpy.zeros_like(x, dtype=numpy.float32))

class ReLU(Function):
  ## commented out pytorch inspired implemention for faster implentation
  def forward(ctx, a:MDArray):
    b = MDArray.zeros(*a.shape)
    ret = function(OPERATOR.RELU, (a,b))
    ctx.membuffer.extend([ret, b])
    return ret

  def backward(ctx, dout:MDArray) -> MDArray: 
    ret, b = ctx.membuffer
    mask = function(OPERATOR.CMPEQ, (b,ret))
    #return function(OPERATOR.MUL, (dout,mask)), function(OPERATOR.MUL, (dout, function(OPERATOR.NOT, (mask,))) )
    return function(OPERATOR.MUL, (dout, function(OPERATOR.NOT, (mask,))) )
    #return function(OPERATOR.MUL, (dout,mask))

def register(name, fxn):
  func = lambda *args, **kwargs: fxn.apply(*args, **kwargs)
  setattr(Tensor, "_"+name if getattr(Tensor, name, None) is not None else name, func)

# register the operators as operation
def register_op(name, fxn):
  setattr(Tensor, f"__{name}__", fxn)
  setattr(Tensor, f"__i{name}__", lambda self,x: self.assign(fxn(self,x)))
  setattr(Tensor, f"__r{name}__", lambda self,x: fxn(x,self))

# register operations
for name,fxn in {'add':Add, 'sub':Sub, 'mul':Mul, 'pow':Pow, 'truediv':Div}.items():
  register(name, fxn)
  register_op(name, getattr(Tensor, name))

# register operations

register('matmul', Matmul)
register('mean', Mean)
register("relu", ReLU)
register("logsoftmax", LogSoftmax)

if __name__ == '__main__':
  import unittest
  import numpy as np
  import operator

  class OperationTests(unittest.TestCase):
    def testforward(self):
      a,b,c = [np.random.uniform(size=(1,3,3)).astype('f') for x in range(3)]
      A,B,C = [MDArray.load(x) for x in (a,b,c)]

      assert (Add().forward(A,B).compute() == np.add(a,b)).all(), f'Error with Function add'
      assert (Sub().forward(A,B).compute() == np.subtract(a,b)).all(), f'Error with Function sub'
      assert (Mul().forward(A,B).compute() == np.multiply(a,b)).all(), f'Error with Function mul'
      assert (Div().forward(A,B).compute() == np.divide(a,b)).all(), f'Error with Function div'
    def testbackward(self):
      #a,b,c = [np.random.uniform(size=(1,3,3)).astype('f') for x in range(3)]
      #A,B,C = [MDArray.load(x) for x in (a,b,c)]
      # TOOO

      pass
  unittest.main()
