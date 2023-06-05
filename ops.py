from __future__ import annotations
from mdarray import MDArray, ManipulationOps, ElementwiseOps, ReduceOps, ConvOps
from tensor import Tensor, Function
import functools

"""
This is where the autograd stuff occures

ManipulationOps = Enum('ManipulationOps', ['RESHAPE','SLICE','PERMUTE','EXPAND']) 
ElementwiseOps = Enum('ElementwiseOps', ['ADD','SUB','MUL','DIV','CMPEQ', 'NEG','LOG','EXP','SIGN','RELU'])
ReduceOps = Enum('ReduceOps', ['SUM','MAX']) 

TODO - run backwards as well if needs grad, then just compute grads when needed
  - what is backward and forward??? but how we we ghet dout??
  - def __call__(self, A, B): pass
  - def gradients(self, dout??) : return grads
"""

# https://stackoverflow.com/questions/3382352/equivalent-of-numpy-argsort-in-basic-python
def argsort(x): return sorted(range(len(x)), key=x.__getitem__)
# ******************** Elementwise Ops ******************** 
class Add(Function):
  def forward(ctx, A, B): 
    return A.elementwise_op(ElementwiseOps.ADD, B)

  def backward(ctx, dout): 
    return dout, dout

class Sub(Function):
  def forward(self, A, B): 
    return A.elementwise_op(ElementwiseOps.SUB, B)

  def backward(self, dout): 
    return dout, dout.elementwise_op(ElementwiseOps.NEG)
 
class Mul(Function):
  def forward(self, A, B): 
    self.buffer.extend([A, B])
    return A.elementwise_op(ElementwiseOps.MUL, B)

  def backward(self, dout ): 
    A, B = self.buffer
    grad_A = B.elementwise_op(ElementwiseOps.MUL, dout) 
    grad_B = A.elementwise_op(ElementwiseOps.MUL, dout) 
    return grad_A, grad_B

class Div(Function):
  def forward(self, A, B): 
    self.buffer.extend([A, B])
    return A.elementwise_op(ElementwiseOps.DIV, B)

class Pow(Function): 
  def forward(ctx, A, B):
    #self.buffer.extend([A])
    return A.elementwise_op(ElementwiseOps.POW, B)

class Neg(Function): 
  def forward(ctx, A):
    #self.buffer.extend([A])
    return A.elementwise_op(ElementwiseOps.NEG)

class Exp(Function): 
  def forward(ctx, A):
    ctx.buffer.extend([A])
    return A.elementwise_op(ElementwiseOps.EXP)
  def backward(ctx, dout):
    A,  = ctx.buffer
    return A.elementwise_op(ElementwiseOps.MUL, dout)

class Log(Function): 
  def forward(ctx, A):
    ctx.buffer.extend([A])
    return A.elementwise_op(ElementwiseOps.LOG)

  def backward(ctx, dout):
    A,  = ctx.buffer
    return dout.elementwise_op(ElementwiseOps.DIV, A)

class Relu(Function): 
  def forward(ctx, A):
    ctx.buffer.extend([A])
    return A.elementwise_op(ElementwiseOps.RELU)

  def backward(self, dout):
    A,  = self.buffer
    ret = A.elementwise_op(ElementwiseOps.SIGN)
    ret = ret.elementwise_op(ElementwiseOps.RELU)
    return ret.elementwise_op(ElementwiseOps.MUL, dout)

class Sign(Function): 
  def forward(ctx, A):
    #self.buffer.extend([A])
    return A.elementwise_op(ElementwiseOps.SIGN)

# ******************** Reduction Ops ******************** 

class Max(Function): 
  def forward(ctx, A, axis=None):
    ret = A.reduce_op(ReduceOps.MAX, reduce_shape(A.shape, axis))
    ctx.buffer.extend([A, ret])
    return ret

  #TODO
  def backward(ctx, dout):
    A, ret  = ctx.buffer
    max_is_1s = A.elementwise_op(ElementwiseOps.CMPEQ, ret.manipulation_op(ManipulationOps.EXPAND, A.shape))

    div = max_is_1s.reduce_op(ReduceOps.SUM, dout.shape)
    div = div.manipulation_op(ManipulationOps.EXPAND, A.shape)

    max_is_amount = max_is_1s.elementwise_op(ElementwiseOps.DIV, div)
    grad_output_expanded = dout.manipulation_op(ManipulationOps.EXPAND, A.shape)
    return max_is_amount.elementwise_op(ElementwiseOps.MUL, grad_output_expanded)


class Sum(Function): 
  def forward(ctx, A, axis=None):
    ctx.buffer.extend( A.shape )
    return A.reduce_op(ReduceOps.SUM, reduce_shape(A.shape, axis))
  def backward(ctx, dout):
    input_shape = ctx.buffer
    return dout.manipulation_op(ManipulationOps.EXPAND, input_shape)

# ******************** ManipulationOps Ops ******************** 
from math import prod
class Reshape(Function):
  def forward(ctx, A, new_shape):
    new_shape = tuple(-prod(A.shape) // prod(new_shape) if s == -1 else s for s in new_shape) # TODO
    ctx.buffer.extend(A.shape)
    return A.manipulation_op(ManipulationOps.RESHAPE, new_shape)

  def backward(ctx, dout):
    input_shape = ctx.buffer
    return dout.manipulation_op(ManipulationOps.RESHAPE, input_shape)

  
class Permute(Function):
  def forward(ctx, A, order=(1,0)):
    ctx.buffer.extend(order)
    return A.manipulation_op(ManipulationOps.PERMUTE, order)

  def backward(ctx, dout):
    norder = argsort(ctx.buffer)
    return dout.manipulation_op(ManipulationOps.PERMUTE, norder)

class Slice(Function):
  def forward(ctx, A, arg):
    return A.manipulation_op(ManipulationOps.SLICE, arg)

class Expand(Function): 
  def forward(ctx, A, shape):
    ctx.buffer.extend(shape)
    return A.manipulation_op(ManipulationOps.EXPAND, shape)

  def backward(ctx, dout):
    input_shape = ctx.buffer
    return dout.reduce_op(ReduceOps.SUM, input_shape)


def reduce_shape(shape, axis): return [1 if i in axis else shape[i] for i in range(len(shape))]
# ******************** Register operations ******************** 
from mdarray import getConvArgs
class Conv2D(Function):
  def _conv(ctx, A, B, args): return A.conv_op(ConvOps.CONV, B, args) # I don't want to save A,B all the time
  def forward(ctx, A,B, stride=1, groups=1, dilation=1,padding=0):
    args = getConvArgs(A.shape, B.shape, stride, groups, padding=padding, dilation=dilation)
    ctx.buffer.extend([A,B,args])
    return ctx._conv(A, B, args)

  def backward(ctx, dout):
    A, B, args = ctx.buffer
    dx, dw = None, None
    if ctx.needs_input_grad[0]: # compute derivative of inputs using ProcessingOps.CONV (this is a transposed conv)
      At = dout
      if args.sx > 1 or args.sy >1: # account for different strides
        At = At.manipulation_op(ManipulationOps.RESHAPE, (*dout.shape[0:3], 1, dout.shape[3],1)) ### 
        At = At.manipulation_op(ManipulationOps.SLICE, (0,xt.shape[0]), (0,xt.shape[1]), (0,xt.shape[2]), (0,args.sy), (0,xt.shape[4]), (0,sx))
        At = At.manipulation_op(ManipulationOps.RESHAPE, (*xt.shape[0:2], xt.shape[2]*args.sy, xt.shape[4]*args.sx)) ### 
      Bt = B.manipulation_op(ManipulationOps.RESHAPE, (args.groups, args.rcout, args.Cin, args.R, args.S))
      Bt = Bt.manipulation_op(ManipulationOps.FLIP, (3,4))
      Bt = Bt.manipulation_op(ManipulationOps.PERMUTE, (0,1,2,3,4))
      Bt = Bt.manipulation_op(ManipulationOps.RESHAPE, (args.groups*args.Cin, args.rcout, args.R, args.S))
      py, px = (args.R-1)*args.dy - args.py, (args.S-1)*args.dx - args.px 
      py_ = A.shape[2] - At.shape[2] + args.py
      px_ = A.shape[3] - At.shape[3] + args.px
      cdx= getConvArgs(At.shape, Bt.shape, dilation=(args.dy, args.dx), padding=(px,px_,py,py_), groups=args.groups )
      dx = ctx._conv(At, Bt, cdx) 

    if ctx.needs_input_grad[1]: # compute derivative of weights using ProcessingOps.CONV (this is a transposed conv)
      AdB = A.manipulation_op(ManipulationOps.RESHAPE, (args.N, args.groups, args.Cin, args.H, args.W))
      AdB = AdB.manipulation_op(ManipulationOps.PERMUTE, (2,1,0,3,4))
      AdB = AdB.manipulation_op(ManipulationOps.RESHAPE, (args.Cin, args.groups*args.N, args.H, args.W))
      dout_dw = dout.manipulation_op(ManipulationOps.PERMUTE, (1,0,2,3))
      dout_dw = dout_dw.manipulation_op(ManipulationOps.RESHAPE, (args.Cout, args.N, args.out_y, args.out_x))
      py_ = (B.shape[2]-1) * args.dy - AdB.shape[2] - args.py + args.sy * (dout_dw.shape[2]-1) +1
      px_ = (B.shape[3]-1) * args.dx - AdB.shape[3] - args.px + args.sx * (dout_dw.shape[3]-1) +1
      cdw = getConvArgs(AdB.shape, dout_dw.shape, padding=(args.px,px_,args.py,py_), stride=(args.dy,args.dx), dilation=(args.sy, args.sx), groups=args.groups)
      dout_weight = ctx._conv(AdB, dout_dw, cdw) 
      dw = dout_weight.manipulation_op(ManipulationOps.PERMUTE, (1,0,2,3)) 
    return dx, dw


# ******************** Register operations ******************** 

def register(name, fxn):
  func = lambda *args, **kwargs: fxn.apply(*args, **kwargs)
  setattr(Tensor, "_"+name if getattr(Tensor, name, None) is not None else name, func)

#### register elementwise ops
for name, fxn in {"add":Add,"sub":Sub,"mul":Mul,"pow":Pow,"div":Div}.items():
  register(name, fxn)
def register_op(name, fxn):
  setattr(Tensor, f"__{name}__", fxn)
  setattr(Tensor, f"__i{name}__", lambda self, other: self.assign(fxn(self,other)))
  setattr(Tensor, f"__r{name}__", lambda self, other: fxn(other,self))
for name in ['add', 'sub', 'mul', 'pow', 'matmul']: register_op(name, getattr(Tensor, name))


#### why we still distinguishing elementwise ops ??? no pre defined symbols
for name, fxn in {'relu':Relu, 'log':Log, 'exp':Exp, 'neg':Neg}.items():
  register(name, fxn)

## register manipulations ops and reduction
for name, fxn in {'reshape':Reshape, 'slice':Slice, 'permute':Permute, 'expand':Expand, 'sum':Sum, 'max':Max}.items():
  register(name, fxn)

## register conv_ops 
register("conv2d", Conv2D)
