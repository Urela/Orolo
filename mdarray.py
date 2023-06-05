from __future__ import annotations 

from enum import Enum
from typing  import Union, Tuple, Type
#from collections import 
import functools
import operator

'''
 Ordered fastest operation to slowest
1) ManipulationOps : operations that apply only structal changes to how data in memeory is traversed/viewed
  - However, it is not always possible to reshape an array and avoid copying data.
2) FusedOps: Union[ManipulationOps, Data Read and Writes]
3) ElementwiseOps: Union[BinaryOps, UnaryOps]
  - Have to read and write every data entry

TODO: Fuzy testing
TODO: minimize all the copy operations, 
  - you have to make a view tracker class to track 
  - is python self a pointer? how to get pointers in python, global varibles?

'''
#FusedOps
ManipulationOps = Enum('ManipulationOps', ['RESHAPE','SLICE','PERMUTE','EXPAND','FLIP'])
ElementwiseOps = Enum('ElementwiseOps', ['ADD','SUB','MUL','DIV','POW','CMPEQ', 'NEG','LOG','EXP','SIGN','RELU'])
ReduceOps = Enum('ReduceOps', ['SUM','MAX'])
ConvOps = Enum('ConvOps', ['CONV'])
LoadOps = Enum('LoadOps', ['FROMCPU'])

LAZY = False
DEBUG = False

import sys
sys.setrecursionlimit(10000)


import numpy
class MACHINE_CPU(numpy.ndarray):
  op_implementation = {
     #ElementwiseOps.ADD: lambda a,b:a+b, ElementwiseOps.SUB: lambda a,b:a-b, ElementwiseOps.MUL: lambda a,b:a*b,
     #ElementwiseOps.DIV: lambda a,b:a/b, ElementwiseOps.CMPEQ: lambda a,b:1.0*(a==b), ElementwiseOps.NEG: lambda a,_:-a,

     ElementwiseOps.ADD: operator.add, ElementwiseOps.SUB: operator.sub, ElementwiseOps.MUL: operator.mul,
     ElementwiseOps.DIV: operator.truediv, ElementwiseOps.CMPEQ: lambda a,b:1.0*(a==b), ElementwiseOps.NEG: lambda a,_:-a,

     #ElementwiseOps.ADD: numpy.add, ElementwiseOps.SUB: numpy.subtract, ElementwiseOps.MUL: numpy.multiply,
     #ElementwiseOps.DIV: operator.truediv, ElementwiseOps.CMPEQ: lambda a,b:1.0*(a==b), ElementwiseOps.NEG: lambda a,_:-a,

     ElementwiseOps.LOG: lambda a,_:a.log(), ElementwiseOps.EXP: lambda a,_:a.exp(), ElementwiseOps.POW: operator.pow,
     ElementwiseOps.RELU: lambda a,_:a.relu(), ElementwiseOps.SIGN:lambda a,_:a.sign()
  }
  def exp(a): return  numpy.exp(a)
  def log(a): return  numpy.log(a)
  def sign(a): return numpy.sign(a)
  #def relu(a): return a*(a>0) # TODO change as it creates negative zeros
  def relu(a): return numpy.maximum(a, 0)

  def cmpeq(a): return  numpy.exp(a)
  def permute(a, order): return a.transpose(order)

  # clean out padd slicing indicies
  def custompad(a, padding): return numpy.pad(a, padding).view(MACHINE_CPU) if any(x > 0 or y > 0 for x,y in padding) else a
  def amax(a, *args, **kwargs): return numpy.amax(a, *args, **kwargs) #TODO
  def expand(a, new_shape): return numpy.broadcast_to(a, new_shape).view(MACHINE_CPU)
  def flip(a, axis): return numpy.flip(a, axis)



  # ***************** Don't call numpy API within the following functions *****************
  @staticmethod
  def load(self):return self.view(MACHINE_CPU)
  def compute(self): return self
  def elementwise_op(a,op,b): return MACHINE_CPU.op_implementation[op](a,b) 

  def reduce_op(a,op,new_shape):
    assert len(a.shape) == len(new_shape)
    axis = tuple( [i for i,(a,b) in enumerate(zip(a.shape, new_shape)) if a!=b ] )
    if a.shape == new_shape: return a[:] # no meaningful reduction, just a copy dataO
    elif op == ReduceOps.SUM: return a.sum(axis, keepdims=True)
    elif op == ReduceOps.MAX: return a.amax(axis, keepdims=True)
    else: raise Exception(f"unspported operation {op}")

  def manipulation_op(a,op,arg):
    if op == ManipulationOps.RESHAPE: return a.reshape(arg)
    elif op == ManipulationOps.PERMUTE: return a.permute(arg)
    elif op == ManipulationOps.FLIP: return a.flip(arg)
    elif op == ManipulationOps.SLICE: 
      padding = [(max(0, -l), max(0, r-a.shape[i])) for i,(l,r) in enumerate(arg)]
      return a.custompad(padding) [tuple(slice(l+padding[i][0],  r+padding[i][0],None) for i,(l,r) in enumerate(arg))] 
    elif op == ManipulationOps.EXPAND: return a.expand(arg)
    else: raise Exception(f"unspported operation {op}")


  def conv_op(a, op, b, C):
    assert op == ConvOps.CONV, f"unsupported operation {op}"
    a = a.manipulation_op(ManipulationOps.SLICE, ((0, a.shape[0]), (0, a.shape[1]), (-C.py, a.shape[2]+C.py_), (-C.px, a.shape[3]+C.px_)))
    ga = a.reshape(C.N, C.groups, C.Cin, a.shape[2], a.shape[3]) # add group dimension to treat each tensor as a object, why??

    ta = numpy.lib.stride_tricks.as_strided(ga,
      shape=(C.N, C.groups, C.Cin, C.out_x, C.out_y, C.R,C.S),
      strides=(*ga.strides[0:3], ga.strides[3]*C.sx, ga.strides[4]*C.sy, ga.strides[3]*C.dx, ga.strides[4]*C.dy ),
      writeable=False,
    )

    tb = b.reshape(C.groups, C.rcout, C.Cin, C.R, C.S)
    tmp = numpy.empty((C.N, C.groups, C.out_y,C.out_x, C.rcout), dtype=a.dtype)
    return numpy.moveaxis(tmp,4,2).reshape(C.N,  C.groups*C.rcout, C.out_y, C.out_x).view(MACHINE_CPU) ##????

#***************** Conv args *****************
#TODO: I don't like this 
from collections import namedtuple

ConvArgs = namedtuple('ConvArgs', ['R','S', 'groups', 'rcout', 'Cin', 'out_x', 'out_y', 'H', 'W', 'N', 'sx', 'sy', 'Cout', 'py','py_', 'px','px_', 'dy','dx', 'out_shape'])
def getConvArgs(a_shape, b_shape, stride=1, groups=1, padding=0, dilation=1):
  N,Cin_,H,W = a_shape # batches, in channels height, width
  Cout,Cin,R,S = b_shape # out channels, in channels, height, width,

  # dealing with asymemetric stride, dialtion and padding
  sy,sx = (stride,stride) if isinstance(stride, int) else stride
  dy,dx = (dilation,dilation) if isinstance(dilation, int) else dilation
  if not isinstance(padding, int) and len(padding) ==4: px,px_,py,py_ = padding
  else : py,px = (padding,padding) if isinstance(padding, int) else padding; py_, px_ = py,px

  out_x = (W + px+px_ - dx * (S-1) -1) // sx + 1
  out_y = (H + py+py_ - dy * (R-1) -1) // sy + 1
  if Cin_*groups != Cin: raise Exception(f"Input Tensor shape {a_shape} does not match the shape of the weights {b_shape}. ({Cin_*groups} vs. {Cin})")
  assert Cout %groups == 0
  return ConvArgs(R,S, groups, Cout//groups, Cin, out_x, out_y, H, W, N, sx, sy, Cout, py,py_, px,px_, dy,dx, (N,Cout,out_y,out_x))


@functools.cache
def stride_from_shape(shape):
  strides = [1]
  for x in shape[1::][::-1]:
    strides += [x*strides[0]] + strides
  return tuple(strides)

@functools.cache
def gen_view(shape): 
  if len(shape) ==1: shape = (1,)
  assert all([ isinstance(x,int) for x in shape])
  return View(tuple(shape), stride_from_shape(shape))

class View:
  def __init__(self, shape,stride,offset=0):
   self.shape, self.stride, self.offset = shape, stride, offset

  def contiguous(self):
    return self.offset == 0 and all(s2==s3 or s1==1 for s1,s2,s3 in zip(self.shape, self.strides, strides_from_shape(self.shape)))

from dataclasses import dataclass 
@dataclass
class ArrayOp:
  op : OP
  src: List[Type[MDArray], Type[ArrayOp]]
  arg: Any=None
class Machine:
  vars()[MACHINE_CPU] = MACHINE_CPU
  DEFAULT = 'CPU'
  classification = {'CPU':MACHINE_CPU}



#***************** View manipulations *****************
# TODO move these into a functions so that we donn't have to make a copy of the view stack
from math import prod

# add or remove a dimensions <change strides>
def reshape_view(views, new_shape):
  shape, stride, offset = views[-1].shape, views[-1].stride, views[-1].offset
  assert all([isinstance(x,int) for x in new_shape ])
  assert prod(shape) == prod(new_shape) #
  if shape == new_shape: return views # no meangingul change

  # we are adding or removing a dimension, check if the wrapped shape is the same
  if tuple([ s for s in shape if s!=1]) == tuple([ s for s in new_shape if s!=1]):
    old_strides = [ s for s,d in zip(stride,shape) if d!=1 ] 
    new_strides = [ 0 if d==1 else old_strides.pop(0) for d in new_shape]
    views[-1] = View(new_shape, new_strides, offset)
    return views

  new_view = View(tuple(new_shape), stride_from_shape(tuple(new_shape)))
  if views[-1].contiguous : views[-1] = new_view
  else:views.append(new_view)
  return views

def slice_view(views, *arg): 
  shape, stride, offset = views[-1].shape, views[-1].stride, views[-1].offset
  assert len(args) == len(self.shape)
  #zeroview = ZeroView(self.shape, arg) ### what is this ????
  new_offset = offset + sum([stride[i]*x for i,(x,_) in enumerate(arg)])
  new_shape = tuple([e-s for s,e in args])
  views[-1] = View(new_shape, stride, new_offset)
  #if zeroview.expr != "valid=valid":
  #  views += [zer
  return views

def permute_view(views, axis): 
  shape, stride, offset = views[-1].shape, views[-1].stride, views[-1].offset
  assert all([isinstance(x, int) and x>=0 and x<=len(shape) for x in axis]) 
  assert len(axis) == len(shape) and len(set(axis)) == len(axis) # Does input corespond to shape
  views[-1] = View( tuple(shape[a] for a in axis), tuple(stride[a] for a in axis), offset) # don't generate new shape when you have it 
  return views

def expanad_view(views, new_shape): 
  shape, stride, offset = views[-1].shape, views[-1].stride, views[-1].offset
  assert all([isinstance(x,int) for x in new_shape]) , f"{new_shape}"
  assert all([a==b or a==1 for a,b in zip(shape, new_shape)])
  stride = [ s if a==b else 0 for s,(a,b) in zip(stride, zip(shape, new_shape))]
  views[-1] = View(new_shape, stride, offset)
  return views

# TODO: this is a mess
def flip_view(views, *axis): 
  shape, strides, offset = views[-1].shape, views[-1].stride, views[-1].offset
  #new_stride(*[-1 if i in axis else 1 for i in range(len((views.shape)))])
  mul = [-1 if i in axis else 1 for i in range(len((shape)))]
  assert all([isinstance(x,int) for x in mul]) 
  strides_ = [z*m for z,m in zip(strides, mul)]
  offset_  = sum([(d-1)*z  for d,z,m in zip(shape, strides, mul) if m < 0 ])
  new_shape = [d+(abs(m)-1)//abs(m) for d,m in zip(shape, mul)]

  views[-1] = View(new_shape, strides_, offset+offset_)
  return views
#manipulate_view = {ManipulationOps.RESHAPE: view_reshape ,ManipulationOps.SLICE:view_slice, ManipulationOps.PERMUTE:view_permute, ManipulationOps.EXPAND:view_expanad}
manipulate_view = {ManipulationOps.RESHAPE: reshape_view ,ManipulationOps.SLICE:slice_view, ManipulationOps.PERMUTE:permute_view, ManipulationOps.EXPAND:expanad_view, ManipulationOps.FLIP:flip_view}

# ***************** Generates tree nodes *****************
def eval_LoadOps(arr):
  return Machine.classification[arr.machine].load(arr.op.arg), [], LoadOps 

def eval_ManipulationOps(arr):
  mdarray  = find_MDArray(arr.op)[0].evaluate()
  arrayops = find_ArrayOp(arr.op)[::-1] # get operations in order they occured
  return functools.reduce(lambda a,o: a.manipulation_op(o.op, o.arg), arrayops, mdarray), [mdarray], ManipulationOps

def eval_ElementwiseOps(arr): 
  src = [ x.evaluate() for x in arr.op.src if x is not None]
  return src[0].elementwise_op(arr.op.op, src[-1]), src, ElementwiseOps 

def eval_ConvOps(arr): 
  src = [ x.evaluate() for x in arr.op.src ]
  return src[0].conv_op(arr.op.op, src[-1],  arr.op.arg), src, ConvOps 

def eval_ReduceOps(arr:MDArray): 
  src = [ x.evaluate() for x in arr.op.src ]
  return src[0].reduce_op(arr.op.op, arr.op.arg), src, ReduceOps


def find_ArrayOp(op) -> List[ArrayOp]:
  arr = [ find_ArrayOp(o) for o in op.src if isinstance(o, ArrayOp) ]
  return functools.reduce(lambda a,b:a+b,arr,[op])

def find_MDArray(op) -> List[MDArray]:
  arr = [ find_MDArray(o) if isinstance(o, ArrayOp) else [o] for o in op.src]
  return functools.reduce(lambda a,b:a+b,arr,[])

evaluator = {ManipulationOps:eval_ManipulationOps, ElementwiseOps:eval_ElementwiseOps, ReduceOps:eval_ReduceOps, ConvOps:eval_ConvOps, LoadOps:eval_LoadOps}

# ***************** Array *****************
class MDArray:
  def __init__(self, machine, dimension:Union[Type[list[View], Tuple[int,...]]], optype, op:ArrayOp):
    self.views = dimension[:] if isinstance(dimension[0], View) else [gen_view(dimension)]
    self.machine = machine
    self.buffer: Optional[Machine] = None
    self.optype, self.op = optype, op
    if not LAZY : self.evaluate()
    pass

  def __repr__(self): return  f"<MDArray shape:{self.shape} and {self.optype}>"

  @property
  def shape(self): return self.views[-1].shape
  @property
  def stride(self): return self.views[-1].stride
  @property
  def offset(self): return self.views[-1].offset

  def contiguous(self): return len(self.views) == 1 and self.views[-1].contiguous

  def evaluate(self) -> Machine:
    if self.buffer is None:
      self.buffer, srcs, optype = evaluator[self.optype](self)
      #if DEBUG: print(f"{self.op} : {', '.join([str(x.shape) for x in srcs])} -> {self.buffer.shape}")
      if DEBUG: print(f"{[x.op for x in find_ArrayOp(self.op)]} : {', '.join([str(x.shape) for x in srcs])} -> {self.buffer.shape}")
      del self.op # prune our AST  to save space
    return self.buffer

  @staticmethod
  def load(a:MDArray, machine=Machine.DEFAULT):
    return MDArray(machine, a.shape, LoadOps, ArrayOp(LoadOps.FROMCPU, tuple(), a))

  def compute(a:MDArray): return a.evaluate().compute()

# ***************** Opertaions *****************
  def elementwise_op(a:MDArray, op:ElementwiseOps, b:MDArray=None):
    return MDArray(a.machine, a.shape, ElementwiseOps, ArrayOp(op, (a,b)))

  def conv_op(a:MDArray, op:ConvOps, b:MDArray ,args:ConvArgs):
    return MDArray(a.machine, args.out_shape, ConvOps, ArrayOp(op, (a,b), args))

  def manipulation_op(a:MDArray, op:ManipulationOps, arg):
    # merge arrays if pervious operation was a manipulation_op, Optimization trick
    A = a.op if a.optype == ManipulationOps and a.buffer is None else a
    ret = MDArray(a.machine, manipulate_view[op](a.views[:], arg), ManipulationOps, ArrayOp(op, (A,) ,arg))

    # if return array is contiguous and same shape as input arr, then we haven't made any meangiful change to view
    if a.buffer is None and ret.contiguous:
      arr = find_MDArray(ret.op)[0]
      if arr.shape == ret.shape: return arr
    return ret

# ***********************************************
  def reduce_op(a:MDArray, op:ReduceOps, new_shape):
    return MDArray(a.machine, tuple(new_shape), ReduceOps, ArrayOp(op, (a,), tuple(new_shape)))

if __name__=='__main__':

  import numpy as np
  DEBUG = True
  A = MDArray.load(np.random.uniform(size=(1,3,3)))
  B = MDArray.load(np.random.uniform(size=(1,3,3)))

  C = A.elementwise_op(ElementwiseOps.ADD, B)
  C = C.elementwise_op(ElementwiseOps.MUL, A)
  C = C.elementwise_op(ElementwiseOps.RELU)

  C = C.manipulation_op(ManipulationOps.PERMUTE, (2,1,0))
  C = C.manipulation_op(ManipulationOps.RESHAPE, (1,1,9))
  z = C.reduce_op(ReduceOps.SUM, (1,1,1))
  z = C.reduce_op(ReduceOps.MAX, (1,1,1))
  input_shape = C.shape 
  z = z.manipulation_op(ManipulationOps.EXPAND, input_shape)

  print(z.compute())
  # ***********************************************

  A = MDArray.load(np.random.uniform(size=(1,3,3,3)))
  B = MDArray.load(np.random.uniform(size=(1,3,3,3)))

  print('A', A.compute())
  print('B', B.compute())

  z = A.conv_op(ConvOps.CONV, B, (getConvArgs(A.shape, B.shape)))
  print()
  print(z.compute())


  pass
