from __future__ import annotations
from collections import namedtuple
from enum import Enum 
from typing import Tuple, Union, NamedTuple, ClassVar
import operator
import functools
import numpy as np
from math import prod

LoadFxns = Enum('LoadFxns', ['FROMCPU'])
PointwiseFxns = Enum('PointwiseFxns', ['ADD','SUB','MUL','DIV','POW','CMPEQ','NEG','LOG','EXP','SIGN','RELU','NOT']) # keep it as relu
TransformFxns = Enum('TransformFxns', ['RESHAPE','PAD','PERMUTE','EXPAND','STRIDE','SHRINK']) 
ReductionFxns = Enum('ReductionFxns', ['SUM','MAX'])  
#FusedFxns  = Enum('FusedFxns', ['MULACC']) # this can give us matmul

# This is fine for now
ops = {**{fxn.name : fxn for fxnType in [PointwiseFxns, TransformFxns, ReductionFxns, LoadFxns] for fxn in fxnType}, }
OPERATOR = namedtuple('OPERATOR', ops.keys())(**ops)

fxn_implementation = {
  PointwiseFxns.ADD: operator.add, PointwiseFxns.SUB: operator.sub, PointwiseFxns.MUL: operator.mul, PointwiseFxns.DIV: operator.truediv, PointwiseFxns.NEG: lambda a,_:-a,
}

from typing import NamedTuple
class DType(NamedTuple): 
  itemsize: int #  The element size of this data-type object.
  name: str     #  A bit-width name for this data-type.
  np: type
  def __repr__(self): return f"dtypes.{self.name}"

class dtypes:
  float16 : Final[DType] = DType(2, "half",  np.float16) # 2 bytes
  float32 : Final[DType] = DType(4, "float", np.float32) # 4 bytes
  @staticmethod
  def from_numpy(x) -> DType: return {np.dtype(np.float16) : dtypes.float16, np.dtype(np.float32) : dtypes.float32}[np.dtype(x.dtype)] 

class GenericShape:
  def __init__(self, shape:Tuple[int,...], dtype:DType=dtypes.float32, flops:int=0): self.shape, self.dtype, self.flops = shape, dtype, flops
  def consume_flops(self): self.flops, ret = 0, self.flops ; return ret # what ????

def shape_to_axis(old_shape:Tuple[int,...], new_shape:Tuple[int, ...]) ->Tuple[int,...]:
  assert len(old_shape) == len(new_shape), "Reduce shapes must have the same number of dimensions"
  return tuple(i for i,(a,b) in enumerate(zip(old_shape, new_shape)) if a!=b)

#Very cool dictionary hack
shape_fxn_for_op : Dict[op : callable] = {
  **{fxn : lambda self,other: GenericShape(self.shape, self.dtype, self.consume_flops()+other.consume_flops()+prod(self.shape)) for fxn in PointwiseFxns},
  **{fxn : functools.partial(lambda trans,self,arg: GenericShape(ViewStack(self.shape).transform_fxn(trans, arg).shape, self.dtype, self.consume_flops()), fxn) for fxn in TransformFxns}
}

#class CompiledArray: pass
class InterpretedArray:
  fxn_implementation: Dict[FxnTypes:callable] = shape_fxn_for_op 
  to_orolo_dtype = staticmethod(dtypes.from_numpy)
  array: Any
  def __init__(self, agent_array: Any): 
    self.array: Any = agent_array
    self.shape: Tuple[int,...] = tuple(agent_array.shape)
    self.dtype: Dtype = tuple(agent_array.shape)
    self.dtype = self._to_dtype(agent_array) if hasattr(self, '_to_dtype') else agent_array.dtype  #this is an inheritance mess TODO
    return 
  def transform_fxn(self, fxn:TransformFxns, arg=None): 
    return type(self)(self.fxn_implementation[fxn](self.array,arg)) if fxn in self.fxn_implementation else type(self)(getattr(self.array, fxn.name.lower())(arg))

  @classmethod
  def exec_graph(cls, graph:MDArrayFxn, output_array:Optional[InterpretedArray]=None, context=None) : # output -> Union[np.ndarry, torch.arry]
    if context is None: context = dict() #hmmm
    if graph in context: return context[graph] # cache out exection graph
    srcs = [cls.exec_graph(x, context=context) if isinstance(x, MDArrayFxn) else x for x in graph.src] # find the graph using recursuion
    if isinstance(graph.fxn, PointwiseFxns) and srcs is not None: assert srcs[0].shape == srcs[0].shape,   F"PointwiseFxns mismatch {srcs[0].shape} != {srcs[1].shape}"
    if isinstance(graph.fxn, ReductionFxns): assert all(r==n or n==1 for r,n in zip(srcs[0].shape, graph.arg)), F"ReductionFxns can't reduce {srcs[0].shape} -> {srcs[1].shape}"
    if isinstance(graph.fxn, TransformFxns): ret = srcs[0].transform_fxn(graph.fxn, graph.arg)
    else: ret = cls(cls.fxn_implementation[graph.fxn] (*([x.array for x in srcs] + ([graph.arg] if graph.arg else [])))) # get input arrays and their arguments

    context[graph] = ret
    if output_array is not None: 
      output_array.array = ret.array
    else: return ret

  @staticmethod
  def load(array): raise NotImplementedError(f'def load() is not implemented Array of type {type(self)}')      # implemented by the next Agent Array
  def compute(self): raise NotImplementedError(f'def compute() is not implemented Array of type {type(self)}') # implemented by the next Agent Array

class CPUArray(InterpretedArray):
  array: Any
  to_orolo_dtype = staticmethod(dtypes.from_numpy)
  fxn_implementation: ClassVar = {
    PointwiseFxns.ADD: operator.add, PointwiseFxns.SUB: operator.sub, PointwiseFxns.MUL: operator.mul, PointwiseFxns.DIV: operator.truediv,
    PointwiseFxns.NEG: lambda a:-a,PointwiseFxns.CMPEQ: lambda a,b:(a==b).astype(np.float32), PointwiseFxns.LOG: np.log, PointwiseFxns.EXP: np.exp,

    PointwiseFxns.NOT: lambda a:(1.0-a),
    PointwiseFxns.POW: operator.pow, PointwiseFxns.RELU: lambda a,b:np.maximum(a,b), PointwiseFxns.SIGN:np.sign,
    TransformFxns.PERMUTE: lambda x,order: x.transpose(order), TransformFxns.EXPAND: np.broadcast_to,

    ReductionFxns.SUM: lambda x, new_shape: x.sum(shape_to_axis(x.shape,new_shape), keepdims=True) if tuple(x.shape) != tuple(new_shape) else x[:],
    ReductionFxns.MAX: lambda x, new_shape: (x.amax if hasattr(x, "amax") else x.max)(shape_to_axis(x.shape, new_shape), keepdims=True) if tuple(x.shape) != tuple(new_shape) else x[:],
  }

  @staticmethod
  def load(array): return CPUArray(array)
  def compute(self): return self.array

@functools.lru_cache(maxsize=None)
def strides_from_shape(shape) -> Tuple[int,...]:
  strides=[1]
  for d in shape[1::][::-1]: strides = [d*strides[0]] + strides
  return strides

@functools.lru_cache(maxsize=None)
def _contiguous(shape:Tuple[int,...], strides:Tuple[int,...], offset:int=0) -> bool:
  return offset==1 and all( s==c or d==1 for d,s,c in zip(shape, strides, strides_from_shape(shape)))

View = namedtuple('view', ['shape','strides','offset'])
class ViewStack:
  def __init__(self, stack:Union[ViewStack, int[int,...]], views:Optional[List[View]]=None):
    self.views: List[View] = views if views is not None else (stack.views[:] if isinstance(stack, ViewStack) else [View(stack,strides_from_shape(stack),0)])

  def copy(self) -> bool: return ViewStack(self.shape, self.views[:])
  def __repr__(self): return f" ViewStack( shape={self.shape}, views={self.views})"

  @property
  def shape(self) -> Tuple[int,...]: return self.views[-1].shape
  @property
  def strides(self) -> Tuple[int,...]: return self.views[-1].strides
  @property
  def offset(self) -> int: return self.views[-1].offset
  @property
  def contiguous(self) -> bool: return len(self.views) == 1 and _contiguous(tuple(self.shape), tuple(self.strides), self.offset)

  def _pad(self, arg:Tuple[int,...]): raise NotImplementedError (f'not implmented')
  def _shrink(self, arg:Tuple[int,...]): raise NotImplementedError (f'not implmented')
  def _stride(self, arg:Tuple[int,...]): raise NotImplementedError (f'not implmented')

  def _expand(self, new_shape:Tuple[int,...]): 
    #raise NotImplementedError (f'not implmented')
    assert all(isinstance(x, int) for x in new_shape), f"Cannot contain ints in expanded shape {new_shape}"
    assert all( a==b or a==1 for a,b in zip(self.shape, new_shape)), f"Cannot expand {self.shape} into {new_shape} "
    strides: Tuple[int,...] = tuple(s if a==b else 0 for s,(a,b) in zip(self.strides, zip(self.shape, new_shape)))
    self.views[-1] = View(new_shape, strides ,self.offset)


  def _reshape(self, new_shape: Tuple[int,...]): 
    if self.shape == new_shape: return self # optim: excess task
    assert all(isinstance(x, int) and x!=0 for x in new_shape), f'shape {new_shape} must be ints and cannont be 0'
    assert prod(new_shape) == prod(self.shape), f"cannot reshape {new_shape} -> {self.shape}"

    # we are adding or removing a dimension, check if the wrapped shape is the same
    # optim: step used to removed calls in merge views
    if tuple(x for x in self.shape if x!=0) == tuple(x for x in new_shape if x!=0): 
      old_strides = [x for x,y in zip(self.strides,self.shape) if y!=1]  
      new_strides = tuple(x if x!=1 else old_strides.pop(0) for x in zip(self.shape))
      return self

    new_view = View(new_shape, strides_from_shape(new_shape),0)
    if self.contiguous: self.views[-1] = view
    else: self.views.append( new_view  )

  def _permute(self, axis:Tuple[int,...]): 
    assert all(isinstance(x, int) and x>=0 and x<len(self.shape) for x in axis), f'invaild permute for {axis} for {self.shape}'
    assert len(set(axis)) == len(axis) and len(axis) == len(self.shape), f"can't permute {shape} with {axis}"
    self.views[-1] = View(tuple(self.shape[i] for i in axis), tuple(self.strides[i] for i in axis), self.offset) # move axis around in the shape

  def transform_fxn(self, fxn:MDArrayFxn, arg:Union[Tuple[int,...], Tuple[Tuple[[int, int],...]]]) -> ViewStack: 
    #TODO
    #assert isinstance(arg, tuple) and (len(tuple(arg)) == len(self.shape) or fxn == TransformFxns.RESHAPE), f"arg {arg} for fxn {fxn} doesn't match {self.shape}"
    dispatch[fxn](self, arg) 
    return self

dispatch: Dict[TransformFxns: callable] = {
  TransformFxns.RESHAPE: ViewStack._reshape, TransformFxns.PERMUTE: ViewStack._permute, TransformFxns.PAD: ViewStack._pad,
  TransformFxns.EXPAND: ViewStack._expand, TransformFxns.SHRINK: ViewStack._shrink, TransformFxns.STRIDE: ViewStack._stride
}

##########
def get_MDArrays(fxn: MDArrayFxn) -> List[MDArrays]:
  return functools.reduce(operator.add, [get_MDArrays(x) if isinstance(x, MDArrayFxn) else [x] for x in fxn.src], [])

def get_MDArrayFxns(fxn: MDArrayFxn) -> List[MDArrays]:
  return functools.reduce(operator.add, [get_MDArrayFxns(x) for x in fxn.src if isinstance(x, MDArrayFxn)], [fxn])

def map_MDArrays(srcs, fxn:MDArrayFxn) -> MDArrayFxn:
  if fxn in srcs: return map_MDArrays(srcs, srcs[fxn]) if isinstance(srcs[x], MDArrayFxn) else srcs[x] 
  new_srcs = tuple((map_MDArrays(srcs, x) if isinstance(x, MDArrayFxn) else srcs[x]) for x in fxn.src)
  return MDArrayFxn(fxn.fxn, new_srcs, fxn.arg)
##########

def eval_pointwise_fxn(array: MDArray) -> MDArrayFxn: 
  #srcs : Tuple[MDArray, Union[MDArray, MDArrayFxn, None]] = {x : None for x in get_MDArrays(arr.fxn)}
  srcs : Tuple[MDArray, Union[MDArray, MDArrayFxn, None]] = {x : x for x in get_MDArrays(array.fxn)}
  graph = map_MDArrays(srcs, array.fxn)
  return graph

def eval_reduction_fxn(array: MDArray) -> MDArrayFxn: 
  src = array.fxn.src[0]
  #TODO add the optimization
  return MDArrayFxn(array.fxn.fxn, (src,), array.fxn.arg)

class MDArrayFxn(NamedTuple):
  fxn: Union[ops.keys()]
  src: Tuple[Union[MDArray, MDArrayFxn]]
  arg: Any = None


LAZY = True 
AgentTypes = {'CPU': CPUArray}
class MDArray:
  def __init__(self, agent:str, stack:[Union[ViewStack, Int[int,...]]], fxn:MDArrayFxn, dtype):
    self.agent, self.fxn, self.dtype = agent, fxn, dtype
    self.viewstack = stack if isinstance(stack, ViewStack) else ViewStack(tuple(stack))
    self.agent_array:  Optional[InterpretedArray] = None 
    self.stored_array: Optional[InterpretedArray] = None #why do we have 2 ?? arrays?

    if not LAZY: self.compute()  # hmm I what to change this
    pass

  def evaluate(self, required_agent='CPU'):
    if self.agent is None : self.agent = required_agent
    if self.stored_array is None:
      if self.fxn.fxn == LoadFxns.FROMCPU:
        self.stored_array = AgentTypes[self.agent].load(self.fxn.arg) #in theory we can deploy mutliple agents
        graph = MDArrayFxn(LoadFxns.FROMCPU, tuple(), self.fxn.arg)
      elif isinstance(self.fxn.fxn, PointwiseFxns): graph = eval_pointwise_fxn(self)
      elif isinstance(self.fxn.fxn, ReductionFxns): graph = eval_reduction_fxn(self)
      elif isinstance(self.fxn.fxn, TransformFxns): 
        # fuse RESHAPE and ReduceOps - this is sort of a hack for IMAGE, otherwise it shouldn't matter
        #if src.agent_arr is None and type(src.fxn.fxn) == ReduceFxns and self.fxn.fxn == ChangeFxns.RESHAPE and len(src.children) <= 1
        # TransformFxns aren't an AST, just run them
        real_src  = self.fxn.src[0].evaluate(self.agent)
        self.stored_array = real_src.transform_fxn(self.fxn.fxn, self.fxn.arg)
        graph = MDArrayFxn(self.fxn.fxn, (real_src))

      if self.stored_array is None:
        graph = map_MDArrays({x : x.evaluate() for x in get_MDArrays(graph)}, graph)   
        self.stored_array = AgentTypes[self.agent].exec_graph(graph, output_array=self.agent_array)
    return self.stored_array

  @property
  def shape(self): return self.viewstack.shape

  @staticmethod
  def load(data, agent='CPU') -> MDArray: return MDArray(agent, data.shape, MDArrayFxn(LoadFxns.FROMCPU, tuple(), data.copy()), dtypes.from_numpy(data))
  def compute(self): return self.evaluate().compute()
  # --------------------------------
  # these are methods are just for testing. But removed numpy from it
  def __add__(a, b): return function(PointwiseFxns.ADD, (a,b))
  def __sub__(a, b): return function(PointwiseFxns.SUB, (a,b))
  def __mul__(a, b): return function(PointwiseFxns.MUL, (a,b))
  def __truediv__(a, b): return function(PointwiseFxns.DIV, (a,b))
  def __radd__(a, b): return function(PointwiseFxns.ADD, (b,a))
  def __rsub__(a, b): return function(PointwiseFxns.SUB, (b,a))
  def __rmul__(a, b): return function(PointwiseFxns.MUL, (b,a))
  def __rtruediv__(a, b): return function(PointwiseFxns.DIV, (b,a))
  def __pow__(self, other): return function(PointwiseFxns.DIV, None, self, other)
  def __neg__(self): return function(PointwiseFxns.NEG, None, self)

  @classmethod
  def random(MDArray, *shape, **kwargs): return MDArray.load(np.random.uniform(size=shape).astype('f'), **kwargs)
  @classmethod
  def zeros(MDArray, *shape, **kwargs): return MDArray.load(np.zeros(shape,dtype=np.float32), **kwargs) 

  # --------------------------------
# TODO: think about creating a single return statemnet, we are just changing the shape, and argumets
def function(ctx:OPERATOR, srcs:Tuple[MDArray], arg:Any=None):
  assert len(srcs) >= 1, f'No srcs found'
  array, agent, shape, dtype = srcs[0], srcs[0].agent, srcs[0].shape, max(x.dtype for x in srcs) # only works with Dtype(NamedTuple)
  if isinstance(ctx, PointwiseFxns): return MDArray(agent, shape, MDArrayFxn(ctx, srcs, None), dtype) 
  if isinstance(ctx, ReductionFxns): 
    new_shape = tuple(arg)
    if shape == new_shape: return array
    return MDArray(agent, new_shape, MDArrayFxn(ctx, srcs, new_shape), dtype)
  if isinstance(ctx, TransformFxns): 
    if ctx == TransformFxns.RESHAPE and shape == arg: return array # optim: excess task

    local_viewstack = ViewStack(shape).transform_fxn(ctx, arg)
    if local_viewstack.contiguous and local_viewstack.shape == shape: return array # optim: excess task
    if array.agent_array is None and array.fxn.fxn == ctx: # optim: merge consecutive fxn if they are the same 
      if ctx in [TransformFxns.RESHAPE, TransformFxns.EXPAND, TransformFxns.SHRINK]: return array.fxn.src[0].transform_fxn(ctx, arg)
      if ctx == TransformFxns.PERMUTE: return array.fxn.srcs[0].transform_fxn(ctx, tuple(array.fxn.arg[i] for i in arg)) #hmmm
      #if ctx == TransformFxns.STRIDE: return array.fxn.srcs[0].transform_fxn(ctx, tuple((a+x, b+y) for (a,b),(x,y) in zip(fxn.arg, arg)))
      #if ctx == TransformFxns.PAD: return array.fxn.srcs[0].transform_fxn(ctx, tuple(i*j for i,j in zip(arg, ctx.arg))) #hmmm

    ret = MDArray(agent, ViewStack(array.viewstack).transform_fxn(ctx, arg), MDArrayFxn(ctx, srcs, arg), dtype)
    return ret
  else: Exception(f'Operation {ctx} is not supported')

def dot(A:Array, B:Array) -> MDArray:
  A = reshape(A, *A.shape[0:-1], 1, A.shape[-1]) # convert all matrix to vectors based on rows
  B = reshape(B, *B.shape[0:-2], 1, B.shape[-2], B.shape[-1]) # convert all matrix to vectors based on cols
  B = transpose(B, -2,-1)
  return reshape(_sum(mul(A,B), axis=-1), *A.shape[0:-2], -1)

def transpose(array:MDArray, axis1=1, axis2=0) -> MDArray:
  order = list(range(len(array.shape)))
  order[axis1], order[axis2] = order[axis2], order[axis1]
  return function(OPERATOR.PERMUTE, (array,), tuple(order))

def reshape(array:MDArray, shape:Tuple[int,...], *args) -> MDArray:
  new_shape = fixArgsType(shape, *args)
  assert len(new_shape)>0 and all(d!=0 for d in new_shape), f'Zeros are not allowed in shape {new_shape}'# we can't remove dimensions
  return function(OPERATOR.RESHAPE, (array,), tuple(-prod(array.shape)//prod(new_shape) if d==-1 else d for d in new_shape))

def expand(array:MDArray, shape, *args) -> MDArray:
  return function(OPERATOR.EXPAND, (array,),  tuple(x if d!=-1 else s for d,x in zip(array.shape, fixArgsType(shape, *args))))

def fixArgsType(*args): return tuple() if len(args)==0 else tuple(args[0]) if isinstance(args[0], (list,tuple)) else tuple(args)
def _sum(array:MDArray, axis=None, keepdims=False): return _reduce(array, OPERATOR.SUM, axis, keepdims)
def _max(array:MDArray, axis=None, keepdims=False): return _reduce(array, OPERATOR.MAX, axis, keepdims)

def mul(A:MDArray, B:MDArray): return _broadcast(OPERATOR.MUL, (A,B))

def _reduce(array:MDArray, fxn:OPERATOR, axis:Optiona[Union[int, Tuple[int,int]]]=None, keepdims=False) -> MDArray:
  _axis:List[int] = (tuple(range(len(array.shape)))) if axis is None else ([axis] if isinstance(axis, int) else list(axis)) # fix input type
  _axis = [x if x>=0 else x+len(array.shape) for x in _axis]  # cleanup negative axis by wrapping
  new_shape = tuple(1 if i in _axis else array.shape[i] for i in range(len(array.shape)))        # 
  ret = function(fxn, (array,), new_shape)
  return ret if keepdims else reshape(ret, shape=([1] if new_shape==[] else new_shape)) ##????

def _broadcast(fxn:OPERATOR, srcs:Tuple[Union[MDArray, float]], reverse:bool=False) -> MDArray:
  srcs = [x if isinstance(x, MDArray) else MDArray.load(x) for x in srcs]                             # type correction
  srcs = [reshape(x, [1]*(max(len(d.shape) for d in srcs)-len(x.shape)) + list(x.shape)) for x in srcs]  # 'pad' smallest dim based on largest dim of our arrays
  shape_ret = tuple( max(x.shape[i] for x in srcs) for i in range(len(srcs[0].shape)))
  return function(fxn, tuple(expand(x,shape_ret) for x in srcs))


if __name__ == '__main__':
  import unittest
  class TestMDArray(unittest.TestCase):
    def test1(self): # ---------------------- Test 1: Testing PointwiseFxn functions
      a,b,c = [np.random.uniform(size=(1,3,3)).astype('f') for x in range(3)]
      A,B,C = [MDArray.load(x) for x in (a,b,c)]
      for fxn in [operator.add, operator.sub, operator.mul, operator.truediv]:
        assert (fxn(A,B).compute() == fxn(a,b)).all(), f'Error with Pointwise function {fxn}, {pred} != {targ}'

    def test2(self): # ---------------------- Test 2: Testing TransformFxns functions
      a,b,c = [np.random.uniform(size=(1,3,3)).astype('f') for x in range(3)]
      A,B,C = [MDArray.load(x) for x in (a,b,c)]

      C = function(TransformFxns.PERMUTE, (A,), (0,2,1)) # NOTE arg= new axes
      pred, targ = C.compute(), a.transpose(0,2,1)
      assert (pred == targ).all(), f'Error with ChangeFxns.PERMUTE, {pred} != {targ}'

      C = function(TransformFxns.RESHAPE, (A,), (1,1,9)) # NOTE arg= new shape
      pred, targ = C.compute(), a.reshape((1,1,9)) 
      assert (pred == targ).all(), f'Error with ChangeFxns.RESHAPE, {pred} != {targ}'

    def test3(self): # ---------------------- Test 3:  Tesitng ReduceFxns
      a,b,c = [np.random.uniform(size=(1,3,3)).astype('f') for x in range(3)]
      A,B,C = [MDArray.load(x) for x in (a,b,c)]
      C = function(ReductionFxns.SUM, (A,), (1,3,1))

      pred, targ = C.compute(), a.sum(-1, keepdims=True)  
      assert (pred == targ).all(), f'Error with ReduceFxns, {pred} != {targ}'
      #pred, targ = sum(C,axis=-1).compute(), a.sum(axis=-1)
      #assert (pred == targ).all(), f'Error with def sum() , {pred} != {targ}'

    def test4(self): # ---------------------- Test 4: Creating Matmul
      a,b = [np.random.uniform(size=(1,3,3)).astype('f') for x in range(2)]
      A,B = [MDArray.load(x) for x in (a,b)]
      pred, targ = dot(A,B).compute(), np.matmul(a,b)
      #assert (pred == targ).all(), f'Error with dot, \n{pred} != \n{targ}'
      assert np.allclose(pred, targ), f'Error with dot, {pred} != {targ}'
  unittest.main()
