# -*- coding:utf8 -*-
import os

import idx2numpy as idx2np
import numpy as np

from utensor_cgen.logger import logger
from utensor_cgen.transformer.optimizer import RefCntOptimizer
from utensor_cgen.utils import NamescopedKWArgsParser

from .snippets import *  # pylint: disable=W0401,W0614


class OperatorFactory():
  # Can easily do something smarter
  _operators = {}

  def createOperatorSnippet(self, op_info, **kwargs):
    op_type = op_info.op_type
    if op_type not in self._operators:
      err_msg = "unsupported op type in uTensor: {}".format(op_type)
      raise ValueError(err_msg)

    op = self._operators[op_type](op_info, **kwargs)  # Create desired object
    return op.snippet  # Ops know how to create their snippets

  @classmethod
  def register(cls, op_cls):
    cls._operators[op_cls.op_type] = op_cls

  @classmethod
  def support_op_types(cls):
    """Return the list of all supported ops
    """
    return list(cls._operators.keys())


class _Operator(object):
  def __init__(self):
    self.name = ""
    self._snippet = None

  @property
  def snippet(self):
    return self._snippet


@OperatorFactory.register
class _AddOperator(_Operator):

  op_type = "Add" # tf op type

  def __init__(self, op_info, **kwargs):
    _Operator.__init__(self)
    inputs = [tensor_info.name for tensor_info in op_info.input_tensors]
    output = op_info.output_tensors[0].name
    tf_dtype = op_info.input_tensors[0].dtype
    parser = NamescopedKWArgsParser(RefCntOptimizer.KWARGS_NAMESCOPE, 
                                    op_info.op_attr)
    ref_count = parser.get('ref_counts', [0])[0]
    to_eval = parser.get('to_eval', False)
    self._snippet = AddOpSnippet(inputs, output, tf_dtype, ref_count, to_eval)


@OperatorFactory.register
class _ArgMaxOperator(_Operator):

  op_type = "ArgMax"

  def __init__(self, op_info, **kwargs):
    _Operator.__init__(self)
    inputs = [tensor_info.name for tensor_info in op_info.input_tensors]
    out_tensor_info = op_info.output_tensors[0]
    output, out_dtype = out_tensor_info.name, out_tensor_info.dtype
    in_dtype = op_info.input_tensors[0].dtype
    parser = NamescopedKWArgsParser(RefCntOptimizer.KWARGS_NAMESCOPE, 
                                    op_info.op_attr)
    ref_count = parser.get('ref_counts', [0])[0]
    to_eval = parser.get('to_eval', False)
    self._snippet = ArgMaxOpSnippet(inputs, output, in_dtype, out_dtype, ref_count, to_eval)


@OperatorFactory.register
class _DequantizeOperator(_Operator):

  op_type = "Dequantize"

  def __init__(self, op_info, **kwargs):
    _Operator.__init__(self)
    inputs = [tensor_info.name for tensor_info in op_info.input_tensors]
    out_tensor_info = op_info.output_tensors[0]
    output, out_dtype = out_tensor_info.name, out_tensor_info.dtype
    parser = NamescopedKWArgsParser(RefCntOptimizer.KWARGS_NAMESCOPE, 
                                    op_info.op_attr)
    ref_count = parser.get('ref_counts', [0])[0]
    to_eval = parser.get('to_eval', False)
    self._snippet = DequantizeOpSnippet(inputs, output, out_dtype, ref_count, to_eval)


@OperatorFactory.register
class _MaxOperator(_Operator):

  op_type = "Max"

  def __init__(self, op_info, **kwargs):
    _Operator.__init__(self)
    inputs = [tensor_info.name for tensor_info in op_info.input_tensors]
    out_tensor_info = op_info.output_tensors[0]
    output, out_dtype, out_shape = (out_tensor_info.name,
                                    out_tensor_info.dtype,
                                    out_tensor_info.shape)
    # FIXME: automatic alloc for uTensor fail
    if not out_shape:
      out_shape = [1]
    parser = NamescopedKWArgsParser(RefCntOptimizer.KWARGS_NAMESCOPE, 
                                    op_info.op_attr)
    ref_count = parser.get('ref_counts', [0])[0]
    to_eval = parser.get('to_eval', False)
    self._snippet = MaxOpSnippet(inputs, output, out_dtype, out_shape, ref_count, to_eval)


@OperatorFactory.register
class _QuantizedMaxPool(_Operator):

  op_type = "QuantizedMaxPool"

  def __init__(self, op_info, **kwargs):
    _Operator.__init__(self)
    inputs = [tensor_info.name for tensor_info in op_info.input_tensors]
    outputs = [tensor_info.name for tensor_info in op_info.output_tensors]
    dtype = op_info.output_tensors[0].dtype
    ksize = op_info.op_attr['ksize'].value.ints_value
    strides = op_info.op_attr['strides'].value.ints_value
    padding = op_info.op_attr['padding'].value.decode('utf8')
    parser = NamescopedKWArgsParser(RefCntOptimizer.KWARGS_NAMESCOPE, 
                                    op_info.op_attr)
    ref_counts = parser.get('ref_counts', [])
    to_eval = parser.get('to_eval', False)
    self._snippet = QuantizedMaxPoolSnippet(inputs, outputs, dtype,
                                            ksize, strides, padding,
                                            ref_counts, to_eval)


@OperatorFactory.register
class _MinOperator(_Operator):

  op_type = "Min"

  def __init__(self, op_info, **kwargs):
    _Operator.__init__(self)
    inputs = [tensor_info.name for tensor_info in op_info.input_tensors]
    out_info = op_info.output_tensors[0]
    output, out_dtype, out_shape = (out_info.name,
                                    out_info.dtype,
                                    out_info.shape)
    # FIXME: automatic alloc for uTensor fail
    if not out_shape:
      out_shape = [1]
    parser = NamescopedKWArgsParser(RefCntOptimizer.KWARGS_NAMESCOPE,
                                    op_info.op_attr)
    ref_count = parser.get('ref_counts', [0])[0]
    to_eval = parser.get('to_eval', False)
    self._snippet = MinOpSnippet(inputs, output, out_dtype, out_shape, ref_count, to_eval)


@OperatorFactory.register
class _QuantizeV2Operator(_Operator):

  op_type = "QuantizeV2"

  def __init__(self, op_info, **kwargs):
    _Operator.__init__(self)
    inputs = [tensor_info.name for tensor_info in op_info.input_tensors]
    outputs = [tensor_info.name for tensor_info in op_info.output_tensors]
    out_dtype = op_info.output_tensors[0].dtype
    parser = NamescopedKWArgsParser(RefCntOptimizer.KWARGS_NAMESCOPE,
                                    op_info.op_attr)
    ref_counts = parser.get('ref_counts', [])
    to_eval = parser.get('to_eval', False)
    self._snippet = QuantizeV2OpSnippet(inputs, outputs, out_dtype, ref_counts, to_eval)


@OperatorFactory.register
class _QuantizedMatMulOperator(_Operator):

  op_type = "QuantizedMatMul"

  def __init__(self, op_info, **kwargs):
    _Operator.__init__(self)
    inputs = [tensor_info.name for tensor_info in op_info.input_tensors]
    outputs = [tensor_info.name for tensor_info in op_info.output_tensors]
    in_tensor_info = op_info.input_tensors[0]
    x_dtype, w_dtype, out_dtype = (op_info.input_tensors[0].dtype,
                                   op_info.input_tensors[1].dtype,
                                   op_info.output_tensors[0].dtype)
    parser = NamescopedKWArgsParser(RefCntOptimizer.KWARGS_NAMESCOPE,
                                    op_info.op_attr)
    ref_counts = parser.get('ref_counts', [])
    to_eval = parser.get('to_eval', False)
    self._snippet = QuantizedMatMulOpSnippet(inputs, outputs,
                                             x_dtype, w_dtype, out_dtype, 
                                             ref_counts, to_eval)


@OperatorFactory.register
class _QuantizedReluOperator(_Operator):

  op_type = "QuantizedRelu"

  def __init__(self, op_info, **kwargs):
    _Operator.__init__(self)
    inputs = [tensor_info.name for tensor_info in op_info.input_tensors]
    outputs = [tensor_info.name for tensor_info in op_info.output_tensors]
    in_dtype, qout_dtype = (op_info.input_tensors[0].dtype,
                            op_info.output_tensors[0].dtype)  #NT: why separate this out?
                                                              #DB: I don't know, it's in the uTensor C code
    out_dtypes = [tensor_info.dtype for tensor_info in op_info.output_tensors[1:]]
    parser = NamescopedKWArgsParser(RefCntOptimizer.KWARGS_NAMESCOPE,
                                    op_info.op_attr)
    ref_counts = parser.get('ref_counts', [])
    to_eval = parser.get('to_eval', False)
    self._snippet = QuantizedReluOpSnippet(inputs, outputs, in_dtype,
                                           out_dtypes, qout_dtype, 
                                           ref_counts, to_eval)


@OperatorFactory.register
class _QuantizedAddOperator(_Operator):

  op_type = "QuantizedAdd"

  def __init__(self, op_info, **kwargs):
    _Operator.__init__(self)
    inputs = [tensor_info.name for tensor_info in op_info.input_tensors]
    outputs = [tensor_info.name for tensor_info in op_info.output_tensors]
    x_dtype, w_dtype, out_dtype = (op_info.input_tensors[0].dtype,
                                   op_info.input_tensors[1].dtype,
                                   op_info.output_tensors[0].dtype)
    parser = NamescopedKWArgsParser(RefCntOptimizer.KWARGS_NAMESCOPE,
                                    op_info.op_attr)
    ref_counts = parser.get('ref_counts', [])
    to_eval = parser.get('to_eval', False)
    self._snippet = QuantizedAddOpSnippet(inputs, outputs, 
                                          x_dtype, w_dtype, out_dtype, 
                                          ref_counts, to_eval)


@OperatorFactory.register
class _RequantizationRangeOperator(_Operator):

  op_type = "RequantizationRange"

  def __init__(self, op_info, **kwargs):
    _Operator.__init__(self)
    inputs = [tensor_info.name for tensor_info in op_info.input_tensors]
    outputs = [tensor_info.name for tensor_info in op_info.output_tensors]
    out_dtype = op_info.output_tensors[0].dtype
    parser = NamescopedKWArgsParser(RefCntOptimizer.KWARGS_NAMESCOPE,
                                    op_info.op_attr)
    ref_counts = parser.get('ref_counts', [])
    to_eval = parser.get('to_eval', False)
    self._snippet = RequantizationRangeOpSnippet(inputs, outputs, out_dtype, 
                                                 ref_counts, to_eval)


@OperatorFactory.register
class _RequantizeOperator(_Operator):

  op_type = "Requantize"
  def __init__(self, op_info, **kwargs):
    _Operator.__init__(self)
    inputs = [tensor_info.name for tensor_info in op_info.input_tensors]
    outputs = [tensor_info.name for tensor_info in op_info.output_tensors]
    qout_dtype = op_info.output_tensors[0].dtype
    range_dtype = op_info.output_tensors[1].dtype
    parser = NamescopedKWArgsParser(RefCntOptimizer.KWARGS_NAMESCOPE,
                                    op_info.op_attr)
    ref_counts = parser.get('ref_counts', [])
    to_eval = parser.get('to_eval', False)
    self._snippet = RequantizeOpSnippet(inputs, outputs,
                                        qout_dtype, range_dtype,
                                        ref_counts, to_eval)


@OperatorFactory.register
class _ReshapeOperator(_Operator):

  op_type = "Reshape"

  def __init__(self, op_info, **kwargs):
    _Operator.__init__(self)
    inputs = [tensor_info.name for tensor_info in op_info.input_tensors]
    output = op_info.output_tensors[0].name
    parser = NamescopedKWArgsParser(RefCntOptimizer.KWARGS_NAMESCOPE,
                                    op_info.op_attr)
    ref_count = parser.get('ref_counts', [0])[0]
    to_eval = parser.get('to_eval', False)
    self._snippet = ReshapeOpSnippet(inputs, output, ref_count, to_eval)


@OperatorFactory.register
class _QuantizedReshapeOperator(_Operator):

  op_type = "QuantizedReshape"

  def __init__(self, op_info, **kwargs):
    _Operator.__init__(self)
    inputs = [tensor_info.name for tensor_info in op_info.input_tensors]
    outputs = [tensor_info.name for tensor_info in op_info.output_tensors]
    parser = NamescopedKWArgsParser(RefCntOptimizer.KWARGS_NAMESCOPE,
                                    op_info.op_attr)
    ref_counts = parser.get('ref_counts', [])
    to_eval = parser.get('to_eval', False)
    self._snippet = QuantizedReshapeOpSnippet(inputs=inputs,
                                              outputs=outputs,
                                              ref_counts=ref_counts,
                                              to_eval=to_eval)


@OperatorFactory.register
class _Conv2DOperator(_Operator):

  op_type = "QuantizedConv2D"

  def __init__(self, op_info, **kwargs):
    _Operator.__init__(self)
    inputs = [tensor_info.name for tensor_info in op_info.input_tensors]
    outputs = [tensor_info.name for tensor_info in op_info.output_tensors]
    in_dtype, filter_dtype = (op_info.input_tensors[0].dtype,
                              op_info.input_tensors[1].dtype)
    out_dtypes = [tensor_info.dtype for tensor_info in op_info.output_tensors]
    strides = op_info.op_attr["strides"].value.ints_value
    padding = op_info.op_attr["padding"].value.decode('utf8')
    parser = NamescopedKWArgsParser(RefCntOptimizer.KWARGS_NAMESCOPE,
                                    op_info.op_attr)
    ref_counts = parser.get('ref_counts', [])
    to_eval = parser.get('to_eval', False)
    self._snippet = Conv2DOpSnippent(inputs, outputs, strides, padding,
                                     in_dtype=in_dtype, filter_dtype=filter_dtype, out_dtypes=out_dtypes,
                                     ref_counts=ref_counts, to_eval=to_eval)


@OperatorFactory.register
class _ConstOperator(_Operator):

  op_type = "Const"

  def __init__(self, op_info, **kwargs):
    out_tensor_info = op_info.output_tensors[0]
    out_tname, out_dtype = (out_tensor_info.name,
                            out_tensor_info.dtype)
    parser = NamescopedKWArgsParser(RefCntOptimizer.KWARGS_NAMESCOPE,
                                    op_info.op_attr)
    ref_count = parser.get('ref_counts', [0])[0]
    pre_tname = self._tf_prepare_tensor_name(out_tname)
    idx_fname = "{}.idx".format(pre_tname)
    idx_dir = kwargs['idx_dir']
    embed_data_dir = kwargs.get('embed_data_dir',
                                os.path.join("/fs", idx_dir))
    self._snippet = CreateTensorIdxSnippet(embed_data_dir, out_tname,
                                           idx_fname=idx_fname,
                                           np_dtype=out_dtype,
                                           ref_count=ref_count)
    idx_path = os.path.join(idx_dir, idx_fname)
    value = op_info.op_attr['value'].value
    self._tf_save_data(idx_path, value)

  def _tf_prepare_tensor_name(self, tensor_name):
    """Replace all ':' and '/' with '_' in a given tensor name
    """
    prepared = tensor_name.replace(":", "_").replace("/", "_")
    return prepared
  
  def _tf_save_data(self, path, value):
    np_array = value.np_array
    if np_array.shape == ():
      np_array = np.array([np_array])
    with open(path, "wb") as fid:
      idx2np.convert_to_file(fid, np_array)
    logger.info("saving %s", path)


@OperatorFactory.register
class _InlineOperator(_Operator):

  op_type = "Inline"
  
  def __init__(self, op_info, **kwargs):
    out_tensor_info = op_info.output_tensors[0]
    out_tname, out_dtype, tensor_shape = (out_tensor_info.name,
                            out_tensor_info.dtype,
                            out_tensor_info.shape)
    parser = NamescopedKWArgsParser(RefCntOptimizer.KWARGS_NAMESCOPE,
                                    op_info.op_attr)
    ref_count = parser.get('ref_counts', [0])[0]
    pre_tname = self._prepare_tensor_name(out_tname)
    inline_tname = self._prepare_inline_array_name(out_tname)
    value = op_info.op_attr['value'].value.np_array.flatten()
    self._snippet = CreateTensorBinarySnippet(out_tname, tensor_shape=tensor_shape,
                                         tf_dtype=out_dtype,
                                         sptr_name=pre_tname,
                                         inline_name=inline_tname,
                                         ref_count=ref_count)

    weight_snippet = WeightSnippet(inline_tname,
                                  out_dtype,
                                  tensor_shape,
                                  value)
    weight_container = kwargs['weight_container']                             
    weight_container.add_snippet(weight_snippet)

  def _prepare_tensor_name(self, tensor_name):
    prepared = tensor_name.replace(":", "_").replace("/", "_")
    return prepared

  def _prepare_inline_array_name(self, tensor_name):
    inline = tensor_name.replace(":", "_").replace("/", "_")
    preapred = "inline_{}".format(inline)
    return preapred
