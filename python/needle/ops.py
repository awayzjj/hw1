"""Operator implementations."""

from numbers import Number
from typing import Optional, List
from .autograd import NDArray
from .autograd import Op, Tensor, Value, TensorOp
from .autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks
import numpy as array_api


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return array_api.power(a, self.scalar)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return array_api.divide(a, b)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, rhs = node.inputs
        return out_grad / rhs, -(out_grad * lhs) / (rhs * rhs)
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a / self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad / self.scalar
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        if self.axes:
            return array_api.swapaxes(a, self.axes[0], self.axes[1])
        else:
            return array_api.swapaxes(a, -1, -2)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return transpose(out_grad, self.axes)
        ### END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.reshape(a, self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # 不是self.shape，因为要和输入的shape一致，self.shape!=输入的shape
        return reshape(out_grad, node.inputs[0].shape)
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.broadcast_to(a, self.shape)

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        input_shape = node.inputs[0].shape
        input_shape_len = len(input_shape) - 1
        self.reduce_dim = []
        # broadcast后的shape从右往左遍历
        for idx in range(len(out_grad.shape)-1, -1, -1): 
            """
            如果当前没有input_shape了
                 |  
            A:     3 3
            B: 4 4 3 3
            则把当前dim也添加到reduce_dim中
            """
            if input_shape_len < 0: 
                self.reduce_dim.append(idx)
                continue
            # 否则取broadcast后的dim，和input的dim            
            broadcast_dim_size = self.shape[idx]
            input_dim_size = input_shape[input_shape_len]
            
            # 比较是否相等，如果不等，说明发生了broadcast，需要将当前dim添加到reduce_dim
            if broadcast_dim_size != input_dim_size: 
                self.reduce_dim.append(idx)
            input_shape_len -= 1
        # 做reduce_sum，并且保证与原始输入shape一致
        return reshape(summation(out_grad, tuple(self.reduce_dim)), input_shape)
        ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.sum(a, axis=self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        #input_shape = node.inputs[0].shape
        #broadcast_shape = list(input_shape)
        #if self.axes: 
        #    # 对reduce的维度，设置shape为1
        #    for i in self.axes: 
        #        broadcast_shape[i] = 1
        ## 如果没指定axes，那么就是对所有维度reduce了，如sum(tensor((4, 4))) -> 1
        ## 我们构造出broadast_shape为(1, 1)
        #else: 
        #    broadcast_shape = [1 for _ in range(len(broadcast_shape))] # (1, 1, 1, 1)
        ## 对out_grad reshape一下
        #out_grad = reshape(out_grad, broadcast_shape)
        ## 借助numpy的广播机制
        #return out_grad * array_api.ones(input_shape, dtype=array_api.float32)

        ipt = node.inputs[0]
        if self.axes:
            grad = array_api.expand_dims(out_grad.cached_data, self.axes)
            if isinstance(self.axes, int):
                repeat = ipt.shape[self.axes]
                grad = array_api.repeat(grad, repeat, self.axes)
            else:
                repeat = []
                for i in self.axes:
                    repeat.append(ipt.shape[i])
                for r, a in zip(repeat, self.axes):
                    grad = array_api.repeat(grad, r, a)
        else:
            grad = array_api.ones_like(ipt.cached_data) * out_grad.cached_data

        grad = Tensor(grad)

        return [grad]

        ### END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return array_api.matmul(a, b)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, rhs = node.inputs
        lhs_tmp_grad = matmul(out_grad, transpose(rhs))
        rhs_tmp_grad = matmul(transpose(lhs), out_grad)
        if lhs_tmp_grad.shape != lhs.shape: 
            # Need Reduce
            lhs_tmp_grad = summation(lhs_tmp_grad, axes=tuple(range(len(lhs_tmp_grad.shape) - 2)))
        if rhs_tmp_grad.shape != rhs.shape: 
            # Need Reduce
            rhs_tmp_grad = summation(rhs_tmp_grad, axes=tuple(range(len(rhs_tmp_grad.shape) - 2)))
        return lhs_tmp_grad, rhs_tmp_grad
        ### END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.negative(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return -out_grad 
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.log(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad / node.inputs[0]
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.exp(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad * exp(node.inputs[0])
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


# TODO
class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        # relu_mask是numpy
        relu_mask = a > 0 
        return a * relu_mask
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        #relu_mask = Tensor((node.inputs[0].cached_data > 0).astype(array_api.float32))
        relu_mask = Tensor((node.inputs[0].cached_data > 0))
        # tensor float64 numpy float64
        return out_grad * relu_mask
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)

