import cupy as cp  # type: ignore
from .base import Backend 


class CuPyBackend(Backend): 
    
    float64 = cp.float64
    device = "cuda"
    
    def asarray(self, data, dtype=None):
        return cp.asarray(data, dtype=dtype)

    def zeros_like(self, x):
        return cp.zeros_like(x)

    def ones_like(self, x):
        return cp.ones_like(x)

    def zeros(self, shape, dtype=None):
        return cp.zeros(shape, dtype=dtype)

    def sum(self, x, axis=None, keepdims=False):
        return cp.sum(
            x,
            axis=axis,
            keepdims=keepdims,
        )

    def max(self, x, axis=None, keepdims=False):
        return cp.max(
            x,
            axis=axis,
            keepdims=keepdims,
        )

    def exp(self, x):
        return cp.exp(x)

    def log(self, x):
        return cp.log(x)

    def sqrt(self, x):
        return cp.sqrt(x)

    def abs(self, x):
        return cp.abs(x)

    def tanh(self, x):
        return cp.tanh(x)

    def maximum(self, x, y):
        return cp.maximum(x, y)

    def minimum(self, x, y):
        return cp.minimum(x, y)

    def clip(self, x, a, b):
        return cp.clip(x, a, b)

    def sign(self, x):
        return cp.sign(x)

    def reshape(self, x, shape):
        return cp.reshape(x, shape)

    def squeeze(self, x, axis=None):
        return cp.squeeze(x, axis=axis)

    def expand_dims(self, x, axis):
        return cp.expand_dims(x, axis=axis)

    def swapaxes(self, x, axis1, axis2):
        return cp.swapaxes(x, axis1, axis2)

    def broadcast_to(self, x, shape):
        return cp.broadcast_to(x, shape)

    def matmul(self, x, y):
        return cp.matmul(x, y)

    def add_at(self, x, indices, values):
        cp.add.at(x, indices, values)

    def where(self, condition, x, y):
        return cp.where(condition, x, y)
    
    def to_cpu(self, x):
        return cp.asnumpy(x)
    
    def to_device(self, x):
        return cp.asarray(x)
    