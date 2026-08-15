import numpy as np 

from .base import Backend 


class NumPyBackend(Backend): 
    
    float64 = np.float64
    device = "cpu"
    
    def asarray(self, data, dtype=None):
        return np.asarray(data, dtype=dtype)

    def zeros_like(self, x):
        return np.zeros_like(x)

    def ones_like(self, x):
        return np.ones_like(x)

    def zeros(self, shape, dtype=None):
        return np.zeros(shape, dtype=dtype)

    def sum(self, x, axis=None, keepdims=False):
        return np.sum(
            x,
            axis=axis,
            keepdims=keepdims,
        )

    def max(self, x, axis=None, keepdims=False):
        return np.max(
            x,
            axis=axis,
            keepdims=keepdims,
        )

    def exp(self, x):
        return np.exp(x)

    def log(self, x):
        return np.log(x)

    def sqrt(self, x):
        return np.sqrt(x)

    def abs(self, x):
        return np.abs(x)

    def tanh(self, x):
        return np.tanh(x)

    def maximum(self, x, y):
        return np.maximum(x, y)

    def minimum(self, x, y):
        return np.minimum(x, y)

    def clip(self, x, a, b):
        return np.clip(x, a, b)

    def sign(self, x):
        return np.sign(x)

    def reshape(self, x, shape):
        return np.reshape(x, shape)

    def squeeze(self, x, axis=None):
        return np.squeeze(x, axis=axis)

    def expand_dims(self, x, axis):
        return np.expand_dims(x, axis=axis)

    def swapaxes(self, x, axis1, axis2):
        return np.swapaxes(x, axis1, axis2)

    def broadcast_to(self, x, shape):
        return np.broadcast_to(x, shape)

    def matmul(self, x, y):
        return np.matmul(x, y)

    def add_at(self, x, indices, values):
        np.add.at(x, indices, values)

    def where(self, condition, x, y):
        return np.where(condition, x, y)
    
    def to_cpu(self, x):
        return x 
    
    def to_device(self, x):
        return x 