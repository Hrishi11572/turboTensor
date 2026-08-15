class Backend:

    def asarray(self, data, dtype=None):
        raise NotImplementedError

    def zeros_like(self, x):
        raise NotImplementedError

    def ones_like(self, x):
        raise NotImplementedError

    def zeros(self, shape, dtype=None):
        raise NotImplementedError

    def sum(self, x, axis=None, keepdims=False):
        raise NotImplementedError

    def max(self, x, axis=None, keepdims=False):
        raise NotImplementedError

    def exp(self, x):
        raise NotImplementedError

    def log(self, x):
        raise NotImplementedError

    def sqrt(self, x):
        raise NotImplementedError

    def abs(self, x):
        raise NotImplementedError

    def tanh(self, x):
        raise NotImplementedError

    def maximum(self, x, y):
        raise NotImplementedError

    def minimum(self, x, y):
        raise NotImplementedError

    def clip(self, x, a, b):
        raise NotImplementedError

    def sign(self, x):
        raise NotImplementedError

    def reshape(self, x, shape):
        raise NotImplementedError

    def squeeze(self, x, axis=None):
        raise NotImplementedError

    def expand_dims(self, x, axis):
        raise NotImplementedError

    def swapaxes(self, x, axis1, axis2):
        raise NotImplementedError

    def broadcast_to(self, x, shape):
        raise NotImplementedError

    def matmul(self, x, y):
        raise NotImplementedError

    def add_at(self, x, indices, values):
        raise NotImplementedError

    def where(self, condition, x, y):
        raise NotImplementedError

    def to_cpu(self, x): 
        raise NotImplementedError
    
    def to_device(self, x): 
        raise NotImplementedError 