from ..backend import default_backend, get_backend

class Tensor:
    def __init__(self, data, _children=(), _op='', backend=None):
        self.backend = (
            backend 
            if backend is not None 
            else default_backend 
        )
        
        self.data = self.backend.asarray(data, dtype=self.backend.float64)
        self.grad = self.backend.zeros_like(self.data)
        self._prev = set(_children)
        self._op = _op 
        self._backward = lambda : None
        
        
    def __repr__(self): 
        return f'Tensor(shape={self.data.shape}, dtype={self.data.dtype})'

    def to(self, device):
        if device == self.device:
            return self

        backend = get_backend(device)

        data = backend.to_device(self.data)

        return Tensor(
            data,
            backend=backend,
        )
    
    def unbroadcast(self, grad, original_shape):
        # Remove extra leading dimensions introduced by broadcasting
        while len(grad.shape) > len(original_shape):
            grad = self.backend.sum(
                grad,
                axis=0
            )
        # Dimensions that were originally 1 were broadcast,
        # so sum across those dimensions.
        for axis, size in enumerate(original_shape):

            if size == 1 and grad.shape[axis] != 1:
                grad = self.backend.sum(
                    grad,
                    axis=axis,
                    keepdims=True
                )

        return grad

    
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, backend=self.backend) 
            
        t = self.data + other.data 
        out = Tensor(t , (self, other), _op='+', backend=self.backend)

        def _backward():
            self.grad += self.unbroadcast(out.grad, self.data.shape)
            other.grad += self.unbroadcast(out.grad , other.data.shape)

        out._backward = _backward
        return out 

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, backend=self.backend)
        
        t = self.data * other.data 
        out = Tensor(t, (self, other), _op='*', backend=self.backend) 

        def _backward(): 
            self.grad += self.unbroadcast(
                out.grad * other.data,
                self.data.shape
            )
            
            other.grad += self.unbroadcast(
                out.grad * self.data, 
                other.data.shape
            )

        out._backward = _backward
        return out 

    def sum(self, axis=None, keepdims=False): 
        if axis is None: 
            axes = None 
        elif isinstance(axis, tuple): 
            axes = axis 
        else: 
            axes = (axis,)
        
        t = self.backend.sum(self.data, axis=axes, keepdims=keepdims)
        out = Tensor(t, (self,), 'sum', backend=self.backend)

        def _backward():
            grad = out.grad 
            if axis is not None and not keepdims:
                for ax in sorted(axes): 
                    grad = self.backend.expand_dims(grad, axis=ax) # this just adds 1 to the missing dimensions 
            self.grad += self.backend.broadcast_to(grad , self.data.shape) # this scales the 1 to correct dimension
            return  

        out._backward = _backward
        return out  

    def mean(self, axis=None, keepdims=False):
        if axis is None:
            num = self.data.size
        elif isinstance(axis, tuple):
            num = self.backend.prod([self.data.shape[ax] for ax in axis])
        else:
            num = self.data.shape[axis]
    
        return self.sum(axis=axis, keepdims=keepdims) / num

    def exp(self): 
        t = self.backend.exp(self.data) 

        out = Tensor(t , (self, ), 'exp', backend=self.backend)

        def _backward():
            self.grad += t * out.grad
            return  

        out._backward = _backward
        return out 

    def relu(self): 
        t = self.backend.maximum(0, self.data)
        
        out = Tensor(t , (self,) , 'ReLU', backend=self.backend)

        def _backward():
            self.grad += (self.data > 0) * out.grad 
            return  

        out._backward = _backward
        return out 

    def tanh(self): 
        t = self.backend.tanh(self.data)
        out = Tensor(t , (self, ), 'tanh', backend=self.backend)

        def _backward(): 
            self.grad += (1 - t**2) * out.grad 

        out._backward = _backward
        return out 

    def log(self): 
        t = self.backend.log(self.data + 1e-12) 

        out = Tensor(t , (self,), 'log', backend=self.backend)

        def _backward(): 
            self.grad += (1.0/(self.data + 1e-12)) * out.grad 
            return 

        out._backward = _backward 
        return out 

    def sigmoid(self): 
        a = self.backend.exp(-self.data)
        t = 1 / (1.0 + a)

        out = Tensor(t, (self,), 'sigmoid', backend=self.backend)
        
        def _backward():
            self.grad += out.data * (1.0 - out.data) * out.grad
            return 
            
        out._backward = _backward
        return out 

    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, backend=self.backend)
    
        A = self.data
        B = other.data
    
        assert A.ndim >= 2
        assert B.ndim >= 2
    
        # Matrix dimensions must align
        assert A.shape[-1] == B.shape[-2]
    
        t = self.backend.matmul(A, B)
    
        out = Tensor(t, (self, other), "matmul", backend=self.backend)
    
        def _backward():
    
            dA = self.backend.matmul(
                out.grad,
                self.backend.swapaxes(B, -1, -2)
            )
    
            dB = self.backend.matmul(
                self.backend.swapaxes(A, -1, -2),
                out.grad
            )
    
            self.grad += self.unbroadcast(
                dA,
                self.data.shape
            )
    
            other.grad += self.unbroadcast(
                dB,
                other.data.shape
            )
    
        out._backward = _backward
    
        return out        

    def __neg__(self): 
        return self * -1

    def __sub__(self, other): 
        other = other if isinstance(other, Tensor) else Tensor(other, backend=self.backend)

        return self + -other 

    def __pow__(self, other): 
        assert isinstance(other , (int, float))

        t = (self.data) ** other 
        out = Tensor(t , (self,), f'pow{other}', backend=self.backend)

        def _backward():
            self.grad += self.unbroadcast(other * (self.data ** (other - 1)) * out.grad , self.data.shape) 
            
        out._backward = _backward
        return out 

    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, backend=self.backend)
        return self * (other ** -1)

    def __radd__(self, other): 
        return self + other 

    def __rsub__(self, other): 
        return -self + other 

    def __rtruediv__(self, other): 
        return other * (self ** -1)

    def __rmul__(self, other):
        return self * other

    def __len__(self):
        return len(self.data)
        
    def reshape(self, new_shape): 
        out = Tensor(self.backend.reshape(self.data, new_shape), (self,), 'reshape', backend=self.backend)

        def _backward(): 
            self.grad += out.grad.reshape(self.data.shape)
            return
        out._backward = _backward
        return out 

    def transpose(self):
        out = Tensor(self.data.T, (self,), 'transpose', backend=self.backend)

        def _backward():
            self.grad += out.grad.T 
            return 
        out._backward = _backward
        return out 

    def sqrt(self): 
        t = self.backend.sqrt(self.data + 1e-12)
        out = Tensor(t, (self,), 'sqrt', backend=self.backend)

        def _backward():
            self.grad += (1/(2 * t)) * out.grad
        out._backward = _backward
        return out 

    def abs(self): 
        t = self.backend.abs(self.data) 
        out = Tensor(t, (self,), 'abs', backend=self.backend)

        def _backward(): 
            self.grad += self.backend.sign(self.data) * out.grad 
            return 
        out._backward = _backward
        return out 

    def clip(self, a , b): 
        t = self.backend.clip(self.data , a , b)
        out = Tensor(t, (self,), 'clip', backend=self.backend)

        def _backward(): 
            mask = (self.data >= a) & (self.data <= b)
            self.grad += mask * out.grad 
            
        out._backward = _backward
        return out 
    
    def __getitem__(self, index): 
        out = Tensor(self.data[index], (self,), "getitem", backend=self.backend)

        def _backward():
            grad = self.backend.zeros_like(self.data)
            self.backend.add_at(
                grad, 
                index, 
                out.grad 
            ) 
            self.grad += grad 

        out._backward = _backward
        return out 

    def max(self, axis=None, keepdims=False):
        if axis is None:
            axes = None
        elif isinstance(axis, tuple):
            axes = axis
        else:
            axes = (axis,)
    
        t = self.backend.max(self.data, axis=axes, keepdims=keepdims)
        out = Tensor(t, (self,), "max", backend=self.backend)
    
        def _backward():
            grad = out.grad
            max_values = t
    
            if axis is not None and not keepdims:
                for ax in sorted(axes):
                    grad = self.backend.expand_dims(grad, axis=ax)
                    max_values = self.backend.expand_dims(max_values, axis=ax)
    
            mask = (self.data == max_values)
    
            count = mask.sum(axis=axis, keepdims=keepdims)
    
            if axis is not None and not keepdims:
                for ax in sorted(axes):
                    count = self.backend.expand_dims(count, axis=ax)
    
            grad = self.backend.broadcast_to(grad, self.data.shape)
            count = self.backend.broadcast_to(count, self.data.shape)
    
            self.grad += mask * grad / count
    
        out._backward = _backward
        return out

    def softmax(self, axis=-1, keepdims=True):
        shifted = self - self.max(axis=axis, keepdims=True)
        exp_x = shifted.exp() 
        return exp_x / exp_x.sum(axis=axis, keepdims=True)
        
    def squeeze(self, axis=None):
        out = Tensor(
            self.backend.squeeze(self.data, axis=axis), 
            (self,), 
            "squeeze", 
            backend=self.backend 
        )

        def _backward():
            self.grad += out.grad.reshape(self.data.shape)
        out._backward = _backward 

        return out 

    def unsqueeze(self, axis): 
        out = Tensor(
            self.backend.expand_dims(self.data, axis=axis),
            (self,), 
            "unsqueeze", 
            backend=self.backend 
        )

        def _backward(): 
            self.grad += self.backend.squeeze(out.grad, axis=axis) 
        out._backward = _backward

        return out 

    def logsumexp(self, axis=-1, keepdims=False): 
        max_x = self.max(axis=axis, keepdims=True)

        shifted = self - max_x 

        out = shifted.exp().sum(
            axis=axis, 
            keepdims=True 
        ).log() + max_x 

        if not keepdims: 
            out = out.squeeze(axis) 

        return out 

    def log_softmax(self, axis=-1): 
        
        return self - self.logsumexp(
            axis=axis, 
            keepdims=True 
        ) 
    
    @property
    def T(self):
        return self.transpose()

    @property
    def shape(self): 
        return self.data.shape 

    @property
    def ndim(self):
        return self.data.ndim 
    
    @property
    def device(self):
        return self.backend.device

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = self.backend.ones_like(self.data)
        for v in reversed(topo):
            v._backward()