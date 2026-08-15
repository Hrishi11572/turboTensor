from ..tensor import Tensor

class Parameter(Tensor):
    '''Marks the tensor as learnable'''
    
    def to(self, device):
        tensor = super().to(device)

        self.data = tensor.data
        self.grad = tensor.grad
        self.backend = tensor.backend

        return self