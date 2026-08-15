from .module import Module
from .parameter import Parameter

import numpy as np


class Linear(Module):
    '''
    X : (batch, in_features)
    W : (in_features, out_features)
    b : (out_features,)
    Y : (batch, out_features)

    Y = XW + b 
    '''
    def __init__(self, in_features, out_features):         
        super().__init__() 
        std = np.sqrt(2.0/in_features)
        
        self.W = Parameter(
            np.random.randn(in_features, out_features) * std 
        )

        self.b = Parameter(
            np.zeros(out_features)
        )
        
        self.register_parameter("W", self.W)
        self.register_parameter("b", self.b)
    
    def __call__(self, x):
        return x @ self.W + self.b 
    
    
    
class Sequential(Module):
    
    def __init__(self, *modules):
        super().__init__() 
        self.modules = modules
        
        for i, module in enumerate(modules): 
            self.add_module(str(i), module)


    def __call__(self, x): 
        for module in self.modules: 
            x = module(x)
        return x 