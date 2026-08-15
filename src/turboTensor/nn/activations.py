from .module import Module

class ReLU(Module): 

    def __call__(self, x):
        return x.relu()
    
    
class Sigmoid(Module): 

    def __call__(self, x): 
        return x.sigmoid()
    
    
    
class Tanh(Module): 

    def __call__(self, x): 
        return x.tanh()