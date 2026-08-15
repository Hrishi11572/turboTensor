from .optimizer import Optimizer 

class SGD(Optimizer):  

    def __init__(self, parameters, lr=0.01): 
        super().__init__(parameters)
        self.lr = lr 

    def step(self): 
        for p in self.parameters: 
            p.data -= self.lr * p.grad 