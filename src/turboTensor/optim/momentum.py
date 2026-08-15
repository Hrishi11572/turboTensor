from .optimizer import Optimizer 
import numpy as np 


class Momentum(Optimizer):

    def __init__(
        self, 
        parameters, 
        lr=0.01, 
        momentum=0.9
    ): 

        super().__init__(parameters)

        self.lr = lr 
        self.momentum = momentum 

        self.velocity = [
            np.zeros_like(p.data)
            for p in self.parameters
        ]

    def step(self): 
        for p, v in zip(self.parameters, self.velocity): 

            v *= self.momentum 
            v += p.grad 

            p.data -= self.lr * v 