from .optimizer import Optimizer 
import numpy as np 


class AdamW(Optimizer):

    def __init__(
        self, 
        parameters, 
        lr = 0.001, 
        beta1 = 0.9, 
        beta2 = 0.999,
        weight_decay = 0.01, 
        eps = 1e-8 
    ): 
        super().__init__(parameters)
        self.lr = lr 
        self.beta1 = beta1 
        self.beta2 = beta2 
        self.weight_decay = weight_decay
        self.eps = eps 
        self.t = 0  

        self.m = [
            np.zeros_like(p.data) 
            for p in self.parameters
        ]

        self.v = [
            np.zeros_like(p.data) 
            for p in self.parameters
        ]


    def step(self): 
        self.t += 1 
        
        for p, m, v in zip(self.parameters, self.m, self.v):

            # First Moment 
            m *= self.beta1
            m += (1.0 - self.beta1) * p.grad 

            # Second Moment 
            v *= self.beta2
            v += (1.0 - self.beta2) * (p.grad ** 2)

            # Bias correction 
            m_cap = m/(1.0 - self.beta1 ** self.t)
            v_cap = v/(1.0 - self.beta2 ** self.t) 

            update = m_cap/ (np.sqrt(v_cap) + self.eps)

            # Decoupled weight decay 
            update += self.weight_decay * p.data 

            # Parameter update 
            p.data -= self.lr * update