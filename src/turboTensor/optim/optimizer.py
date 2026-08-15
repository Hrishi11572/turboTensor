class Optimizer: 

    def __init__(self, parameters): 
        self.parameters = list(parameters)

    def zero_grad(self): 
        for p in self.parameters: 
            p.grad.fill(0)

    def step(self): 
        raise NotImplementedError