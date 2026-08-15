from .module import Module 
import numpy as np 


class MSELoss(Module): 

    def __call__(self, predictions, target): 
        return ((predictions - target)**2).mean()
    
    
class MAELoss(Module): 

    def __call__(self, predictions, target): 
        return (predictions - target).abs().mean() 
    

class BCELoss(Module): 

    def __call__(self, predictions, target): 
        eps = 1e-12 

        predictions = predictions.clip(eps, 1.0 - eps) 

        return - (
            target * (predictions.log())
            + 
            (1 - target) * (1 - predictions).log()
        ).mean()
        

class CrossEntropyLoss(Module): 

    def __call__(self, logits, target): 
        log_probs = logits.log_softmax(axis=-1)

        batch_indices = np.arange(len(target)) 

        correct_log_probs = log_probs[
            batch_indices,
            target
        ]

        return -correct_log_probs.mean()