import numpy as np 


class DataLoader:

    def __init__(self, X, y, batch_size, shuffle=False): 
        self.X = X 
        self.y = y 
        self.batch_size = batch_size
        self.shuffle = shuffle 


    def __iter__(self): 
        n = len(self.X)

        indices = np.arange(n)

        if self.shuffle: 
            np.random.shuffle(indices)


        for start in range(0, n, self.batch_size): 
            batch_indices = indices[start: start+self.batch_size]

            yield(
                self.X[batch_indices], 
                self.y[batch_indices]
            )