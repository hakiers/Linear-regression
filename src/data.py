import numpy as np
import pandas as pd

class DataLoader:
    def __init__(self, path):
        self.path = path
        self.data = None

    def open_file(self):
        # Load data from the file
        df = pd.read_csv(self.path, delimiter='\t', decimal=',')
        self.data = df.values
    
    def split(self, train=0.6, valid=0.2, test=0.2):
        
        # Randomly shuffle the data
        np.random.shuffle(self.data)
        
        n = len(self.data)
        train_end = int(train * n)
        valid_end = int((train + valid) * n)
        
        train_data = self.data[:train_end]
        valid_data = self.data[train_end:valid_end]
        test_data = self.data[valid_end:]
        
        return train_data, valid_data, test_data