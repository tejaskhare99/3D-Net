import  h5py
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset

from utils import load_mnist3d, load_mnist


class Dataset2d_3d(Dataset):
    def __init__(self,istrain=True):
        if(istrain):
            x,y,_,_ = load_mnist3d('data/mnist3d/full_dataset_vectors.h5')
            mnist,_ = load_mnist('data/mnist/')
        else:
            _,_,x, y= load_mnist3d('data/mnist3d/full_dataset_vectors.h5')
            _,mnist = load_mnist('data/mnist/')

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item):
        x,y = self.x[item],self.y[item]
        x_2d = self.map_label(y)

        return x,y


    def map_label(self,index):
        return self.mnist[self.mnist['label'] == index].iloc[np.random.randint(0, len(self.mnist[self.mnist['label'] == index]))].iloc[1:-1]






if __name__=='__main__':
    Dataset2d_3d()
    















