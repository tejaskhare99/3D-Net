import  h5py
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset

from utils import load_mnist3d, load_mnist


class Dataset2d_3d(Dataset):
    def __init__(self,istrain=True):
        if(istrain):
            self.x,self.y,_,_ = load_mnist3d('data/mnist3d/full_dataset_vectors.h5')
            self.mnist,_ = load_mnist('data/mnist/')
        else:
            _,_,self.x, self.y= load_mnist3d('data/mnist3d/full_dataset_vectors.h5')
            _,self.mnist = load_mnist('data/mnist/')

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item):
        x,y = self.x[item],self.y[item]
        x_2d = self.map_label(y)
        x_2d = np.array(x_2d.values).reshape(1,28,28)


        return x_2d,x


    def map_label(self,index):
        return self.mnist[self.mnist['label'] == index].iloc[np.random.randint(0, len(self.mnist[self.mnist['label'] == index]))].iloc[0:-1]






if __name__=='__main__':
    Dataset2d_3d()
















