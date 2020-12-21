import plotly.graph_objs as go
from os.path import join
import pandas as pd
import matplotlib.pyplot as plt
import h5py
import numpy as np

def load_mnist(path):
    train_path = join(path,'mnist_train.csv')
    test_path = join(path,'mnist_test.csv')
    trainmnist = pd.read_csv(train_path)
    testmnist = pd.read_csv(test_path)

    return trainmnist,testmnist

def plot3d(number):
    import plotly.express as px
    print(number.shape)
    if(number.shape== (16,16,16)):
        number = rasterise(number)


    else:
        number = number[0].reshape(16,16,16)
        number = rasterise(number)

    fig = px.scatter_3d(number, x='x', y='y', z='z')
    fig.show()

def rasterise(number):
    list = pd.DataFrame(columns=['x','y','z'])
    for i in range(16):
        for j in range(16):
            for k in range(16):
                if(number[i,j,k]>0.0):
                   list = list.append({'x':i,'y':j,'z':k},ignore_index=True)
    return list




def load_mnist3d(path):
    with h5py.File(path, "r") as hf:
        # Split the data into training/test features/targets
        X_train = hf["X_train"][:]
        targets_train = hf["y_train"][:]
        X_test = hf["X_test"][:]
        targets_test = hf["y_test"][:]

        # Determine sample shape


        # Reshape data into 3D format
        X_train = rgb_data_transform(X_train)
        X_test = rgb_data_transform(X_test)

        return X_train,targets_train,X_test,targets_test

        # Convert target vectors to categorical targets
        # targets_train = to_categorical(targets_train).astype(np.integer)
        # targets_test = to_categorical(targets_test).astype(np.integer)

def array_to_color(array, cmap="Oranges"):
    s_m = plt.cm.ScalarMappable(cmap=cmap)
    return s_m.to_rgba(array)[:,:-1]


def rgb_data_transform(data):
    data_t = []
    for i in range(data.shape[0]):
        data_t.append(data[i].reshape(16, 16, 16))
    return np.asarray(data_t, dtype=np.float32)
