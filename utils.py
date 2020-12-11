import plotly.graph_objs as go
from os.path import join
import pandas as pd
import matplotlib.pyplot as plt
import h5py
import numpy as np
def plot3d(digit):
    x_c = [r[0] for r in digit[1]]
    y_c = [r[1] for r in digit[1]]
    z_c = [r[2] for r in digit[1]]
    trace1 = go.Scatter3d(x=x_c, y=y_c, z=z_c, mode='markers',
                          marker=dict(size=12, color=z_c, colorscale='Viridis', opacity=0.7))

    data = [trace1]
    layout = go.Layout(height=500, width=600, title="Digit: " + str(digit[0][2]) + " in 3D space")
    fig = go.Figure(data=data, layout=layout)
    fig.show()

def load_mnist(path):
    train_path = join(path,'mnist_train.csv')
    test_path = join(path,'mnist_test.csv')
    trainmnist = pd.read_csv(train_path)
    testmnist = pd.read_csv(test_path)

    return trainmnist,testmnist

def load_mnist3d(path):
    with h5py.File(path, "r") as hf:
        # Split the data into training/test features/targets
        X_train = hf["X_train"][:]
        targets_train = hf["y_train"][:]
        X_test = hf["X_test"][:]
        targets_test = hf["y_test"][:]

        # Determine sample shape
        sample_shape = (16, 16, 16, 3)

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
