
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
import h5py    
import numpy as np  
import random


def plot3d(pt3d):
  data=[]
  if np.shape(pt3d) == (16,16,16):
    vals = pt3d
    for i in range(16):
      for j in range(16):
          for k in range(16):
              if vals[i,j,k]>0.05:
                data.append([i,j,k,vals[i,j,k]])

  else:
    ri = random.randint(0,len(pt3d))
    vals = pt3d[ri]
    for i in range(16):
      for j in range(16):
          for k in range(16):
              if vals[i,j,k]>0.05:
                data.append([i,j,k,vals[i,j,k]])

  

  df = pd.DataFrame(data,columns=["x","y","z","val"])


  fig = px.scatter_3d(df, x='x', y='y', z='z',color="val", )
  fig.show()