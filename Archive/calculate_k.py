import math
import itertools
import numpy as np
import pandas as pd

def calculate_sample_size(R, c):
    k_s = int(math.log(1-R, c)) + 1
    k_t = int(math.log(1-R)/(c-1)) +1
    diff = k_s-k_t
    return R, c, k_s, k_t, diff

#R = np.arange(0.7, .9999, 0.0005)
#c = np.arange(0.7, .9999, 0.0005)

R = np.arange(0.7, 0.99, 0.01)
c = np.arange(0.7, 0.99, 0.01)

argument_grid = itertools.product(R,c)
sample_size = [calculate_sample_size(*argument) for argument in argument_grid]

sample_size_df = pd.DataFrame(sample_size,columns=["R", "c", "k_s", "k_t", "diff"])
#sample_size_df.sort_values(by='diff', ascending=False)

#sample_size_df[sample_size_df["diff"] <-1]["diff"]


X, Y = np.meshgrid(R, c)
Z = np.array(sample_size_df["diff"]).reshape(30,30)
#Z = np.array(sample_size_df["diff"]).reshape(600,600)


import plotly.express as px
fig = px.scatter_3d(sample_size_df, x="R", y="c", z="diff")
fig.update_traces(marker=dict(size=3))
fig.show()

### boxplot

calculate_sample_size(0.83,0.9)


fig = px.scatter_3d(sample_size_df, x="R", y="c", z="k_s")
fig.update_traces(marker=dict(size=3))
fig.show()







R = np.arange(0.7, .9999, 0.0005)
c = np.arange(0.7, .9999, 0.0005)

argument_grid = itertools.product(R,c)
sample_size = [calculate_sample_size(*argument) for argument in argument_grid]

sample_size_df = pd.DataFrame(sample_size,columns=["R", "c", "k_s", "k_t", "diff"])

X, Y = np.meshgrid(R, c)
#Z = np.array(sample_size_df["diff"]).reshape(30,30)
Z = np.array(sample_size_df["diff"]).reshape(600,600)
import plotly.express as px
fig = px.scatter_3d(sample_size_df, x="R", y="c", z="diff")
fig.update_traces(marker=dict(size=.5))
fig.show()

C = np.array(sample_size_df["diff"])