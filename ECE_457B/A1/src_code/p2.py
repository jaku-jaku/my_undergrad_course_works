# lib
import numpy as np
import matplotlib.pyplot as plt
import copy
from typing import List, Dict, Optional

train_x = np.array([[-1, 1], [1, -1], [1, 1], [-1,-1]])
train_y = np.array([-1, -1, 1, 1])

# Hard-coded weights [biasi, wi1, wi2]
w1 = [-1, 1, 1] 
w2 = [1, 1, 1]
w3 = [0, -1, 1]

f_hyper_plane = lambda x, w: (w[0] - w[1] * x)/w[2]

# plot
fig = plt.figure()
ax = plt.axes()
ax.grid(True)
MIN, MAX = -2, 2
ax.set_xlim(MIN, MAX) 
ax.set_ylim(MIN, MAX) 

CLR_LUT = {-1:'r', 1:'b'}
MRK_LUT = {-1:'o', 1:'*'}
CLS_LABEL = {-1:"Class = -1", 1:"Class = 1"}

train_x_1 = train_x[np.where(train_y == 1)][:]
train_x_2 = train_x[np.where(train_y == -1)][:]

ax.scatter(train_x_1[:, 0], train_x_1[:, 1], c=CLR_LUT[1], marker=MRK_LUT[1], label=CLS_LABEL[1])
ax.scatter(train_x_2[:, 0], train_x_2[:, 1], c=CLR_LUT[-1], marker=MRK_LUT[-1], label=CLS_LABEL[-1])

X = np.linspace(MIN, MAX, 10)
Y1 = f_hyper_plane(X, w1)
Y2 = f_hyper_plane(X, w2)

plt.plot(X, Y1, label="hyperplane w1")
plt.plot(X, Y2, label="hyperplane w2")

ax.fill_between(X, MIN, Y1, facecolor='blue', alpha=0.1,label=CLS_LABEL[1])
ax.fill_between(X, Y2, MAX, facecolor='blue', alpha=0.1)
ax.fill_between(X, Y1,  Y2, facecolor='red', alpha=0.1, label=CLS_LABEL[-1])
ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
ax.set_aspect(1)
plt.xticks([-1, 1], [-1, 1])
plt.yticks([-1, 1], [-1, 1])
fig.savefig("fig/madaline.png", bbox_inches = 'tight')

print("Please find result @ fig/madaline.png")

