from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from collections import Counter
import warnings

style.use('fivethirtyeight')

dataset = {'k':[[1,2],[2,3],[3,1]], 'r':[[6,5],[7,7],[8,6]] }
new_features = [5,7]

# [[plt.scatter(j[0], j[1], s=100, color=i) for j in dataset[i]] for i in dataset]
#
# plt.show()


def k_nn(data, predict, k=3):
    if len(data) >= k:
        warnings.warn('K is set to a value less than total voting groups!')

    dist = []
    for row in data:
        s = 0
        for i in range(len(row)):
            s += (row[i]+predict[i])**2
        dist.append(sqrt(s))

    return sorted(dist)

vls = []
for v in dataset.values():
    vls += v

print(k_nn(vls, new_features))

# class KNN:
#     def __init__(self):


# euclidean_distance = sqrt((plot1[0]-plot2[0])**2 + (plot1[1]-plot2[1])**2)
#
# print(euclidean_distance)
