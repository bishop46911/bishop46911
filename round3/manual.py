#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 10:00:57 2024

@author: luowen
"""

import numpy as np

base = 7500
multipler_list = [[24, 70, 41, 21, 60], 
                  [47, 82, 87, 80, 35], 
                  [73, 89, 100, 90, 70], 
                  [77, 83, 85, 79, 55], 
                  [12, 27, 52, 15, 30]]

hunter_list = [[2, 4, 3, 2, 4],
               [3, 5, 5, 5, 3],
               [4, 5, 8, 7, 2],
               [5, 5, 5, 5, 4],
               [2, 3, 4, 2, 3]]

m = len(multipler_list[0]) * len(multipler_list)

A = [[0 for j in range(m)] for i in range(m)]

b = [0 for i in range(m)]

for i in range(len(multipler_list)):
    for j in range(len(multipler_list[i])):
        if i==0 and j==0:
            for k in range(m):
                A[0][k] = 1
            b[0] = 100
        else:
            A[i*len(multipler_list)+j][i*len(multipler_list)+j] = multipler_list[0][0]
            A[i*len(multipler_list)+j][0] = - multipler_list[i][j]
            b[i*len(multipler_list)+j] = multipler_list[i][j] * hunter_list[0][0] - multipler_list[0][0] * hunter_list[i][j] 

A = np.matrix(A)
b = np.matrix(b)
X = np.linalg.inv(A) * b.T

# print(X)

X = X.tolist()

x = [[X[i*len(multipler_list[0])+j][0] for j in range(len(multipler_list[0]))] for i in range(len(multipler_list))]
print(x)

average_list = [[multipler_list[i][j]/(max(x[i][j], 0)+hunter_list[i][j])*base for j in range(len(multipler_list[0]))] for i in range(len(multipler_list))]
print(average_list)