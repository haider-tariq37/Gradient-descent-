# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 15:26:53 2020

@author: Haider
"""

import numpy as np
import matplotlib.pyplot as plt

"pre-processing"
data = np.genfromtxt('ex1data2.txt', dtype=None, delimiter=",")
x=data[:,0:2]
y=data[:,2]

x1 = np.ones(shape=[47, 1], dtype="int")
x1= np.c_[ x1, x] 
weights=np.zeros(shape=[3,1])

"normal equation implementation"
Xtrans=np.transpose(x1)
t1= np.linalg.inv(np.dot(Xtrans,x1))
t2=np.dot(t1,Xtrans)
weights=np.dot(t2,y)

"price prediction"
testset=np.array([1650,3])
priceprediction1=weights[0] + weights[1]*testset[0]+weights[2]*testset[1]
