import numpy as np
import matplotlib.pyplot as plt

"preprocessing"
data = np.genfromtxt('ex1data1.txt', dtype=None, delimiter=",")
x=data[:,0]
y=data[:,1]
x1 = np.ones(shape=[97, 1], dtype="float64")
x1= np.c_[ x1, data[:,0] ] 
weights=np.zeros(shape=[2,1])

"data plot"
#plt.scatter(x, y, color = "r", marker = "x") 
#plt.xlabel('population in 10,000s')
#plt.ylabel('profits in 10,000s')
#plt.show

np.reshape(y,newshape=[97,1])
iterations = 1500
alpha = 0.01

"gradient descent"
for _ in range(iterations):
    predictions = np.dot(x1, weights)
    cost=np.transpose(predictions)-y
    costvalue=(1/(2*97))*np.sum(cost**2)
    gradient = np.dot(np.transpose(x1), np.transpose(cost))
    weights -= (alpha/97)*(gradient)
    


predictionsf= weights[0]*x1[:,0] + weights[1]*x1[:,1]

"plot with regression line"
plt.xlabel('population in 10,000s')
plt.ylabel('profits in 10,000s')
plt.xlim(4, 24)
plt.ylim(-5, 25)
plt.scatter(x, y, color = "r", marker = "x") 
plt.plot(x, predictionsf, 'blue', label='Fitted line')
plt.show()