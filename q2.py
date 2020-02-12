import numpy as np
import sklearn.preprocessing
import matplotlib.pyplot as plt

data = np.genfromtxt('ex1data2.txt', dtype=None, delimiter=",")
normData = np.genfromtxt('ex1data2.txt', dtype=None, delimiter=",")
#normData=sklearn.preprocessing.normalize(data)

"normalization"
meanX1=np.mean(data[:,0])
meanX2=np.mean(data[:,1])
stdX1=np.std(data[:,0])
stdX2=np.std(data[:,1])
meanY=np.mean(data[:,2])
stdY=np.std(data[:,2])

#x=normData[:,0:2]
x=np.zeros([47,2])
x[:,0]=data[:,0]
x[:,1]=data[:,1]
x[:,0]=(x[:,0]-meanX1)/stdX1
x[:,1]=(x[:,1]-meanX2)/stdX2
y=normData[:,2]
y=(y-meanY)/stdY

"test data normalization"
testset=np.zeros([2,1])
testset[0]=1650
testset[1]=3
#new=np.reshape(testset,newshape=[1,-1])
#normtestdata=sklearn.preprocessing.normalize(new)
testset[0]=(testset[0]-meanX1)/stdX1
testset[1]=(testset[1]-meanX2)/stdX2

"bias column"
x1 = np.ones(shape=[47, 1], dtype="int")
x1= np.c_[ x1, x] 
weights=np.zeros(shape=[3,1])
"gradient descent"
iterations = 800
alpha = 0.01
costarr =[]
for _ in range(iterations):
    predictions = np.dot(x1, weights)
    cost=np.transpose(predictions)-y
    "value of cost function"
    costvalue=(1/(2*47))*np.sum(cost**2) 
    gradient = np.dot(np.transpose(x1), np.transpose(cost))
    weights -= (alpha/47)*(gradient)
    costarr.append(costvalue)


predictionsf= weights[0]*x1[:,0] + weights[1]*x1[:,1]+weights[2]*x1[:,2]
#priceprediction=weights[0] + weights[1]*normtestdata[:,0]+weights[2]*normtestdata[:,1]
#priceprediciton=(priceprediction*stdY)+meanY

"price prediction form test data and denormalization"
priceprediction1=weights[0] + weights[1]*testset[0]+weights[2]*testset[1]
priceprediciton1=(priceprediction1*stdY)+meanY

"cost vs iterations plot"
plt.xlabel('iterations')
plt.ylabel('costs') 
plt.plot(list(range(iterations)), costarr, 'green', label='Fitted line')
plt.show()






