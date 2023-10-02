# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required libraries.

2.Load the dataset.

3.Define X and Y array.

4.Define a function for costFunction,cost and gradient.

5.Define a function to plot the decision boundary. 6.Define a function to predict the Regression value.

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: Safeeq Fazil.A
RegisterNumber:  212222240086
*/
```
```
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

data=np.loadtxt("ex2data1.txt",delimiter=',')
X=data[:,[0,1]]
y=data[:,2]

X[:5]

y[:5]

plt.figure()
plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted")
plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not Admitted")
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.legend()
plt.show()

def sigmoid(z):
    return 1/(1+np.exp(-z))

plt.plot()
X_plot=np.linspace(-10,10,100)
plt.plot(X_plot,sigmoid(X_plot))
plt.show()

def costFunction (theta,X,y):
    h=sigmoid(np.dot(X,theta))
    J=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
    grad=np.dot(X.T,h-y)/X.shape[0]
    return J,grad

X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
J,grad=costFunction(theta,X_train,y)
print(J)
print(grad)

X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([-24,0.2,0.2])
J,grad=costFunction(theta,X_train,y)
print(J)
print(grad)

def cost (theta,X,y):
    h=sigmoid(np.dot(X,theta))
    J=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
    return J

def gradient (theta,X,y):
    h=sigmoid(np.dot(X,theta))
    grad=np.dot(X.T,h-y)/X.shape[0]
    return grad

X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
res=optimize.minimize(fun=cost,x0=theta,args=(X_train,y),method='Newton-CG',jac=gradient)
print(res.fun)
print(res.x)

def plotDecisionBoundary(theta,X,y):
    x_min,x_max=X[:,0].min()-1,X[:,0].max()+1
    y_min,y_max=X[:,1].min()-1,X[:,1].max()+1
    xx,yy=np.meshgrid(np.arange(x_min,x_max,0.1),np.arange(y_min,y_max,0.1))
    X_plot=np.c_[xx.ravel(),yy.ravel()]
    X_plot=np.hstack((np.ones((X_plot.shape[0],1)),X_plot))
    y_plot=np.dot(X_plot,theta).reshape(xx.shape)
    
    plt.figure()
    plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted")
    plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not Admitted")
    plt.contour(xx,yy,y_plot,levels=[0])
    plt.xlabel("Exam 1 score")
    plt.ylabel("Exam 2 score")
    plt.legend()
    plt.show()


plotDecisionBoundary(res.x,X,y)

prob=sigmoid(np.dot(np.array([1,45,85]),res.x))
print(prob)

def predict(theta,X):
    X_train =np.hstack((np.ones((X.shape[0],1)),X))
    prob=sigmoid(np.dot(X_train,theta))
    return (prob>=0.5).astype(int)
np.mean(predict(res.x,X)==y)

```

## Output:

![image](https://github.com/Safeeq-Fazil/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118680361/27e26ef4-4fc5-4e13-8213-6429abc26d68)

![image](https://github.com/Safeeq-Fazil/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118680361/96219c5e-916a-45fa-88cf-1f53aa7f3d2b)

![image](https://github.com/Safeeq-Fazil/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118680361/37b84c10-f2ee-4171-8100-41cc29eca2a8)

![image](https://github.com/Safeeq-Fazil/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118680361/3def0acd-cc05-416a-9e31-c18f5889d711)

![image](https://github.com/Safeeq-Fazil/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118680361/5921c606-7031-4f89-b827-c25e418b7753)

![image](https://github.com/Safeeq-Fazil/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118680361/12d24498-013e-46ac-b6ff-a47c0769e547)

![image](https://github.com/Safeeq-Fazil/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118680361/3ca71148-3043-490d-b21a-5f0802b3ad49)

![image](https://github.com/Safeeq-Fazil/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118680361/24575073-374e-44b4-99e7-3e52d4510531)

![image](https://github.com/Safeeq-Fazil/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118680361/7c87580a-1725-4d2a-a603-ea53160a27a3)

![image](https://github.com/Safeeq-Fazil/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118680361/1b780564-39fc-4520-aaef-259f28dd80b1)

## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

