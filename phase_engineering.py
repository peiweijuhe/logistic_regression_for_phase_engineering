# -*- coding: utf-8 -*-
"""
Spyder Editor

"""
import xlrd 
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from reg_utils import sigmoid, relu
from reg_utils import compute_cost, predict, update_parameters,forward_propagation

data = xlrd.open_workbook('phase_engineering.xlsx')

table_H = data.sheets()[0]
data_H = [[0 for col in range(4)] for row in range(4)]
for i in range(4):
    for j in range(4):
        data_H[i][j] = table_H.row_values(i)[j]

table_T = data.sheets()[1]
data_T = [[0 for col in range(4)] for row in range(4)]
for i in range(4):
    for j in range(4):
        data_T[i][j] = table_T.row_values(i)[j]

array_H = np.array(data_H)
array_T = np.array(data_T)

x = np.arange(-10,20.1,10)
y = np.arange(20,-10.1,-10)

xx = np.arange(-10,20.1,2)
yy = np.arange(-10,20.1,2)
X,Y = np.meshgrid(xx,yy)
strain_a = np.reshape(X,(-1,1))
strain_b = np.reshape(Y,(-1,1))
train_X = np.append(strain_a,strain_b,axis=1).T

f_H = interpolate.interp2d(x,y,array_H,kind='linear')
z_H = f_H(xx,yy)

f_T = interpolate.interp2d(x,y,array_T,kind='linear')
z_T = f_T(xx,yy)

phase = (z_H < z_T)
train_Y = np.reshape(phase, (-1,1)).T
#print(train_X.T)
#print(train_Y.T)





#print(train_X.shape)
#print(phase)

def model(X,Y,learning_rate = 0.05, num_iterations = 100000, print_cost = True, lambd = 0):
    grads = {}
    costs = []
    m = X.shape[1]
    layers_dims = [X.shape[0],50,20,1]
    #print(layers_dims)
    parameters = initialize_parameters(layers_dims)
    #print(parameters)
    for i in range(0, num_iterations):
        a3, cache = forward_propagation(X, parameters)
        cost = compute_cost_with_regularization(a3, Y, parameters, lambd)
        grads = backward_propagation_with_regularization(X, Y, cache, lambd)
        parameters = update_parameters(parameters, grads, learning_rate)
        if print_cost and i % 1000 == 0:
            print("Cost after iteration {}: {}".format(i, cost))
        if print_cost and i % 10 == 0:
            costs.append(cost)
            
    # plot the cost
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (x1,000)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters
    
def initialize_parameters(layers_dims):
        #np.random.seed(3)
    parameters = {}
    L = len(layers_dims) - 1 # integer representing the number of layers
    for l in range(1, L + 1):
        parameters['W' + str(l)] = np.random.randn(layers_dims[l],layers_dims[l-1])/ np.sqrt(layers_dims[l-1])
        parameters['b' + str(l)] = np.zeros((layers_dims[l],1))
        print l
    return parameters

def compute_cost_with_regularization(A3, Y, parameters, lambd):
    m = Y.shape[1]
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    W3 = parameters["W3"]
    cross_entropy_cost = compute_cost(A3, Y) 

    L2_regularization_cost = lambd*(np.sum(np.square(W1))+np.sum(np.square(W2))+np.sum(np.square(W3)))/(2*m)

    cost = cross_entropy_cost + L2_regularization_cost
    
    return cost

def backward_propagation_with_regularization(X, Y, cache, lambd):
    m = X.shape[1]
    (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3) = cache
    
    dZ3 = A3 - Y

    dW3 = 1./m * np.dot(dZ3, A2.T) + (lambd*W3)/m

    db3 = 1./m * np.sum(dZ3, axis=1, keepdims = True)
    
    dA2 = np.dot(W3.T, dZ3)
    dZ2 = np.multiply(dA2, np.int64(A2 > 0))

    dW2 = 1./m * np.dot(dZ2, A1.T) + (lambd*W2)/m

    db2 = 1./m * np.sum(dZ2, axis=1, keepdims = True)
    
    dA1 = np.dot(W2.T, dZ2)
    dZ1 = np.multiply(dA1, np.int64(A1 > 0))

    dW1 = 1./m * np.dot(dZ1, X.T) + (lambd*W1)/m
    db1 = 1./m * np.sum(dZ1, axis=1, keepdims = True)
    
    gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3,"dA2": dA2,
                 "dZ2": dZ2, "dW2": dW2, "db2": db2, "dA1": dA1, 
                 "dZ1": dZ1, "dW1": dW1, "db1": db1}
    
    return gradients

parameters = model(train_X, train_Y,lambd = 0.3)
print ("On the training set:")
predictions_train = predict(train_X, train_Y, parameters)


xx = np.arange(-10,20.1,0.05)
yy = np.arange(-10,20.1,0.05)
X,Y = np.meshgrid(xx,yy)
strain_a = np.reshape(X,(-1,1))
strain_b = np.reshape(Y,(-1,1))
test_X = np.append(strain_a,strain_b,axis=1).T
a3, cache = forward_propagation(test_X, parameters)
Z = (a3>0.5)
Z = Z.reshape(X.shape)
#plt.title("Intersection contour of 1T' and 1T ")
plt.ylabel('b(%)')
plt.xlabel('a(%)')
plt.contourf(X, Y, Z, cmap=plt.cm.Spectral)
plt.scatter(train_X[0, :], train_X[1, :], c=np.reshape(train_Y,(-1,)),cmap=plt.cm.Spectral)
plt.show()

#print(parameters['W1'])
