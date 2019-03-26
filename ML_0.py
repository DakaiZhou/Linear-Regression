import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



def Polynomial_Regression_One_Feature(x, y, theta, *args, **kwargs):
# example: h, theta = Polynomial_Regression_One_Feature(x, y, theta, alpha, err, method = "BGD")
#          h, theta = Polynomial_Regression_One_Feature(x, y, theta, tau, i, method = "LWR")
# Input:
# x,y             trainning data(vectors)
# theta           initial theta
# alpha           learning rate(not required for close form method)
# err             error for stopping condition(not required for close form method)
# tau             bandwith parameter for weights(required only in locally weighted linear regression)
# i               indx of the point with highest weight(required only in locally weighted linear regression)
# method          BGD - Batch gradient descent, SGD - Stochastic gradient descent, CF - Close Form, LWR - Locally weighted linear regression
# Output:
# h               values of the hypothesis
# theta           parameters for the regression
#    
    
    if len(args) == 0:
        theta = ClosedForm(x, y, theta)
        print("CF")
        print(theta)
    elif len(args) >2:
        raise Exception("Too many inputs.")
    elif len(args) == 2:
        if kwargs["method"] == "LWR":
            tau = args[0]
            i = args[1]
            theta = LocallyWeightedLinearRegression(x, y, theta, tau, i)
            print("LWR")
            print(theta)
        else:
            alpha = args[0]
            err = args[1]
            if kwargs["method"] == "BGD":
                theta = BatchGradientDescent(x, y, theta, alpha, err)
                print("BGD")
                print(theta)
            elif kwargs["method"] == "SGD":
                theta = StochasticGradientDescent(x, y, theta, alpha, err)
                print("SGD")
                print(theta)
            elif kwargs["method"] == "CF":
                theta = ClosedForm(x, y, theta)
                print("CF")
                print(theta)
    else:
        raise Exception("Invalid inputs.")

    # calculate hypothesis
    n = theta.size
    X = np.tile(x,(1,n)) ** (np.arange(n))
    h = np.dot(X, theta) # hypothesis
    return h, theta

##########################
# Batch gradient descent #
##########################
def BatchGradientDescent(x, y, theta, alpha, err):
    tmp = 0
    ls = 0
    pls = 0
    n = len(theta) # number of freedom
    m = len(x)
    X = np.tile(x,(1,n)) ** (np.arange(n)) # matrix contains input data, the design matrix
    pj = 0
    h = np.dot(X, theta) # hypothesis

    while(1):
        tmp = (sum((h - y) * (x ** np.arange(n)))).reshape([n ,1]) * alpha / m
        theta = theta-tmp # update theta
        h = np.dot(X, theta)
        j = (sum((h - y)**2) / (2*m)).reshape([1,1]) # mean least square
        if abs(j - pj) <= err: # stopping condition
            return theta
        else:
            pj = j
            
###############################
# Stochastic gradient descent #
###############################
def StochasticGradientDescent(x, y, theta, alpha, err):
    n = len(theta) # number of freedom
    m = len(x)
    tmp = 0
    pj = 0
    X = np.tile(x,(1,n)) ** (np.arange(n)) # matrix contains input data
    while(1):
        for i in range(m):
            h = np.dot(X[i, :], theta)
            j = ((h - y[i])**2 / 2).reshape([1,1]) # mean least square
            if abs(j - pj) <= err: # stopping condition
                return theta
            else:
                tmp = ((h - y[i]) * (x[i] ** np.arange(n))).reshape([n ,1]) * alpha
                theta = theta - tmp

##################################################
# using normal equation theta = (X^T*X)^-1*X^T*y #
##################################################
def ClosedForm(x, y, theta):
    n = len(theta)
    X = np.tile(x,(1,n)) ** (np.arange(n))
    theta = np.dot(np.dot(np.linalg.inv(np.dot(X.transpose(), X)), X.transpose()), y)
    h = np.dot(X, theta)
    return theta

####################################################
# Locally weighted linear regression in close form #
####################################################
def LocallyWeightedLinearRegression(x, y, theta, tau, i):
    n = len(theta)
    X = np.tile(x,(1,n)) ** (np.arange(n))
    w = np.exp(np.sum(-(x[i, :] - x)**2, axis = 1).reshape([len(x),1]) / (2*tau**2)) # weights in matrix form
    w_sqrt = np.sqrt(w) 
    theta = np.dot(np.dot(np.linalg.inv(np.dot((w_sqrt*X).transpose(), w_sqrt*X)), (w_sqrt*X).transpose()), w_sqrt*y)
    h = np.dot(X, theta)
    plt.figure(1)
    plt.plot(range(len(x)),w)
    plt.title("Plot of weights for each trainning data")
    plt.xlabel("Index of trainning data")
    plt.ylabel("Weight")
    #plt.show()
    return theta

    



#################################################################



# plot the original data
fname = 'ml_01_data.csv'
data = pd.read_csv(fname, header = None)
x = data.iloc[:,0] # the column of year
y = data.iloc[:,2] # the column of number of visitor
num = len(y)
x = x[1:num].values
y = y[1:num].values
x = x.astype(int).reshape([num-1,1])
y = y.astype(int).reshape([num-1,1])

# value normalization, ortherwise, overflow
mx = x.mean()
sigmax = x.std()
xx = (x - mx) / sigmax
my = y.mean()
sigmay = y.std()
yy = (y - my) / sigmay

ini_theta = np.array([[0.2], [0.2], [0.2], [0.2]]) # the number of element in ini_element stands for the degree of the polynomial

# polynomial regression
# h, theta = Polynomial_Regression_One_Feature(xx, yy, ini_theta, 0.5, 21, method = "LWR")
h, theta = Polynomial_Regression_One_Feature(xx, yy, ini_theta, 0.02, 10**-7, method = "SGD")
H = h * sigmay + my # convert to unnormalized data

# prediction
px = np.arange(2017, 2026).reshape(9,1)
pxx = (px - mx) / sigmax
pX = np.tile(pxx,(1,len(ini_theta))) ** (np.arange(len(ini_theta)))
ph = np.dot(pX, theta)
pH = ph * sigmay + my

plt.figure(2)
plt.plot(x, H, px, pH)
plt.scatter(x, y)
plt.xlabel('Year')
plt.ylabel('Number of Vistors')
plt.title('Raw data, polynomial regression and prediction')
plt.legend(['Polynomial Rgression', 'Prediction'])

plt.show()
