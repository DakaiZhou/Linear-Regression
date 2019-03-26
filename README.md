# Linear-Regression
This linear Regression is specificly for polynomial regression with one feature. It contains Batch gradient descent, Stochastic gradient descent, Close Form and Locally weighted linear regression.

To run this code, you only need to change one line
For example:
h, theta = Polynomial_Regression_One_Feature(xx, yy, ini_theta, 0.5, 21, method = "LWR"), or
h, theta = Polynomial_Regression_One_Feature(xx, yy, ini_theta, 0.02, 10**-7, method = "SGD")


