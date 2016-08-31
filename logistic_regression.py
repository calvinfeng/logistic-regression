import match_loader as matches
import pdb
from numpy import dot
from numpy.linalg import norm
from math import exp
from math import log
from math import floor

def sigmoid(theta, x_vector):
    return 1.0/(1 + exp(-1*dot(theta, x_vector)))

def cost_function(theta, x, y, l):
    m = len(y) # number of training data
    n = len(theta) # number of features + 1
    # l represents lambda, the regularization factor
    theta_sum = 0
    for i in range(1, n):
        theta_sum += theta[i]**2
    theta_sum *= l/(2*m)

    net_cost = 0
    for i in range(0, m):
        if sigmoid(theta, x[i]) == 1:
            pdb.set_trace()
        elif sigmoid(theta, x[i]) == 0:
            pdb.set_trace()
        net_cost += y[i]*log(sigmoid(theta, x[i])) + (1 - y[i])*log(1 - sigmoid(theta, x[i]))
    return theta_sum - (net_cost/m)

def dj_theta_0(theta, x, y):
    m = len(y)
    derivative = 0
    for i in range(0, m):
        derivative += (sigmoid(theta, x[i]) - y[i])*x[i][0]
    return float(derivative)/m

def dj_theta_j(theta, x, y, j, l):
    m = len(y)
    derivative = 0
    for i in range(0, m):
        derivative += (sigmoid(theta, x[i]) - y[i])*x[i][j]
    return float(derivative)/m + (l*theta[j]/m)

def gradient_descent(x, y):
    # specify learning rate and regularization factor
    a, l = 0.05, 0.005
    # initialize theta to be a zero vector
    theta = [0, 0, 0, 0]
    n = len(theta)
    while True:
        dj_theta = [dj_theta_0(theta, x, y)]
        for j in range(1, n):
            dj_theta.append(dj_theta_j(theta, x, y, j, l))

        for j in range(0, n):
            theta[j] = theta[j] - a*dj_theta[j]

        cost = cost_function(theta, x, y, l)
        print "Cost function value: %s, norm(dJ): %s" % (cost, norm(dj_theta))

        if norm(dj_theta) <= 0.001:
            break
    return theta

x, y = [], []
match_data = matches.get_parsed_data()
for data in match_data[0:200]:
    x.append(data['x'])
    y.append(data['y'])

features = {'x1': {'max': None, 'min': None}, 'x2': {'max': None, 'min':None}, 'x3': {'max': None, 'min': None}}
for x_vector in x[0:180]:
    if features['x1']['max'] == None or features['x1']['max'] < x_vector[1]:
        features['x1']['max'] = x_vector[1]
    if features['x2']['max'] == None or features['x2']['max'] < x_vector[2]:
        features['x2']['max'] = x_vector[2]
    if features['x3']['max'] == None or features['x3']['max'] < x_vector[3]:
        features['x3']['max'] = x_vector[3]

    if features['x1']['min'] == None or features['x1']['min'] > x_vector[1]:
        features['x1']['min'] = x_vector[1]
    if features['x2']['min'] == None or features['x2']['min'] > x_vector[2]:
        features['x2']['min'] = x_vector[2]
    if features['x3']['min'] == None or features['x3']['min'] > x_vector[3]:
        features['x3']['min'] = x_vector[3]

for x_vector in x:
    x_vector[1] = (x_vector[1] - features['x1']['min'])/(features['x1']['max'] - features['x1']['min'])
    x_vector[2] = (x_vector[2] - features['x2']['min'])/(features['x2']['max'] - features['x2']['min'])
    x_vector[3] = (x_vector[3] - features['x3']['min'])/(features['x3']['max'] - features['x3']['min'])

# print gradient_descent(x, y)
theta = [-10.924254587810285, 16.680540953194516, 12.86498581225716, -3.018842162598431]
for i in range(180, 200):
    prediction = floor(sigmoid(theta, x[i])*10000)/100.0
    print "Prediction: %s%%, Actual value: %s" % (prediction, y[i])
