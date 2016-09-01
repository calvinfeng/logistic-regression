import match_loader as matches
import pdb
from numpy import dot
from numpy import array
from numpy import meshgrid
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

def find_min_max(x, n):
    # n represents the number of features, do not scale x0
    feature_min_max = []
    for i in range(0, n + 1):
        if i == 0:
            feature_min_max.append({'min': 0, 'max': 1})
        else:
            feature_min_max.append({'min': None, 'max': None})
    for x_vector in x:
        for i in range(1, n + 1):
            if feature_min_max[i]['max'] == None or feature_min_max[i]['max'] < x_vector[i]:
                feature_min_max[i]['max'] = x_vector[i]
            if feature_min_max[i]['min'] == None or feature_min_max[i]['min'] > x_vector[i]:
                feature_min_max[i]['min'] = x_vector[i]
    return feature_min_max

def feature_scale(feature_min_max, x):
    scaled_x = []
    for x_vector in x:
        scaled_x_vector = []
        for i in range(0, len(x_vector)):
            scaled_x_vector.append((x_vector[i] - feature_min_max[i]['min'])/(feature_min_max[i]['max'] - feature_min_max[i]['min']))
        scaled_x.append(scaled_x_vector)
    return scaled_x

x, y = [], []
match_data = matches.get_parsed_data()
for data in match_data[0:200]:
    x.append(data['x'])
    y.append(data['y'])

feature_min_max = find_min_max(x[0:180], 3)
scaled_x = feature_scale(feature_min_max, x)

# print gradient_descent(x, y)
# theta = [-10.924254587810285, 16.680540953194516, 12.86498581225716, -3.018842162598431]
# for i in range(150, 200):
#      prediction = floor(sigmoid(theta, scaled_x[i])*10000)/100.0
#      print "Prediction: %s%%, Actual value: %s" % (prediction, y[i])
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as pyplot

xx, yy = meshgrid(range(20, 45), range(0, 6))
normal = array([9.263*(10**-3), 2.658 , -3.784*(10**-1)])
d = 9.294 * (10**-3)
z = ((-1*d) - (normal[0]*xx) - (normal[1]*yy))/normal[2]
# plt3d = pyplot.figure().gca(projection='3d')
# plt3d.plot_surface(xx, yy, z)

ax = pyplot.figure().add_subplot(111, projection='3d')
ax.plot_surface(xx, yy, z, alpha=0.1)
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
for i in range(0, len(x)):
    if y[i] == 1:
        ax.scatter(x[i][1], x[i][2], x[i][3], c = 'green')
    else:
        ax.scatter(x[i][1], x[i][2], x[i][3], c = 'red')

pyplot.show()
