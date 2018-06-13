import random
import matplotlib.pyplot as plt
from matplotlib.patches import Circle 
import numpy as np
from sklearn import linear_model
import time
import matplotlib.patches as mpatches

# Problem 3.3
# For the double semi-circle task in Problem 3.1, set sep = -5 and generate 
# 2,000 examples.

# semi-circle variables
rad = 10 # radian length
thk=5 # thickness width
sep = -5 # amount of separation
n = 2000 # number of examples to generate
buff = 5 # additional edge space on graph

# pos and neg semi-circle centers
x_origin = 0
y_origin = 0
x_sec = x_origin + rad + (thk / 2)
y_sec = y_origin - sep

# max and min values for the x- and y- axes
x_min = x_origin - (rad + thk + buff)
x_max = x_origin + (rad + (thk / 2) + rad + thk + buff)
y_min = y_origin - (sep + rad + thk + buff)
y_max = y_origin + (rad + thk + buff)

# area for positive circle
circ_pos_inner = Circle((x_origin, y_origin), radius = rad)
circ_pos_outer = Circle((x_origin, y_origin), radius = rad + thk)

# area for negative circle
circ_neg_inner = Circle((x_sec, y_sec), radius = rad)
circ_neg_outer = Circle((x_sec, y_sec), radius = rad + thk)

# arrays to hold x and y point values
pts_pos_x = []
pts_pos_y = []
pts_neg_x = []
pts_neg_y = []

# check if n number of points have been generated within bounds
while not len(pts_pos_x) + len(pts_neg_x) == n:
    x = random.uniform(x_min + buff, x_max - buff) # randomly generate x value
    y = random.uniform(y_min + buff, y_max - buff) # randomly generate y value

    # check if point is in positive semi-circle, and add point to pos arrays
    if circ_pos_outer.contains_point([x,y]) and \
       not circ_pos_inner.contains_point([x,y]) and y > y_origin:
        pts_pos_x.append(x)
        pts_pos_y.append(y)
    # check if point is in negative semi-circle, and add point to neg arrays    
    elif circ_neg_outer.contains_point([x,y]) and \
        not circ_neg_inner.contains_point([x,y]) and y < y_sec:
        pts_neg_x.append(x)
        pts_neg_y.append(y)

def plot_points(x1, x2, y1, y2):
    """ Plots the generated points, sorted into the +1 and -1 classes.
    """
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    
    plt.scatter(x1, y1, c='red') # +1 points
    plt.scatter(x2, y2, c='blue') # -1 points 

def combine_data():
    """ Combines the dummy point (x[0]), x and y values, and group value into 
        an array.
    """
    d = np.zeros((2000, 4)).astype('float') # refit arrays into a matrix
    
    for i in range(0, len(pts_pos_x)): # append +1 datapoints
        d[i, 0] = 1
        d[i, 1] = pts_pos_x[i] 
        d[i, 2] = pts_pos_y[i]
        d[i, 3] = 1

    for j in range(len(pts_pos_x), n): # append -1 datapoints
        d[j, 0] = 1
        d[j, 1] = pts_neg_x[j - len(pts_pos_x)] 
        d[j, 2] = pts_neg_y[j - len(pts_pos_x)]
        d[j, 3] = -1
    
    return d    
    
def perceptron_calc(w, x):
    """ Calculates the cross product of the x-values and the corresponding 
        weights.
    """
    return w[0]*x[0] + w[1]*x[1]  + w[2]*x[2] 

def sign(x):
    """ Gives sign of value
    """
    return 1 if x>=0 else -1
       
def update_rule(w, d):
    """ Updates the weights according to the perceptron linear algorithm.
    """       
    w[0] += d[0] * d[3] # update dummy weight
    w[1] += d[1] * d[3] # update x value weight
    w[2] += d[2] * d[3] # update y value weight
    
    return w

def ein(w, d):
    """ Calculated the Ein value for the pocket algorithm
    """
    count = 0 # initialize count
    for dpt in d: # count number of incorrect predictions
        if sign(perceptron_calc(w, dpt)) != dpt[3]:
            count += 1

    return count / n # return count over total number of points
        
def pocket_algorithm(t):
    """ Performs the pocket algorithm on the data for t number of iterations.
    """
    dataset = combine_data() # rearrange the dataset
    weights = [0.0, 0.0, 0.0] # set starting weight values    
    
    e_t = [] # array to keep current best Ein value for each iteration
    best_ein = 1 # initialize to worst case Ein (100% error)
    
    # loop until array is full with t iterations of Ein
    while len(e_t) != t:
        # shuffle to access different points
        rand_pts = np.random.choice(range(n), n, replace=False)        
        for p in rand_pts:
            # check if sign of calculated classification is equal to actual
            if sign(perceptron_calc(weights, dataset[p, :])) != dataset[p, -1]:
                temp_weights = list(weights) # copy weights
                # update weights temporarily
                temp_weights = update_rule(temp_weights, dataset[p, :])
                # calcuate current Ein
                temp_ein = ein(temp_weights, dataset)
                # check if current Ein is better than the current best Ein
                if temp_ein < best_ein:
                    best_ein = temp_ein # current Ein becomes the new best Ein
                    # current weights becomes the new weights
                    weights = temp_weights 
                # add the Ein value to the array
                e_t.append(best_ein)
                break
        
    return e_t, weights, dataset

def plot_ein_t(t, y_val):
    """ Plots the Ein values for each iteration of the pocket algorithm.
    """
    x_val = np.arange(1, t + 1) # iterations plotted on the x-axis
    
    plt.xlim(0, t) # values for the x-axis
    plt.ylim(0, 0.5) # values for the y-axis
    
    plt.xlabel('Iteration Number, t') # x-axis label
    plt.ylabel('Error, Ein') # y-axis label
    
    plt.plot(x_val, y_val) # graph the Ein value

def calc_m_and_b(w):
    """ Calculates the slope and y-intercept, where the formulas were 
        calculated from a previous problem.
    """
    m = -w[1]/w[2] # calculate slope
    b = -w[0]/w[2] # calculate y-intercept
    
    return m, b
    
def plot_hx(weights, min_val, max_val):    
    """ Gets slope and y-intercept for the h(x) function.
    """
    m_h, b_h = calc_m_and_b(weights) # calculate the slope and y-intercept
    
    # plot h(x)
    plt.plot(np.arange(min_val, max_val), m_h*np.arange(min_val, max_val) + \
             b_h, color='green')  
        
def combine_lra_data():
    """ Refits the data into x and y matrices that can be used by the linear
        regression model.
    """
    d = combine_data() # refit the data into a matrix
    x = d[:, :-1] # split x-values
    y = d[:, -1] # split y-values
    
    return x, y

def linear_regression_algorithm():
    """ Runs the linear regression algorithm using sklearn's linear_model. Uses
        the data to predict where the y-values should occur, then checks if the
        values are as predicted. Counts the incorrect number of predictions and
        returns the value over the total number of points, as well as the
        calculated weight values
    """
    
    regr = linear_model.LinearRegression() # linear regression package

    X_train, y_train = combine_lra_data() # split x- and y-values
    
    regr.fit(X_train, y_train) # fit the model with the x- and y-values
    y_pred = regr.predict(X_train) # predict the +1 and -1 classes
    
    y_pred[y_pred > 0] = 1 # all positive numbers go into the +1 class
    y_pred[y_pred < 0] = -1 # all negative numbers go into the -1 class
    
    count = sum(y_pred != y_train) # count number of incorrect predictions
    
    return count / n, regr.coef_ # return Ein and weights of the LRA

def plot_lr(weights, min_val, max_val):
    """ Plots the line of the linear regression algorithm calculation
    """
    m, b = calc_m_and_b(weights) # get the slope and y-intercept of the LRA
    # plot the line calculated from the linear regression algorithm
    plt.plot(np.arange(min_val, max_val), m*np.arange(min_val, max_val) + b, \
             color='yellow')
    
def compare_algorithms(e_pa, e_lra, t_pa, t_lra):
    """ Print values for the Ein values and computation times for the pocket 
        and linear regression algorithms
    """
    
    print("Pocket Algorithm Ein: ", e_pa[-1])
    print("Linear Regression Algorithm Ein: ", e_lra)
    print("Pocket Algorithm Time: ", t_pa)
    print("Linear Regression Algorithm Time: ", t_lra)
    
# (a) What will happen if you run PLA on those examples?
# If I run PLA on those examples, PLA will not terminate because the learning 
# set is not linearly separable. Since the vectors are not linearly separable 
# learning will never reach a point where all vectors are classified properly.
 
# (b) Run the pocket algorithm for 100,000 iterations and plot Ein versus the
# iteration number t.
iterations = 100000

start_time_pa = time.time()
ein_t, final_weights, data = pocket_algorithm(iterations)
end_time_pa = time.time() - start_time_pa

plot_ein_t(iterations, ein_t)
plt.show()

# (c) Plot the data and the final hypothesis in part (b).
plot_points(pts_pos_x, pts_neg_x, pts_pos_y, pts_neg_y)
plot_hx(final_weights, x_min, x_max)
plt.show()

# (d) Use the linear regression algorithm to obtain the weights w, and compare 
# this result with the pocket algorithm in terms of computation time and 
# quality of the solution.
start_time_lra = time.time()
ein_lra, w = linear_regression_algorithm()
end_time_lra = time.time() - start_time_lra

print("Linear Regression Algorithm Weights: ", w)

compare_algorithms(ein_t, ein_lra, end_time_pa, end_time_lra)

plot_points(pts_pos_x, pts_neg_x, pts_pos_y, pts_neg_y)
plot_hx(final_weights, x_min, x_max)
plot_lr(w, x_min, x_max)
p = mpatches.Patch(color='green', label='Pocket Algorithm')
lr = mpatches.Patch(color='yellow', label='Linear Regression Algorithm')
plt.legend(handles=[p, lr])
plt.show()

# The pocket algorithm's Ein is better than the linear regression algorithm's,
# but the linear regression algorithm's computation time is much better than 
# the pocket algorithms