#We won't exactly be implementing Gradient descent in ML pursuits but we're learning GD rn to understand the internals of ML
#Mean square erro is also called cost function

#MSE = 1/n sigma (yi - (mxi+b))^2
#We cannot take each and every line in a massive dataset (not a smart approach), hence we use grdient descent
#GD is an algorithm that finds best fit line for given training data set

# 2 approaches 1. fixed steps, 2. steps that follow the curvature
#approach 2 is better, to implement we have to calculate slope at each step
# this slope is nothing but derivative of b wrt cost function