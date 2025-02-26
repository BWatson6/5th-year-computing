# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 14:44:52 2025

@author: Ben
monty carlo
"""
import numpy as np
import matplotlib.pyplot as plt

def MontyCarloMethod(function, interval, sample_number):
    """
    takes in a function that its going to be evaluating at random intevals
    
    also need to then also take in the width of the intergral 
    going to be a list.
    
    also inputs the number of samples wanted
    
    outbuts the value of the intergral
    """
    # assume to begin that only a 1d1 integral
    difference = interval[1]-interval[0]
    random_x = np.random.uniform(interval[0], interval[1], size=sample_number)
    sampled_function = function(random_x)
    Intergral = difference/sample_number * np.sum(sampled_function)
    
    return Intergral




def RNG(interval, number_of_numbers):
    """
    going to take in the interval valuse and generate a bunch of sample 
    x values that can be inputed into the function.
    """
    
    
    x_samples = np.array([])
    return x_samples
#%%
# testing numpys random number generator
Size = 10000
random_numpy = np.random.random(size=Size)
fig0, (ax11, ax12) = plt.subplots(1, 2, figsize=(12, 6))
ax11.plot(random_numpy[0:int(Size/2)], random_numpy[int(Size/2):,], '.k')

# testing a home grown RNG (how not to do it apparently) (also very slow)
start_array = np.array([])
a_0 = 111
a = 16807
c = 0
m = 2**31-1
m = 2**31
for i in range(Size):    
    random_value = ((a*a_0+c)%m)
    start_array = np.append(start_array, random_value)
    a_0 = start_array[-1]
    
test_random = start_array/max(start_array) # normalisation
ax12.plot(test_random[0:int(Size/2)], test_random[int(Size/2):,], '.b')
#%%

def pi_func(x):
    return 4/(1+x**2)

# we have a monty carlo yay
intergral_range = [0,1] 
test_intergral = MontyCarloMethod(pi_func, intergral_range, 1000000)





