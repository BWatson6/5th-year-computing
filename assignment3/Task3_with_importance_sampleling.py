# -*- coding: utf-8 -*-
"""
Created on Mon Mar  3 14:36:20 2025

@author: Ben
"""

import numpy as np
# import scipy as sp
from mpi4py import MPI

comm = MPI.COMM_WORLD
numb_of_ranks = comm.Get_size()


class paralelMonty():
    # in the initiation it basicaly does the monte carlo part of this
    def __init__(self, function, numb_of_samples, dimentions, *args, interval=[0]):
        self.dimentions = dimentions
        self.numb_of_samples = numb_of_samples
        self.interval = interval
        self.function = function
        
        if len(interval) == 1: # does the sample_function with no interval
            
            sampled_function = self.OverAllSpace(function, 
                                                 self.dimentions,
                                                 numb_of_samples, 
                                                 *args)
            
            self.difference = 2
            self.expected_rank = self.expectation(sampled_function)
            # 2^dimentions is because the interval for this is always going to be (1--1)
            self.intergral = self.expected_rank*2**dimentions


        
        else: # do the same thing but with the defined limits          
            # creating random values and reshaping to be coordinats space of the defined dimention
            random_inputs = np.random.uniform(interval[0], interval[1], 
                                              size=(numb_of_samples*dimentions))      
            random_inputs = np.resize(random_inputs, (numb_of_samples, dimentions))
            
            self.difference = interval[1]-interval[0]
          
            # inputing the random input values into the function getting evaluated
            # interval part only works if its the same in all dimensions
            sampled_function = function(random_inputs, *args)
            self.expected_rank = self.expectation(sampled_function)
            self.intergral = self.expected_rank*(interval[1]-interval[0])**self.dimentions
            
        self.varience_rank = self.varience(sampled_function)
        # print('a rank varience', self.varience_rank)


    def expectation(self, value):
        return np.mean(value)

    def varience(self, value): # think were not quite ther yet with this hmmm
        return np.var(value)/self.numb_of_samples

    def addition(self):
        
        expectaion_sum  = comm.reduce(self.expected_rank/numb_of_ranks, 
                                      op=MPI.SUM, 
                                      root=0)
        
        varience_sum  = comm.reduce(self.varience_rank/numb_of_ranks, 
                                      op=MPI.SUM, 
                                      root=0)
        intergral_sum = comm.reduce(self.intergral/numb_of_ranks, 
                                    op=MPI.SUM, 
                                    root=0)



        # rank 0 calcualates the statistics with what it recives from the other ranks
        if comm.Get_rank()==0: 
            print('-----------\nfunction mean value:', expectaion_sum)
            print('function varience value:', varience_sum)
            print('INTERGRAL value:', intergral_sum)
            Intergral_error = np.sqrt(varience_sum)*self.difference**self.dimentions
            print('\nintergral error', Intergral_error)
        
        
    def OverAllSpace(self, function, dimention, samples, *args):
        def duPart(t_val): # might not be necessary to be a function now
            return(1+t_val**2)/(1-t_val**2)**2
        
        # generating the random numbers for a given dimention
        t_values = np.random.uniform(-1, 1, size=(samples*dimention))
        t_values = np.resize(t_values, (samples, dimention))
        
        # calculating the values for a new function
        product_part = np.product(duPart(t_values), axis=1)
        new_coord = t_values/(1-t_values**2)
        new_function = function(new_coord, *args) * product_part
        return new_function



def RandomCoordinates(interval, dimensions, numb_of_samples):
    random_inputs = np.random.uniform(interval[0], interval[1], 
                                      size=(numb_of_samples*dimensions))      
    random_inputs = np.resize(random_inputs, (numb_of_samples, dimensions))
    return random_inputs
    
    
def FillFration(trial_coordinates):
    radius = 1
    point_dist_from_origin = np.sum(trial_coordinates**2, axis=1)
    points_in_circle_index = np.where(point_dist_from_origin<radius, 1, 0)
    return points_in_circle_index

def Task2Function(trial_coordinates, x_0, sigma):
    top_of_fraction = -np.sum((trial_coordinates-x_0)**2, axis=1)
    function = 1/(sigma*np.sqrt(2*np.pi))*np.exp(top_of_fraction/2*sigma**2)
    return function

def Task1ImportanceFunc(x, dimention):
    # probibility part
    distribution = 1/(1+np.exp(-6*x+dimention))
    normaling = 6/(np.log((np.exp(6*np.sqrt(dimention))+np.exp(dimention))))
    distribution = distribution*normaling
    
    #Task one functioned copied from above:
    radius = 1
    point_dist_from_origin = np.sum(x**2, axis=1)
    points_in_circle_index = np.where(point_dist_from_origin<radius, 1, 0)
    
    function = points_in_circle_index/distribution
    return function

def Task1RandomDistribution(x, dimention):
    distribution = np.log(np.exp(6*x)-np.exp(dimention))
    return distribution


    

dimention = 2
# samples is samples per rank
samples = 100000
interval = [-1, 1] # square/cube is centerd on the origin for simplicity 

#Task 1
test = paralelMonty(FillFration, samples, dimention, interval=interval)
a = test.addition()

# test = paralelMonty(FillFration, 3, dimention, interval=interval)
# a = test.addition()

# test = paralelMonty(FillFration, 4, dimention, interval=interval)
# a = test.addition()

# test = paralelMonty(FillFration, 5, dimention, interval=interval)
# a = test.addition()



#Task 2
sigma = 1.0
d = 2
x_0 = np.zeros(d)
test2 = paralelMonty(Task2Function, samples, d, x_0, sigma)
b = test2.addition()

MPI.Finalize()

