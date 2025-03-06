# -*- coding: utf-8 -*-
"""
Created on Mon Mar  3 14:36:20 2025

@author: Ben
"""

import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
numb_of_ranks = comm.Get_size()


class paralelMonty():
    # in the initiation it basicaly does the monte carlo part of this
    def __init__(self, function, numb_of_samples, dimentions, *args, interval=[0]):
        self.dimentions = dimentions
        self.numb_of_samples = numb_of_samples
        self.interval = interval
        
        if len(interval) == 1: # does the sample_function with no interval
            # 2^dimentions is because the interval for this is always going to be (1--1)
            sampled_function = self.OverAllSpace(function, 
                                                 self.dimentions,
                                                 numb_of_samples, 
                                                 *args) * 2**dimentions
            
            self.expected_rank = self.expectation(sampled_function)


        
        else: # do the same thing but with the defined limits          
            # creating random values and reshaping to be coordinats space of the defined dimention
            random_inputs = np.random.uniform(interval[0], interval[1], 
                                              size=(numb_of_samples*dimentions))      
            random_inputs = np.resize(random_inputs, (numb_of_samples, dimentions))
          
            # inputing the random input values into the function getting evaluated
            # interval part only works if its the same in all dimensions
            sampled_function = function(random_inputs, *args)*(interval[1]-interval[0])**self.dimentions 
            
        self.expected_rank = self.expectation(sampled_function)
        self.varience_rank = self.varience(sampled_function)
        



    def expectation(self, value):
        expectation = 1/self.numb_of_samples * np.sum(value)
        return expectation

    def varience(self, value):
        varience = self.expectation(value**2) - self.expectation(value)**2
        return varience
    



    def addition(self):
        
        expectaion_sum  = comm.reduce(self.expected_rank/numb_of_ranks, 
                                      op=MPI.SUM, 
                                      root=0)
        
        varience_sum  = comm.reduce(self.varience_rank/numb_of_ranks, 
                                      op=MPI.SUM, 
                                      root=0)


        
        # rank 0 calcualates the statistics with what it recives from the other ranks
        if comm.Get_rank()==0: 
            print('mean value:', expectaion_sum)
            print('varience value:', varience_sum)
        
        
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
    
    
def FillFration(trial_coordinates):
    radius = 1
    point_dist_from_origin = np.sum(trial_coordinates**2, axis=1)
    points_in_circle_index = np.where(point_dist_from_origin<radius, 1, 0)
    return points_in_circle_index

def Task2Function(trial_coordinates, x_0, sigma):
    top_of_fraction = -np.sum((trial_coordinates-x_0)**2, axis=1)
    function = 1/(sigma*np.sqrt(2*np.pi))*np.exp(top_of_fraction/2*sigma**2)
    return function


dimention = 1
# samples is samples per rank
samples = 10000000
interval = [-1, 1] # square/cube is centerd on the origin for simplicity 

#Task 1
test = paralelMonty(FillFration, samples, dimention, interval=interval)
a = test.addition()


#Task 2
sigma = 1.0
x_0 = 5.0
test2 = paralelMonty(Task2Function, samples, dimention, x_0, sigma)
b = test2.addition()

MPI.Finalize()

