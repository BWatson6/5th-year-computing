# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 14:27:30 2025

@author: Ben
"""

import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
numb_of_ranks = comm.Get_size()

class paralelMonty():
    # in the initiation it basicaly does the monte carlo part of this
    def __init__(self, function, interval, numb_of_samples, dimentions):

        # creating random values and reshaping to be coordinats space of the defined dimention
        random_inputs = np.random.uniform(interval[0], interval[1], 
                                          size=(numb_of_samples*dimentions))       
        random_inputs = np.resize(random_inputs, (numb_of_samples, dimentions))
        
        
        # inputing the random input values into the function getting evaluated
        sampled_function = function(random_inputs)
        
        # not doing proper monte carlo but this is ration of points inside to 
        # total number of points
        Intergral = np.sum(sampled_function)/numb_of_samples
        
        # saves the value of the volume of the round thing to the class
        self.value = Intergral * 2**dimentions
    
    def addition(self):
        # rank sum is adding all the values for volum together and sending to rank 0
        rank_sum = comm.reduce(self.value, op=MPI.SUM, root=0)
        # V[X] = E[X^2]-E[X]^2 the following value is used to calculate E[X^2]
        varience_sum_step = comm.reduce(self.value**2, op=MPI.SUM, root=0)
        
        # rank 0 calcualates the statistics with what it recives from the other ranks
        if comm.Get_rank()==0: 
            
            # calculation expected and varience value and prints them out
            # not sure if the varinece quite works yet 
            expected_val = rank_sum/numb_of_ranks
            variance = (varience_sum_step/numb_of_ranks) - expected_val**2

            print('round thing volume:', expected_val)
            print('variance:', variance)


# def MontyCarloish(function, interval, numb_of_samples, dimentions):  
#     random_inputs = np.random.uniform(interval[0], interval[1], 
#                                       size=(numb_of_samples*dimentions))
    
#     random_inputs = np.resize(random_inputs, (numb_of_samples, dimentions))
#     sampled_function = function(random_inputs)

#     Intergral = np.sum(sampled_function)/numb_of_samples
#     return Intergral*2**dimentions # area/volume whatever of round thing

#%%


def FillFration(trial_coordinates):
    radius = 1
    point_dist_from_origin = np.sum(trial_coordinates**2, axis=1)
    points_in_circle_index = np.where(point_dist_from_origin<radius)
    return len(points_in_circle_index[0])



dimention = 2
samples = 10000
interval = [-1, 1] # square/cube is centerd on the origin for simplicity 

test = paralelMonty(FillFration, interval, samples, dimention)
test.addition()

MPI.Finalize()

