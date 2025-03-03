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
    def __init__(self, function, numb_of_samples, dimentions, interval=[0], *args):
        self.dimentions = dimentions
        self.numb_of_samples = numb_of_samples

        # creating random values and reshaping to be coordinats space of the defined dimention
        random_inputs = np.random.uniform(interval[0], interval[1], 
                                          size=(numb_of_samples*dimentions))       
        random_inputs = np.resize(random_inputs, (numb_of_samples, dimentions))
        
        
        # inputing the random input values into the function getting evaluated
        sampled_function = function(random_inputs, *args)

        # not doing proper monte carlo but this is ration of points inside to 
        # total number of points
        Intergral = np.sum(sampled_function)/numb_of_samples
        
        # saves the value of the volume of the round thing to the class
        self.value = Intergral * 2**dimentions

        # this part here is a test will see if it works for task 1
        # from the function you get the number of sucsesses
        
        self.hit_probibility = Intergral




    def addition(self):
        # rank sum is adding all the values for volum together and sending to rank 0
        rank_sum = comm.reduce(self.value, op=MPI.SUM, root=0)
        # V[X] = E[X^2]-E[X]^2 the following value is used to calculate E[X^2]
        varience_sum_step = comm.reduce(self.value**2, op=MPI.SUM, root=0)
        
        # TEST PART MIGHT DELETE LATER
        # divided by the number of ranks to calculate the global probibility
        # need to divide by the number of ranks (kinda a mean)
        probibility_global = comm.reduce(self.hit_probibility/numb_of_ranks, 
                                         op=MPI.SUM, 
                                         root=0)

        # rank 0 calcualates the statistics with what it recives from the other ranks
        if comm.Get_rank()==0: 

            # calculation expected and varience value and prints them out
            # not sure if the varinece quite works yet 
            expected_val = rank_sum/numb_of_ranks
            variance = (varience_sum_step/numb_of_ranks) - expected_val**2

            print('round thing volume:', expected_val)
#            print('variance:', variance)

            Volume = probibility_global * 2**self.dimentions
#            print('\nvolume:', Volume)
            # for total number of samples do samples_per_rank * rank
            total_samples = self.numb_of_samples*numb_of_ranks
            mean_val = probibility_global*total_samples
#            print('mean:', mean_val)
            varience_val = total_samples*probibility_global*(1-probibility_global)
            print('varience', varience_val/total_samples)
            return mean_val, varience_val



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


def Task2Function(trial_coordinates, x_0, sigma):

    top_of_fraction = -np.sum((trial_coordinates-x_0)**2, axis=1)
    function = 1/(sigma*np.sqrt(2*np.pi))*np.exp(top_of_fraction/2*sigma**2)
    return function

def OverAllSpace(function, dimention, samples, *args):
    t_values = np.random.uniform(-1, 1, size=(samples*dimention))
    t_values = np.resize(t_values, (samples, dimention))
    
    new_coord = t_values/(1-t_values**2)
    new_function = function(new_coord, *args) * (1+t_values**2)/(1-t_values**2)**2
    return new_function
    
    
    
    


dimention = 2
samples = 10000000
interval = [-1, 1] # square/cube is centerd on the origin for simplicity 

test = paralelMonty(FillFration, samples, dimention, interval)
a = test.addition()

test2 = paralelMonty(Task2Function, samples, dimention)

MPI.Finalize()

#will want to have it so that we can intergrate over all space insode the class i think but anyways...

