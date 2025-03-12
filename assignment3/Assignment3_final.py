# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 13:49:04 2025

@author: Ben
"""

#clean slate
import numpy as np
from mpi4py import MPI


class MontyCarlos():
    def __init__(self, numb_of_samples, dimentions, interval=[0]):
        self.numb_of_samples = numb_of_samples
        self.dimentions = dimentions
        self.interval = interval
        
    def uniformSampeling(self):      
        if len(self.interval)==1:
            t_values = np.random.uniform(-1, 1, size=(self.numb_of_samples*self.dimentions))
            self.random_inputs = np.resize(t_values, (self.numb_of_samples, self.dimentions))
            
        else:
            random_inputs = np.random.uniform(self.interval[0], self.interval[1], 
                                              size=(self.numb_of_samples*self.dimentions))      
            self.random_inputs = np.resize(random_inputs, (self.numb_of_samples, self.dimentions))
        
        return
    
    def ImportanceSampling(self, function, *args):
        self.uniformSampeling()
        self.random_inputs = function(self.random_inputs, *args)
        return
    
    def intergrate(self, function, *args):
        # this seems a bit convoluted but these definitions below shouldn't 
        # be called outside of the intergrate funciton?
        def overAllSpace(function, t_values, *args):

            def duPart(t_val): # might not be necessary to be a function now
                return(1+t_val**2)/(1-t_val**2)**2
            
            product_part = np.product(duPart(t_values), axis=1)
            new_coord = t_values/(1-t_values**2)
            new_function = function(new_coord, *args) * product_part
            return new_function
        
        if len(self.interval)==1:
            self.difference = 2 # will always be 2 here due to how its set up
            self.function_samples = overAllSpace(function, self.random_inputs, *args)

        else:
            self.difference = self.interval[1]-self.interval[0]
            self.function_samples = function(self.random_inputs, *args)
        
        self.expected_val = np.mean(self.function_samples)
        self.varience_val = self.varience(self.function_samples)
        
        # if interval/boundry become asymetric will need to change these lines
        self.Intergal = self.expected_val*(self.difference)**self.dimentions
        self.uncertanty = np.sqrt(self.varience_val)*self.difference**self.dimentions
#        print('rank', self.Intergal, '+/-', self.uncertanty)
        return

    def StartMPI(self):
        self.comm = MPI.COMM_WORLD
        self.numb_of_ranks = self.comm.Get_size()
        self.rank = self.comm.Get_rank()
        return
    
    def CombineParalel(self):
        expected_func_squared = np.mean(self.function_samples**2)
        
        expected_val_global = self.comm.reduce(self.expected_val, 
                                               op=MPI.SUM, root=0)
        
        varience_step = self.comm.reduce(expected_func_squared, 
                                         op=MPI.SUM, 
                                         root=0)
        
        if self.rank==0:
            intergral_global = (expected_val_global)*self.difference**self.dimentions/self.numb_of_ranks
            total_samples = self.numb_of_ranks*self.numb_of_samples
            varience_global = 1/total_samples * (1/self.numb_of_ranks * varience_step - (1/self.numb_of_ranks * expected_val_global)**2)
            uncertainty_global = np.sqrt(varience_global)*self.difference**self.dimentions
            print('Intergral is:', intergral_global, 
                  '+/-', uncertainty_global)
            # MPI.Finalize()
        return
    
    def varience(self, array):
        # wasn't sure I was doing the varience right with np so got this
        part1 = np.mean(array**2)
        part2 = np.mean(array)**2
        var = (part1-part2)/len(array)
        return var
     
        
def FillFration(trial_coordinates):
    radius = 1
    point_dist_from_origin = np.sum(trial_coordinates**2, axis=1)
    points_in_circle_index = np.where(point_dist_from_origin<radius, 1, 0)
    return points_in_circle_index 
            
def Task2Function(trial_coordinates, x_0, sigma):
    top_of_fraction = -np.sum((trial_coordinates-x_0)**2, axis=1)
    function = 1/(sigma*np.sqrt(2*np.pi))*np.exp(top_of_fraction/(2*sigma**2))
    return function



'''
# How to use class
1. inisalise MontyCarlos class while defining:
    number of samples,
    number of dimensions,
    interval to intergrate over (if between +/- infinity leave blank)

2. If wanting to do paralisation call StartMPI

3. define how you want to generate the random numbers current optioins are:
    uniformSampling - all numbers within interval has equal probiblity
    ImportanceSampling - need tom imput a function to change how commen some numbers are
    
4. call intergrate with desired function to intergrate it

5. If using Paralisation use CombineParalel to output to print out the combined 
    integral with its uncertainty
'''


samples_per_rank = 1000000

# Task 1
     
test = MontyCarlos(samples_per_rank, 2, interval=[-1,1])
test.StartMPI()
test.uniformSampeling()
test.intergrate(FillFration)
test.CombineParalel()


# Task 2

d=2
x_0 = np.ones(d)
sigma = 1
test2 = MontyCarlos(samples_per_rank, d)
test2.StartMPI()
test2.uniformSampeling()
test2.intergrate(Task2Function, x_0, sigma)
test2.CombineParalel()
#%% task1 with importance

def Task1ImportanceFunc(x, dimention):
    # probibility part
    r_array = np.sqrt(np.sum(x**2, axis=1))
    distribution = 1/(1+np.exp(-6*r_array+dimention))
    normaling = 6/(np.log((np.exp(6*np.sqrt(dimention))+np.exp(dimention))))
    distribution = distribution*normaling
    
    #Task one functioned copied from above:
    radius = 1
    point_dist_from_origin = np.sum(x**2, axis=1)
    points_in_circle_index = np.where(point_dist_from_origin<radius, 1, 0)
    
    function = points_in_circle_index/distribution
    return function

def Task1RandomDistribution(x, dimention):
    a = -6
    distribution = (np.log(np.exp(x*a)-1)+dimention)/a
    return distribution
d=2
Task1 = MontyCarlos(samples_per_rank, d, interval=[-1, 1])
Task1.StartMPI()
Task1.ImportanceSampling(Task1RandomDistribution, d)
Task1.intergrate(Task1ImportanceFunc, d)
Task1.CombineParalel()


MPI.Finalize()