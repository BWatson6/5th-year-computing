# -*- coding: utf-8 -*-
"""
Created on Fri May  2 12:11:53 2025

@author: Ben
"""

import numpy as np

#clean slate
import time
import math
import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
numb_of_ranks = comm.Get_size()
rank = comm.Get_rank()

start_time = time.time()


class MonteCarlo():
    """
    # How to use class
    1. inisalise MontyCarlos class while defining:
        number of samples,
        number of dimensions,
        interval to intergrate over (if between +/- infinity leave blank)

    2. If wanting to do paralisation call StartMPI

    3. define how you want to generate the random numbers current optioins are:
        uniformSampling - all numbers within interval has equal probiblity
        ImportanceSampling - need to input a function to change how commen some numbers are

    4. call intergrate with desired function to intergrate it

    5. use Paralisation use CombineParalel to output to print out the combined
        integral with its uncertainty
        
    
    """

    def __init__(self, numb_of_samples, dimentions, interval=None):
        self.numb_of_samples = numb_of_samples
        self.dimentions = dimentions
        self.interval = interval
        self.random_inputs = np.array([])
        self.function_samples = np.array([])
        self.expected_val = 0
        self.varience_val = 0
        self.intergal = 0
        self.uncertanty = 0


    def uniform_sampeling(self):
        """
        Generates random numbers with a uniform distribution
        and saves them to random_inputs
        """
        if self.interval is None:
            random_inputs = np.random.uniform(-1, 1,
                                              size=self.numb_of_samples*self.dimentions)
        else:
            random_inputs = np.random.uniform(self.interval[0], self.interval[1],
                                              size=self.numb_of_samples*self.dimentions)

        self.random_inputs = np.resize(random_inputs, (self.numb_of_samples, self.dimentions))


    def importance_sampling(self, function, *args):
        """
        creates an array of random numbers following
        a distribution defined by function with 
        aditional perameters *args
        
        saves to random inputs
        
        --not acctualy nessisary for the way I tried to do importance sampling
        in the end as I but the random number generation into the function
        aswell
        """

        self.uniform_sampeling()
        self.random_inputs = function(self.random_inputs, *args)



    def intergrate(self, function, *args):
        """
        calculates an intergral of a given function with extra 
        perameters *args
        
        saves intergral, uncertainty, expected val of function and its varience
        within the class
        """

        # this seems a bit convoluted but these definitions below shouldn't
        # be called outside of the intergrate funciton?
        def over_all_space(function, t_values, *args):

            def du_part(t_val): # might not be necessary to be a function now
                return(1+t_val**2)/(1-t_val**2)**2

            product_part = np.product(du_part(t_values), axis=1)
            new_coord = t_values/(1-t_values**2)
            new_function = function(new_coord, *args) * product_part
            return new_function

        if self.interval is None:
            self.interval = [-1, 1]
            self.function_samples = over_all_space(function, self.random_inputs, *args)

        else:

            self.function_samples = function(self.random_inputs, *args)

        difference = self.interval[1]-self.interval[0]
        self.expected_val = self.mean(self.function_samples)
        self.varience_val = self.varience(self.function_samples)

        # if interval/boundry become asymetric will need to change these lines
        self.intergal = self.expected_val*(difference)**self.dimentions
        self.uncertanty = np.sqrt(self.varience_val)*difference**self.dimentions
#        print('rank', self.Intergal, '+/-', self.uncertanty)



    def combine_paralel(self):
        """
        calculates a value for intergral and uncertainty using values from
        all the ranks
        
        prints out the intergal +/- the uncertainty
        """
        expected_func_squared = np.mean(self.function_samples**2)

        expected_val_global = comm.reduce(self.expected_val,
                                               op=MPI.SUM, root=0)

        varience_step = comm.reduce(expected_func_squared,
                                         op=MPI.SUM,
                                         root=0)

        if rank==0:
            boundry_product = (self.interval[1]-self.interval[0])**self.dimentions
            intergral_global = (expected_val_global)*boundry_product/numb_of_ranks
            total_samples = numb_of_ranks*self.numb_of_samples

            varience_part1 = 1/numb_of_ranks * varience_step
            varience_part2 = (1/numb_of_ranks * expected_val_global)**2

            varience_global = 1/total_samples * (varience_part1 - varience_part2)
            uncertainty_global = np.sqrt(varience_global)*boundry_product

            end_time = time.time()
            print('------------\n Intergral is:', intergral_global,
                  '+/-', uncertainty_global)
            print('\nfunction is:', expected_val_global/numb_of_ranks,
                  '+/-', uncertainty_global/boundry_product)
            print('\ntime to calculate', end_time-start_time)


    def varience(self, array):
        """
        calculates the varience when not paralised, uses the 
        mean function below, makeing use of fsum, unsure if this is better 
        than just useing the np functions. 
        """
        # wasn't sure I was doing the varience right with np so got this
        part1 = self.mean(array**2)
        part2 = self.mean(array)**2
        var = (part1-part2)/len(array)
        return var

    def mean(self, array):
        """
        returns the mean of an array of values useing fsum to 
        try and increase the accuracy of the sum on each rank

        """
        return math.fsum(array)/len(array)

