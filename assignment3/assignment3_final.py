# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 13:49:04 2025

@author: Ben

Copyright (C) 2025 Ben Watson. Subject to MIT License  
"""

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


def fill_fration(trial_coordinates):
    """
    Task 1 function: takes in a set of coordinates in any dimension and
    calculates its magnitude, if its less than 1 the function returns a 1
    otherwise returns 0
    
    what is returned is an array of 0s and 1s
    """
    radius = 1
    point_dist_from_origin = np.sum(trial_coordinates**2, axis=1)
    points_in_circle_index = np.where(point_dist_from_origin<radius, 1, 0)

    return points_in_circle_index

def task2_function(trial_coordinates, x_0_input, sigma_input):
    """
    Task2 function: takes in a set of coordinates of any dimentions and returns
    an array of the resulting values
    """

    top_of_fraction = -np.linalg.norm((trial_coordinates - x_0_input), axis=1)**2
    function = 1/(sigma_input*np.sqrt(2*np.pi))*np.exp(top_of_fraction/(2*sigma_input**2))
    return function




#%%

def task2_hell(coords_in, x_0_in, sigma_in):
    '''
    Importance sampling function attempt for task 2
    - this function doesnt actualy work but am leaving it in as evidence that I
    tried to get importance sampling working to improve the accuracy of the
    second task. 
    
    When subed into the monte carlo class the value of the intergral should be 
    read off just as the mean of the function and its uncertainty rather than
    the number defined as the intergral as there is no need to ultiply by the 
    boundry conditions.
    
    think the main issue is from incorrectly normalising the weight function
    and the sample_vals function but also likely code errors as it has gotten 
    conviluted
    
    '''

    def task2_base_func(trial_coordinates, x_0_input, sigma_input):
        """
        Task2 function: takes in a set of coordinates of any dimentions and returns
        an array of the resulting values
        """

        top_of_fraction = -np.linalg.norm((trial_coordinates - x_0_input), axis=1)**2
        function = 1/(sigma_input*np.sqrt(2*np.pi))*np.exp(top_of_fraction/(2*sigma_input**2))
        return function

    def weight(coords, x_0_in, sigma_in):
        '''
        weight function is a function that takes the shape of a form e^-x
        when greater than the ofset value and e^x when smaller than that
        '''
        dimenstion = len(coords[0])
        def greater(coords, x_o):
            return np.exp(-(coords-x_o)/(2*sigma_in**2))
        def fewer(coords, x_o):
            return np.exp((coords-x_o)/(2*sigma_in**2))

        function = coords*1.0

        for i in range (len(coords[0])):
            function[:,i] = np.where(coords[:,i]>x_0_in[i],
                                     greater(coords[:,i], x_0_in[i]),
                                     fewer(coords[:,i], x_0_in[i]))

        function = (4*sigma_in**2)**-dimenstion * np.product(function, axis=1)

        return function


    def sample_vals(coords, x_0_in, sigma):
        '''
        this function that is used to change a uniform distribution of random
        numbers into ones with a distribution similar to that of the function
        of Task2
        
        '''

        dimension = len(coords[0])
        normaling = (4*sigma_in**2)**-dimension
        def equ1(coords, x_o):
            value1 = np.log((4*normaling*sigma**2-coords)/(2*normaling*sigma**2))*2*sigma**2
            return x_o-value1

        def equ2(coords, x_o):
            value1 = 2*sigma**2*np.log(coords/(2*normaling*sigma**2))
            return value1+x_o

        new_coords = coords*1.0
        for i in range (len(coords[0])):
            new_coords[:,i] = np.where(coords[:,i]>(2*normaling*sigma),
                                       equ1(coords[:,i], x_0_in[i]),
                                       equ2(coords[:,i], x_0_in[i]))

        return new_coords

    changed_coords = sample_vals(coords_in, x_0_in, sigma_in)
    final_function_part1 = task2_base_func(changed_coords, x_0_in, sigma_in)
    final_function_part2 = weight(changed_coords, x_0_in, sigma_in)
    return final_function_part1/final_function_part2

#%% example code to show how its used



TOTAL_SAMPLES = 1000000



#SAMPLES_PER_RANK = TOTAL_SAMPLES//numb_of_ranks
SAMPLES_PER_RANK = TOTAL_SAMPLES
# Task 1

task1 = MonteCarlo(SAMPLES_PER_RANK, 2, interval=[-1,1])
task1.uniform_sampeling()
task1.intergrate(fill_fration)
task1.combine_paralel()


# Task 2

D=2
#X_0 = np.array([1, 0.5, 2, 1, 0.3, 4])
X_0 = np.ones(D)*4
SIGMA = 10
task2 = MonteCarlo(SAMPLES_PER_RANK, D)
task2.uniform_sampeling()
task2.intergrate(task2_function, X_0, SIGMA)
task2.combine_paralel()

#%% importance sampling - very much doesn't work

X_0 = np.ones(D)*4
SIGMA = 10
# interval here is definitly wrong but at least it runs lol
task2 = MonteCarlo(SAMPLES_PER_RANK, D, interval=[0,5])
task2.uniform_sampeling()
task2.intergrate(task2_function, X_0, SIGMA)
task2.combine_paralel()



MPI.Finalize()
