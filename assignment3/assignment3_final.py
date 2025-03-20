# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 13:49:04 2025

@author: Ben
"""

#clean slate
import time
import math
import numpy as np
from mpi4py import MPI


start_time = time.time()


class MontyCarlos():
    """
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
            print('Intergral is:', intergral_global,
                  '+/-', uncertainty_global)
            print('time to calculate', end_time-start_time)


    def varience(self, array):
        """
        calculates the varience when not paralised 
        """
        # wasn't sure I was doing the varience right with np so got this
        part1 = np.mean(array**2)
        part2 = np.mean(array)**2
        var = (part1-part2)/len(array)
        return var

    def mean(self, array):
        """
        returns the mean of an array of values useing fsum to 
        try and increase the accuracy of the sum

        """
        return math.fsum(array)/len(array)


def start_mpi():
    """
    initalises parallel codeing and returns:
    comm - comunication class?
    number of ranks
    rank index
    
    """
    comm_in = MPI.COMM_WORLD
    numb_of_ranks_in = comm_in.Get_size()
    rank_in = comm_in.Get_rank()
    return comm_in, numb_of_ranks_in, rank_in

def end_mpi():
    """
    unnecessary definition to stop the paralisation 
    """
    MPI.Finalize()

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


#%% task1 with importance

# def Task1ImportanceFunc(x, dimention):
#     # probibility part
#     r_array = np.sqrt(np.sum(x**2, axis=1))
#     distribution = 1/(1+np.exp(-6*r_array+dimention))
#     normaling = 6/(np.log((np.exp(6*np.sqrt(dimention))+np.exp(dimention))))
#     distribution = distribution*normaling

#     #Task one functioned copied from above:
#     radius = 1
#     point_dist_from_origin = np.sum(x**2, axis=1)
#     points_in_circle_index = np.where(point_dist_from_origin<radius, 1, 0)

#     function = points_in_circle_index/distribution
#     return function

# def Task1RandomDistribution(x, dimention):
#     a = -6
#     distribution = (np.log(np.exp(x*a)-1)+dimention)/a
#     return distribution


#%%



TOTAL_SAMPLES = 10000000

comm, numb_of_ranks, rank = start_mpi()



#SAMPLES_PER_RANK = TOTAL_SAMPLES//numb_of_ranks
SAMPLES_PER_RANK = TOTAL_SAMPLES
# Task 1

test = MontyCarlos(SAMPLES_PER_RANK, 2, interval=[-1,1])
test.uniform_sampeling()
test.intergrate(fill_fration)
test.combine_paralel()




# test = MontyCarlos(SAMPLES_PER_RANK, 3, interval=[-1,1])
# test.uniform_sampeling()
# test.intergrate(fill_fration)
# test.combine_paralel()


# test = MontyCarlos(SAMPLES_PER_RANK, 4, interval=[-1,1])
# test.uniform_sampeling()
# test.intergrate(fill_fration)
# test.combine_paralel()


# test = MontyCarlos(SAMPLES_PER_RANK, 5, interval=[-1,1])
# test.uniform_sampeling()
# test.intergrate(fill_fration)
# test.combine_paralel()


# Task 2

D=6
#X_0 = np.array([1, 0.5, 2, 1, 0.3, 4])
X_0 = np.ones(D)*0
SIGMA = 10
test2 = MontyCarlos(SAMPLES_PER_RANK, D)
test2.uniform_sampeling()
test2.intergrate(task2_function, X_0, SIGMA)
test2.combine_paralel()

if rank==0:
    print('\n\n',max(test2.function_samples))
    print(min(test2.function_samples))


# X_0 = -6
# SIGMA = 1
# test2 = MontyCarlos(SAMPLES_PER_RANK, D)
# test2.uniform_sampeling()
# test2.intergrate(task2_function, X_0, SIGMA)
# test2.combine_paralel()

# X_0 = 20
# SIGMA = 1
# test2 = MontyCarlos(SAMPLES_PER_RANK, D)
# test2.uniform_sampeling()
# test2.intergrate(task2_function, X_0, SIGMA)
# test2.combine_paralel()


# D=6
# X_0 = np.array([5, 6, 7, 8, 9, 4])
# SIGMA = 1
# test2 = MontyCarlos(SAMPLES_PER_RANK, D)
# test2.uniform_sampeling()
# test2.intergrate(task2_function, X_0, SIGMA)
# test2.combine_paralel()

# X_0 = np.array([0.2, 4, 7, 7, 0.5, 20])
# SIGMA = 10
# test2 = MontyCarlos(SAMPLES_PER_RANK, D)
# test2.uniform_sampeling()
# test2.intergrate(task2_function, X_0, SIGMA)
# test2.combine_paralel()

# X_0 = np.array([20, -4, -5, 6, 6, -0.5])
# SIGMA = 0.5
# test2 = MontyCarlos(SAMPLES_PER_RANK, D)
# test2.uniform_sampeling()
# test2.intergrate(task2_function, X_0, SIGMA)
# test2.combine_paralel()

end_mpi()
