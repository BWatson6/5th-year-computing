# -*- coding: utf-8 -*-
"""
Created on Mon May 12 09:27:46 2025

@author: Ben

greens function in parrelell 
"""


'''
when we call the function we want to initiate open mpi

from the total number of walkers we want to divide it up as best as possible 
between the nodes

if theres not an exact split then we reduce the number of samples to that so it can evenly divide

on each of the ranks we send out walkers and record where they are going on an
empty grid that includes the boundrys

will also need to add values for the number of times asite has been visited 
squared per walker so that we have enough info for doing the variences 

once I have got that I can find the averarage number of times that each site is visited 
and varience for each node and then use mpi sum to combine - yes I can do that






'''
import numpy as np
from mpi4py import MPI


comm = MPI.COMM_WORLD
numb_of_ranks = comm.Get_size()
rank = comm.Get_rank()

arrray = np.ones((numb_of_ranks, 4))*rank

test_array=comm.reduce(arrray, op=MPI.SUM, root=0)

if rank==0:
    print(test_array)












