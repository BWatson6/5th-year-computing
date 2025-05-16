# -*- coding: utf-8 -*-
"""
Created on Fri May 16 00:34:41 2025

@author: Ben
"""

import numpy as np
from mpi4py import MPI
import again_but_little_different as cc

comm = MPI.COMM_WORLD
numb_of_ranks = comm.Get_size()
rank = comm.Get_rank()

# print(rank)

def troubles_func(greens_potential_array, val_relaxed, n_val, border_indexs):
    
    possion_contribution = np.sum(greens_potential_array[1:n_val+1,1:n_val+1])
    print('---\nposssion part:', possion_contribution)
    true_contribution = val_relaxed - np.sum(greens_potential_array[border_indexs])
    print('should be:', true_contribution)
    print('---\nratio:', true_contribution/possion_contribution)

N_val = 9
bounbry_value = 1
interval = np.array([0, 10e-2]) # in meters to match with SI units
walk_number = 20000
point_of_interest = np.array([5e-2, 5e-2]) # now as spacial coordinate

a = cc.GridThing(N_val, bounbry_value, interval)
cc.boundrys_c(a, N_val)




green_array, green_laplace = a.greens_calc(walk_number, point_of_interest)

answer_per_node = a.function_val(green_array, green_laplace, cc.function0)
# print('on line', answer_per_node)

sumed1_array = comm.reduce(green_array/numb_of_ranks, op=MPI.SUM, root=0)
sumed2_array = comm.reduce(green_laplace/numb_of_ranks, op=MPI.SUM, root=0)

sumed_answe = comm.reduce(answer_per_node/numb_of_ranks, op=MPI.SUM, root=0)

if rank == 0:
    
    
    c = a.function_val(sumed1_array, sumed2_array, cc.function0)
    print('paralel answer', c)
    print('other answer', answer_per_node)
    a.intergrator(cc.function0)
    index = a.coord_to_grid_index(point_of_interest)
    print('real ans', a.grid_array[index[0], index[1]])
    
MPI.Finalize()