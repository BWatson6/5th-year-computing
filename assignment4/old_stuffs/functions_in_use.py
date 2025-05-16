# -*- coding: utf-8 -*-
"""
Created on Wed May 14 16:25:50 2025

@author: Ben
"""

import numpy as np
from mpi4py import MPI
import cleaner_code as cc

def troubles_func(greens_potential_array, val_relaxed, n_val, border_indexs):
    
    possion_contribution = np.sum(greens_potential_array[1:n_val+1,1:n_val+1])
    print('---\nposssion part:', possion_contribution)
    true_contribution = val_relaxed - np.sum(greens_potential_array[border_indexs])
    print('should be:', true_contribution)
    print('---\nratio:', true_contribution/possion_contribution)

N_val = 11
initial_value = 5
bounbry_value = 1
interval = np.array([0, 10e-2]) # in meters to match with SI units

point_of_interest = np.array([2.5e-2, 2.5e-2]) # now as spacial coordinate

a = cc.GridThing(N_val, initial_value, bounbry_value, interval)

# cc.boundrys_c(a, N_val)

a.intergrator(cc.function0)
a.plot()
index = a.coord_to_grid_index(point_of_interest)
c0 = a.grid_array[index[0], index[1]]

comm = MPI.COMM_WORLD
numb_of_ranks = comm.Get_size()
rank = comm.Get_rank()
print(rank)
#%%
# green_array, green_varience = a.greens_calc(500, point_of_interest)
# c = a.function_val(green_array, green_varience, cc.function0)
# troubles_func(a.temp, c0, N_val, a.boundry_index)

# sumed_array = comm.reduce(a.temp/numb_of_ranks, op=MPI.SUM, root=0)
# sumed_result = comm.reduce(c/numb_of_ranks, op=MPI.SUM, root=0)

# if rank == 0:
#     troubles_func(sumed_array, c0, N_val, a.boundry_index) # temp is greens*potential
    
#     print('answer', c)
#     print('relaxed method', c0)


# MPI.Finalize()