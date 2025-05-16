# -*- coding: utf-8 -*-
"""
Created on Fri May 16 00:34:41 2025

Copyright (C) 2025 Ben Watson. Subject to MIT License

Trying to use the class with paralisation
"""

import numpy as np
from mpi4py import MPI
import cleaner_code as cc

comm = MPI.COMM_WORLD
numb_of_ranks = comm.Get_size()
rank = comm.Get_rank()

# print(rank)

N_VAL = 9
BOUNDRY_VAL = 1
GRID_START_VAL = 0
interval = np.array([0, 10e-2]) # in meters to match with SI units
WALK_NUMB = 20000
point_of_interest = np.array([2.5e-2, 5e-2]) # now as spacial coordinate

a = cc.GridThing(N_VAL, GRID_START_VAL, BOUNDRY_VAL, interval)
cc.boundrys_c(a, N_VAL)

green_array, green_varience = a.greens_calc(WALK_NUMB, point_of_interest)


sumed1_array = comm.reduce(green_array/numb_of_ranks, op=MPI.SUM, root=0)
sumed2_array = comm.reduce(green_varience/numb_of_ranks, op=MPI.SUM, root=0)


if rank == 0:
    MPI.WAI()
    # walker answer
    c = a.function_val(sumed1_array, sumed2_array, cc.function0)
    print('paralel answer', c)

    # deterministic check
    a.intergrator(cc.function0)
    index = a.coord_to_grid_index(point_of_interest)
    print('real ans', a.grid_array[index[0], index[1]])
    MPI.Finalize()
    