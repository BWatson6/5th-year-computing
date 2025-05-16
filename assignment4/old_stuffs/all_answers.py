# -*- coding: utf-8 -*-
"""
Created on Wed May 14 16:25:50 2025

@author: Ben

if this code makes no sence its becaus it was written at 4am in the morning
"""

import numpy as np
from mpi4py import MPI
import again_but_little_different as cc

comm = MPI.COMM_WORLD
numb_of_ranks = comm.Get_size()
rank = comm.Get_rank()

#%% the setup
# use 21 in grid for assignment
N_val = 21
bounbry_value = 1
interval = np.array([0, 10e-2]) # in meters to match with SI units
walk_number = 100000 # 100000 for assignment
point1 = np.array([5e-2, 5e-2]) # now as spacial coordinate
point2 = np.array([2.5e-2, 2.5e-2])
point3 = np.array([0.1e-2, 2.5e-2])
point4 = np.array([0.1e-2, 0.1e-2])

# a class for each point
a = cc.GridThing(N_val, bounbry_value, interval)
b = cc.GridThing(N_val, bounbry_value, interval)
c = cc.GridThing(N_val, bounbry_value, interval)
d = cc.GridThing(N_val, bounbry_value, interval)

#%% all thats needed in paralelll stuffs

#green func1
green_array1_para, green_laplace1_para = a.greens_calc(walk_number, point1)
#varience for laplace
v1p1_array = comm.reduce(a.green_laplace, op=MPI.SUM, root=0)
#varience for poisson
v1p2_array = comm.reduce(a.poission_square, op=MPI.SUM, root=0)
v1p3_array = comm.reduce(a.poission_expectaion, op=MPI.SUM, root=0)

# green func2
green_array2_para, green_laplace2_para = b.greens_calc(walk_number, point2)
#varience for laplace
v2p1_array = comm.reduce(b.green_laplace, op=MPI.SUM, root=0)
#varience for poisson
v2p2_array = comm.reduce(b.poission_square, op=MPI.SUM, root=0)
v2p3_array = comm.reduce(b.poission_expectaion, op=MPI.SUM, root=0)

# green func 3
green_array3_para, green_laplace3_para = c.greens_calc(walk_number, point3)
#varience for laplace
v3p1_array = comm.reduce(c.green_laplace, op=MPI.SUM, root=0)
#varience for poisson
v3p2_array = comm.reduce(c.poission_square, op=MPI.SUM, root=0)
v3p3_array = comm.reduce(c.poission_expectaion, op=MPI.SUM, root=0)

#green func 4
green_array4_para, green_laplace4_para = d.greens_calc(walk_number, point4)
#varience for laplace
v4p1_array = comm.reduce(d.green_laplace, op=MPI.SUM, root=0)
#varience for poisson
v4p2_array = comm.reduce(d.poission_square, op=MPI.SUM, root=0)
v4p3_array = comm.reduce(d.poission_expectaion, op=MPI.SUM, root=0)



# need this for each of the greens arrays
green_array1 = comm.reduce(green_array1_para/numb_of_ranks, op=MPI.SUM, root=0)
green_laplace1 = comm.reduce(green_laplace1_para/numb_of_ranks, op=MPI.SUM, root=0)


green_array2 = comm.reduce(green_array2_para/numb_of_ranks, op=MPI.SUM, root=0)
green_laplace2 = comm.reduce(green_laplace2_para/numb_of_ranks, op=MPI.SUM, root=0)

green_array3 = comm.reduce(green_array3_para/numb_of_ranks, op=MPI.SUM, root=0)
green_laplace3 = comm.reduce(green_laplace3_para/numb_of_ranks, op=MPI.SUM, root=0)

green_array4 = comm.reduce(green_array4_para/numb_of_ranks, op=MPI.SUM, root=0)
green_laplace4 = comm.reduce(green_laplace4_para/numb_of_ranks, op=MPI.SUM, root=0)


#%% stuff don on rank 1

if rank == 0:
    
    fig1, ((ax11, ax12),
           (ax13, ax14)) = plt.subplots()
    
    
#%%

    tot_walks = numb_of_ranks*walk_number
    index = a.coord_to_grid_index(point1)
    print('real ans', a.grid_array[index[0], index[1]])
    # calcualtions for global variences of greens functions
    a.varience_boundry = 1/tot_walks * (v1p1_array-v1p1_array**2)
    a.varience_walks = 1/tot_walks * (v1p2_array-v1p3_array**2)
    
    b.varience_boundry = 1/tot_walks * (v2p1_array-v2p1_array**2)
    b.varience_walks = 1/tot_walks * (v2p2_array-v2p3_array**2)
    
    c.varience_boundry = 1/tot_walks * (v3p1_array-v3p1_array**2)
    c.varience_walks = 1/tot_walks * (v3p2_array-v3p3_array**2)
    
    #ex4
    ## function = 0 boundrys (a)
    cc.boundrys_a(a, N_val)
    # a.intergrator(cc.function0)
    # check_a = a.grid_array*1

    
    ex4a1 = a.function_val(green_array1, green_laplace1, cc.function0)
    ex4a2 = b.function_val(green_array2, green_laplace2, cc.function0)
    ex4a3 = c.function_val(green_array3, green_laplace3, cc.function0)
    ex4a4 = d.function_val(green_array4, green_laplace4, cc.function0)
    
    print('------\n\nEx4\n')
    print('---\nf(x, y) = 0')
    print('+1V at all boundrys')
    print('at (5, 5)cm', ex4a1)
    print('at (2.5, 2.5)cm', ex4a2)
    print('at (0.1, 2.5)cm', ex4a3)
    print('at (0.1, 0.1)cm', ex4a4) 
    
    ## function = 0 boundrys (b)
    cc.boundrys_b(a, N_val)
    # a.intergrator(cc.function0)
    # check_b = a.grid_array
    ex4b1 = a.function_val(green_array1, green_laplace1, cc.function0)
    ex4b2 = b.function_val(green_array2, green_laplace2, cc.function0)
    ex4b3 = c.function_val(green_array3, green_laplace3, cc.function0)
    ex4b4 = d.function_val(green_array4, green_laplace4, cc.function0)
    print('\n+1V top/bottem, -1V left/right')
    print('at (5, 5)cm', ex4b1)
    print('at (2.5, 2.5)cm', ex4b2)
    print('at (0.1, 2.5)cm', ex4b3)
    print('at (0.1, 0.1)cm', ex4b4)
    
    ##function = 0 boundrys (c)
    cc.boundrys_c(a, N_val)
    ex4c1 = a.function_val(green_array1, green_laplace1, cc.function0)
    ex4c2 = b.function_val(green_array2, green_laplace2, cc.function0)
    ex4c3 = c.function_val(green_array3, green_laplace3, cc.function0)
    ex4c4 = d.function_val(green_array4, green_laplace4, cc.function0)
    print('\n +2V top/left, 0V bottem, -4V right')
    print('at (5, 5)cm', ex4c1)
    print('at (2.5, 2.5)cm', ex4c2)
    print('at (0.1, 2.5)cm', ex4c3)
    print('at (0.1, 0.1)cm', ex4c4)
    
    #NEXT FUNCTION
    print('\n---\nf(x,y) = 10V')
    cc.boundrys_a(a, N_val)
    # a.intergrator(cc.function0)
    # check_a = a.grid_array*1
    ex4a1 = a.function_val(green_array1, green_laplace1, cc.function_a)
    ex4a2 = b.function_val(green_array2, green_laplace2, cc.function_a)
    ex4a3 = c.function_val(green_array3, green_laplace3, cc.function_a)
    ex4a4 = d.function_val(green_array4, green_laplace4, cc.function_a)
    print('+1V at all boundrys')
    print('at (5, 5)cm', ex4a1)
    print('at (2.5, 2.5)cm', ex4a2)
    print('at (0.1, 2.5)cm', ex4a3)
    print('at (0.1, 0.1)cm', ex4a4) 
    
    ##boundrys (b)
    cc.boundrys_b(a, N_val)
    # a.intergrator(cc.function0)
    # check_b = a.grid_array
    ex4b1 = a.function_val(green_array1, green_laplace1, cc.function_a)
    ex4b2 = b.function_val(green_array2, green_laplace2, cc.function_a)
    ex4b3 = c.function_val(green_array3, green_laplace3, cc.function_a)
    ex4b4 = d.function_val(green_array4, green_laplace4, cc.function_a)
    print('\n+1V top/bottem, -1V left/right')
    print('at (5, 5)cm', ex4b1)
    print('at (2.5, 2.5)cm', ex4b2)
    print('at (0.1, 2.5)cm', ex4b3)
    print('at (0.1, 0.1)cm', ex4b4)
    
    ##boundrys (c)
    cc.boundrys_c(a, N_val)
    ex4c1 = a.function_val(green_array1, green_laplace1, cc.function_a)
    ex4c2 = b.function_val(green_array2, green_laplace2, cc.function_a)
    ex4c3 = c.function_val(green_array3, green_laplace3, cc.function_a)
    ex4c4 = d.function_val(green_array4, green_laplace4, cc.function_a)
    print('\n +2V top/left, 0V bottem, -4V right')
    print('at (5, 5)cm', ex4c1)
    print('at (2.5, 2.5)cm', ex4c2)
    print('at (0.1, 2.5)cm', ex4c3)
    print('at (0.1, 0.1)cm', ex4c4)
    
    #NEXT FUNCTION
    print('\n---\nf(x,y) = gradient')
    cc.boundrys_a(a, N_val)
    # a.intergrator(cc.function0)
    # check_a = a.grid_array*1
    ex4a1 = a.function_val(green_array1, green_laplace1, cc.function_b)
    ex4a2 = b.function_val(green_array2, green_laplace2, cc.function_b)
    ex4a3 = c.function_val(green_array3, green_laplace3, cc.function_b)
    ex4a4 = d.function_val(green_array4, green_laplace4, cc.function_b)
    print('+1V at all boundrys')
    print('at (5, 5)cm', ex4a1)
    print('at (2.5, 2.5)cm', ex4a2)
    print('at (0.1, 2.5)cm', ex4a3)
    print('at (0.1, 0.1)cm', ex4a4) 
    
    ##boundrys (b)
    cc.boundrys_b(a, N_val)
    # a.intergrator(cc.function0)
    # check_b = a.grid_array
    ex4b1 = a.function_val(green_array1, green_laplace1, cc.function_b)
    ex4b2 = b.function_val(green_array2, green_laplace2, cc.function_b)
    ex4b3 = c.function_val(green_array3, green_laplace3, cc.function_b)
    ex4b4 = d.function_val(green_array4, green_laplace4, cc.function_b)
    print('\n+1V top/bottem, -1V left/right')
    print('at (5, 5)cm', ex4b1)
    print('at (2.5, 2.5)cm', ex4b2)
    print('at (0.1, 2.5)cm', ex4b3)
    print('at (0.1, 0.1)cm', ex4b4)
    
    ##boundrys (c)
    cc.boundrys_c(a, N_val)
    ex4c1 = a.function_val(green_array1, green_laplace1, cc.function_b)
    ex4c2 = b.function_val(green_array2, green_laplace2, cc.function_b)
    ex4c3 = c.function_val(green_array3, green_laplace3, cc.function_b)
    ex4c4 = d.function_val(green_array4, green_laplace4, cc.function_b)
    print('\n +2V top/left, 0V bottem, -4V right')
    print('at (5, 5)cm', ex4c1)
    print('at (2.5, 2.5)cm', ex4c2)
    print('at (0.1, 2.5)cm', ex4c3)
    print('at (0.1, 0.1)cm', ex4c4)
    
    #NEXT FUNCTION
    print('\n---\nf(x,y) = point charge at centre')
    cc.boundrys_a(a, N_val)
    # a.intergrator(cc.function0)
    # check_a = a.grid_array*1
    ex4a1 = a.function_val(green_array1, green_laplace1, cc.function_c)
    ex4a2 = b.function_val(green_array2, green_laplace2, cc.function_c)
    ex4a3 = c.function_val(green_array3, green_laplace3, cc.function_c)
    ex4a4 = d.function_val(green_array4, green_laplace4, cc.function_c)
    print('+1V at all boundrys')
    print('at (5, 5)cm', ex4a1)
    print('at (2.5, 2.5)cm', ex4a2)
    print('at (0.1, 2.5)cm', ex4a3)
    print('at (0.1, 0.1)cm', ex4a4) 
    
    ##boundrys (b)
    cc.boundrys_b(a, N_val)
    # a.intergrator(cc.function0)
    # check_b = a.grid_array
    ex4b1 = a.function_val(green_array1, green_laplace1, cc.function_c)
    ex4b2 = b.function_val(green_array2, green_laplace2, cc.function_c)
    ex4b3 = c.function_val(green_array3, green_laplace3, cc.function_c)
    ex4b4 = d.function_val(green_array4, green_laplace4, cc.function_c)
    print('\n+1V top/bottem, -1V left/right')
    print('at (5, 5)cm', ex4b1)
    print('at (2.5, 2.5)cm', ex4b2)
    print('at (0.1, 2.5)cm', ex4b3)
    print('at (0.1, 0.1)cm', ex4b4)
    
    ##boundrys (c)
    cc.boundrys_c(a, N_val)
    ex4c1 = a.function_val(green_array1, green_laplace1, cc.function_c)
    ex4c2 = a.function_val(green_array2, green_laplace2, cc.function_c)
    ex4c3 = a.function_val(green_array3, green_laplace3, cc.function_c)
    ex4c4 = a.function_val(green_array4, green_laplace4, cc.function_c)
    print('\n +2V top/left, 0V bottem, -4V right')
    print('at (5, 5)cm', ex4c1)
    print('at (2.5, 2.5)cm', ex4c2)
    print('at (0.1, 2.5)cm', ex4c3)
    print('at (0.1, 0.1)cm', ex4c4)
    
    MPI.Finalize()