# -*- coding: utf-8 -*-
"""
Created on Fri May 16 00:58:25 2025

@author: Ben
checking answers
"""
import numpy as np
import again_but_little_different as cc


N_val = 21
bounbry_value = 1
interval = np.array([0, 10e-2]) # in meters to match with SI units
point1 = np.array([5e-2, 5e-2]) # now as spacial coordinate
point2 = np.array([2.5e-2, 2.5e-2])
point3 = np.array([0.1e-2, 2.5e-2])
point4 = np.array([0.1e-2, 0.1e-2])

a = cc.GridThing(N_val, bounbry_value, interval)

index1 = a.coord_to_grid_index(point1)
index2 = a.coord_to_grid_index(point2)
index3 = a.coord_to_grid_index(point3)
index4 = a.coord_to_grid_index(point4)

print('------\n\nEx5\n')
print('---\nf(x, y) = 0')

cc.boundrys_a(a, N_val)
a.intergrator(cc.function0)
print('+1V at all boundrys')
print('at (5, 5)cm', a.grid_array[index1[0], index1[1]])
print('at (2.5, 2.5)cm', a.grid_array[index2[0], index2[1]])
print('at (0.1, 2.5)cm', a.grid_array[index3[0], index3[1]])
print('at (0.1, 0.1)cm', a.grid_array[index4[0], index4[1]]) 
 

cc.boundrys_b(a, N_val)
a.intergrator(cc.function0)
print('+1V top/bottem, -1V left/right')
print('at (5, 5)cm', a.grid_array[index1[0], index1[1]])
print('at (2.5, 2.5)cm', a.grid_array[index2[0], index2[1]])
print('at (0.1, 2.5)cm', a.grid_array[index3[0], index3[1]])
print('at (0.1, 0.1)cm', a.grid_array[index4[0], index4[1]])

cc.boundrys_c(a, N_val)
a.intergrator(cc.function0)
print('+2V top/left, 0V bottem, -4V right')
print('at (5, 5)cm', a.grid_array[index1[0], index1[1]])
print('at (2.5, 2.5)cm', a.grid_array[index2[0], index2[1]])
print('at (0.1, 2.5)cm', a.grid_array[index3[0], index3[1]])
print('at (0.1, 0.1)cm', a.grid_array[index4[0], index4[1]])
#%%
## first function done
print('\n')
print('---\nf(x, y) = 10')
cc.boundrys_a(a, N_val)
a.intergrator(cc.function_a)
print('+1V at all boundrys')
print('at (5, 5)cm', a.grid_array[index1[0], index1[1]])
print('at (2.5, 2.5)cm', a.grid_array[index2[0], index2[1]])
print('at (0.1, 2.5)cm', a.grid_array[index3[0], index3[1]])
print('at (0.1, 0.1)cm', a.grid_array[index4[0], index4[1]]) 
 

cc.boundrys_b(a, N_val)
a.intergrator(cc.function_a)
print('+1V top/bottem, -1V left/right')
print('at (5, 5)cm', a.grid_array[index1[0], index1[1]])
print('at (2.5, 2.5)cm', a.grid_array[index2[0], index2[1]])
print('at (0.1, 2.5)cm', a.grid_array[index3[0], index3[1]])
print('at (0.1, 0.1)cm', a.grid_array[index4[0], index4[1]])

cc.boundrys_c(a, N_val)
a.intergrator(cc.function_a)
print('+2V top/left, 0V bottem, -4V right')
print('at (5, 5)cm', a.grid_array[index1[0], index1[1]])
print('at (2.5, 2.5)cm', a.grid_array[index2[0], index2[1]])
print('at (0.1, 2.5)cm', a.grid_array[index3[0], index3[1]])
print('at (0.1, 0.1)cm', a.grid_array[index4[0], index4[1]])
#%%

print('\n')
print('---\nf(x, y) = gradient')
cc.boundrys_a(a, N_val)
a.intergrator(cc.function_b)
print('+1V at all boundrys')
print('at (5, 5)cm', a.grid_array[index1[0], index1[1]])
print('at (2.5, 2.5)cm', a.grid_array[index2[0], index2[1]])
print('at (0.1, 2.5)cm', a.grid_array[index3[0], index3[1]])
print('at (0.1, 0.1)cm', a.grid_array[index4[0], index4[1]]) 
 

cc.boundrys_b(a, N_val)
a.intergrator(cc.function_b)
print('+1V top/bottem, -1V left/right')
print('at (5, 5)cm', a.grid_array[index1[0], index1[1]])
print('at (2.5, 2.5)cm', a.grid_array[index2[0], index2[1]])
print('at (0.1, 2.5)cm', a.grid_array[index3[0], index3[1]])
print('at (0.1, 0.1)cm', a.grid_array[index4[0], index4[1]])

cc.boundrys_c(a, N_val)
a.intergrator(cc.function_b)
print('+2V top/left, 0V bottem, -4V right')
print('at (5, 5)cm', a.grid_array[index1[0], index1[1]])
print('at (2.5, 2.5)cm', a.grid_array[index2[0], index2[1]])
print('at (0.1, 2.5)cm', a.grid_array[index3[0], index3[1]])
print('at (0.1, 0.1)cm', a.grid_array[index4[0], index4[1]])

#%%

print('\n')
print('---\nf(x, y) = point_charge')
cc.boundrys_a(a, N_val)
a.intergrator(cc.function_c)
print('+1V at all boundrys')
print('at (5, 5)cm', a.grid_array[index1[0], index1[1]])
print('at (2.5, 2.5)cm', a.grid_array[index2[0], index2[1]])
print('at (0.1, 2.5)cm', a.grid_array[index3[0], index3[1]])
print('at (0.1, 0.1)cm', a.grid_array[index4[0], index4[1]]) 
 

cc.boundrys_b(a, N_val)
a.intergrator(cc.function_c)
print('+1V top/bottem, -1V left/right')
print('at (5, 5)cm', a.grid_array[index1[0], index1[1]])
print('at (2.5, 2.5)cm', a.grid_array[index2[0], index2[1]])
print('at (0.1, 2.5)cm', a.grid_array[index3[0], index3[1]])
print('at (0.1, 0.1)cm', a.grid_array[index4[0], index4[1]])

cc.boundrys_c(a, N_val)
a.intergrator(cc.function_c)
print('+2V top/left, 0V bottem, -4V right')
print('at (5, 5)cm', a.grid_array[index1[0], index1[1]])
print('at (2.5, 2.5)cm', a.grid_array[index2[0], index2[1]])
print('at (0.1, 2.5)cm', a.grid_array[index3[0], index3[1]])
print('at (0.1, 0.1)cm', a.grid_array[index4[0], index4[1]])
