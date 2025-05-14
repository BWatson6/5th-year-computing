# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 17:01:06 2025

@author: Ben
"""

import numpy as np
import matplotlib.pyplot as plt

from matplotlib import cm
#from montycarlo_class import MonteCarlo

class gridthing():
    def __init__(self, N, start_val, boundry_val, interval):
        # - this sets up a grid that is NxN with a boumdry around that
        # - start value is the initial value for all possitions on the grid 
        # - boundry value is the fixed value at the boundry of the area that 
        # is being intergrated over
        # - interval is the is the real space that the grid takes up
        self.N = N
        self.boundry_val = boundry_val
        self.interval = interval
        # makeing up the grid
        grid_array = np.zeros((N+2, N+2))
        N_grid = np.ones((N, N)) * start_val
        boundrys = np.ones(N) * boundry_val
        grid_array[1:N+1,1:N+1] = N_grid
        grid_array[1:N+1,0], grid_array[1:N+1,N+1] = boundrys, boundrys
        grid_array[0][1:N+1], grid_array[N+1][1:N+1] = boundrys, boundrys
        self.grid_array = grid_array
        self.start_copy = self.grid_array*1
        self.boundry_index = np.where(grid_array==boundry_val)
    
    # now for the iteration parts hahahaha :'(
    def intergrator(self, function):
        grid_array = self.grid_array
        X = np.linspace(self.interval[0], self.interval[1], self.N)
        Y = X * 1
        h = (self.interval[1]-self.interval[0])/self.N
        change = 1.0
        counter=1
        
        # over relaxed equation for omega from notes.
        omega = 2/(1+np.sin(np.pi/self.N))
        
        # might need to change this error thing if code gets stuck in a loop
        # currently have it so that the larger grid (N) the smaller the error
        # must be.
        while change > 1e-4/self.N**2:         
            previous_grid = grid_array*1
            for y_counter in range (1, self.N+1):
                for x_counter in range (1, self.N+1):
                    part1 = grid_array[y_counter+1][x_counter]+grid_array[y_counter-1][x_counter]+grid_array[y_counter][x_counter+1]+grid_array[y_counter][x_counter-1]
                    part2 = h**2* function(X[x_counter-1], Y[y_counter-1])
                    grid_array[y_counter][x_counter] = 0.25*(part1+part2)

                    #overrelaxed_version
                    # grid_array[y_counter][x_counter] = 0.25*omega*(part1+part2)+(1-omega)*grid_array[y_counter][x_counter]


            if counter>10: # this is just to make sure that theres no divide by 0 issue wth this   
                change = np.max(np.abs((previous_grid[1:self.N+1,1:self.N+1]-grid_array[1:self.N+1,1:self.N+1])/previous_grid[1:self.N+1,1:self.N+1]))
            # print('change value', change)
            counter +=1
        self.previous_grid = previous_grid
        self.grid_array=grid_array
        print('largest change from previous iteration', change)
        print('number of iterations', counter)
        


    def plot(self):
        # after loop
        ax = plt.axes(projection='3d')
        ax.set_xlim3d(self.interval[0], self.interval[1])
        ax.set_ylim3d(self.interval[0], self.interval[1])
        X = np.linspace(self.interval[0], self.interval[1], self.N)
        Y = X * 1
        X, Y = np.meshgrid(X, Y)
        N_grid = self.grid_array[1:self.N+1,1:self.N+1]
        ax.plot_surface(X,Y, N_grid, cmap=cm.coolwarm_r)


    def walker_vesion2(self, walk_numb, start_value):
        # input is coordinats in the form (x,y) need to also switch order to 
        # get into indexing format
        h = (self.interval[1]-self.interval[0])/self.N
        print('h value', h)
        start_index = (start_value-self.interval[0])//h
        start_index = (start_index+1).astype(int) # the plus 1 to include the indexes of the voundry condisions
        start_index = [start_index[1], start_index[0]] # flipping the order
        # print('Point index:', start_index)
        self.start_index = start_index # just for trouble shooting
        # assuming that the start index is in terms of the grid_array index
        global_counts = self.grid_array*0
        global_counts_squared = self.grid_array*0
        
        step_conter = 0
        for i in range (0, walk_numb):
            
            position = start_index*1
            walker_counts = self.grid_array*0
            while position[0] in range(1, self.N+1) and position[1] in range(1, self.N+1):
                direction = np.random.choice([0, 1, 2, 3], 
                                             p=[0.25, 0.25, 0.25, 0.25], 
                                             size=(1))
                walker_counts[position[0], position[1]] += 1 # adding a step to the walk
                if direction==1:
                    position = [position[0]+1, position[1]] # +1 in y direction                     
                elif direction==2:
                    position = [position[0], position[1]+1] # +1 in x direction
                elif direction==3:
                    position = [position[0]-1, position[1]] # -1 in y direction
                else:
                    position = [position[0], position[1]-1] # -1 in x direction
                    
                
                step_conter += 1
            # after a walk is finiished
            # for greens function
            global_counts[1:self.N+1,1:self.N+1] += h**2*walker_counts[1:self.N+1,1:self.N+1]*0.25 # not sure why i need this but it helps
            global_counts[position[0], position[1]] += 1  # 
            
            # for varience calculation
            global_counts_squared[1:self.N+1,1:self.N+1] += (walker_counts[1:self.N+1,1:self.N+1])**2
            global_counts_squared[position[0], position[1]] += 1 # 1 squared is still 1
    
            

        # VAREINECE PART
        global_counts[1:self.N+1,1:self.N+1]*h**2 # h**2 needs to be after
        self.Greens_function = global_counts/(walk_numb)
        Varience_step = global_counts_squared/walk_numb
        Varience = Varience_step-self.Greens_function**2
        # print('varience is:', np.sum(Varience))
        # print('error is:', np.sqrt(np.sum(Varience)))
        print('the average walk length is', step_conter/walk_numb)
        return self.Greens_function, Varience # maybe this works??

    def function_val(self, walk_numb, point_index, function):
        Y = np.linspace(self.interval[0], self.interval[1], self.N)
        Y_outer_product = np.outer(Y, np.ones(len(Y)))
        
        
        # grid of function values and including the boundry conditions 
        potential_grid = self.grid_array*1
        potential_grid[1:self.N+1,1:self.N+1] = function(np.transpose(Y_outer_product), Y_outer_product)
        
        Green_array, varience_array = self.walker_vesion2(walk_numb, point_index)
       # test_array = np.sum(potential_grid[1:self.N+1,1:self.N+1]*Green_array[1:self.N+1,1:self.N+1])
        function_at_point = np.sum(Green_array*potential_grid)
        Varience_combined_with_potential = varience_array * potential_grid**2
        uncertainty_in_ans = np.sqrt(np.sum(Varience_combined_with_potential))
        print('uncertainty is ans:', uncertainty_in_ans)
        self.troubleshoot(function_at_point, potential_grid) #struggling to figure this all out
        
        return function_at_point

    def troubleshoot(self, result, potential_grid):
        laplace_contribution = self.Greens_function[self.boundry_index]*potential_grid[self.boundry_index]
        laplace_contribution = np.sum(laplace_contribution)


        possion_contribution = self.Greens_function[1:self.N+1,1:self.N+1]*potential_grid[1:self.N+1,1:self.N+1]
        possion_contribution = np.sum(possion_contribution)

        real_ans = self.grid_array[self.start_index[0], self.start_index[1]]
        real_ans_laplace_diff = real_ans-laplace_contribution

        ratio = real_ans_laplace_diff/possion_contribution
        print('\n---------')
        print('ratio for', self.start_index, 'is:', ratio)
        print('----------')
        return
    
def function(x, y): # this one is just for testing
    return 0

def function_a(x, y):
    return 10**x

def function_b(x, y):
    return y/10

def function_c(x,y):
    center = np.array([5,5])
    r = np.sqrt((x-center[0])**2 + (y-center[1])**2)
    return np.exp(-2000*r)


# varience seems to small!
# feel like im off by about 4


N_val = 9
initial_value = 5
bounbry_value = 1
interval = np.array([0, 10])

point_of_interest = np.array([1, 6]) # now as spacial coordinate




a = gridthing(N_val, initial_value, bounbry_value, interval)
#%%
# need to be carful not to boundry conditions to the corners!
# a.grid_array[1:N_val+1,0]=-1
# a.grid_array[1:N_val+1,N_val+1]=-1

# a.grid_array[0, 1:N_val+1] = 1
# a.grid_array[N_val+1, 1:N_val+1] = 1

a.intergrator(function_c)
# print('better_val:', a.grid_array[point_of_interest[0]+1, point_of_interest[1]+1])
a.plot()


c = a.function_val(100000, point_of_interest, function_c)
# c = a.function_val(1000, np.array([2,7]), function)
# c = a.function_val(1000, np.array([5,2]), function)
# c = a.function_val(1000, np.array([2,2]), function)
# print('walker method 2:', c)



#%%  TASK 3 don't run unless needed
# 
# walk_number = 1000
# Greens_array1,_ = np.flip(a.walker_vesion2(walk_number, [5, 5]), axis=1)
# Greens_array2,_ = np.flip(a.walker_vesion2(walk_number, [2.5, 2.5]), axis=1)
# Greens_array3,_ = np.flip(a.walker_vesion2(walk_number, [0.1, 2.5]), axis=1)
# Greens_array4,_ = np.flip(a.walker_vesion2(walk_number, [0.1, 0.1]), axis=1)

# #%% this break is so that i dont need to runn the greens functions every time


# fig1, ([ax11, ax12], [ax13, ax14]) = plt.subplots(2, 2, figsize=(10, 10))
# Interval_List = [0, 10, 0, 10]
# ax11.imshow(Greens_array1, extent=Interval_List)
# ax12.imshow(Greens_array2, extent=Interval_List)
# ax13.imshow(Greens_array3, extent=Interval_List)
# ax14.imshow(Greens_array4, extent=Interval_List)

# ax11.set(title='(a) [5,5]')
# ax12.set(title='(b) [2.5, 2.5]')
# ax13.set(title='(c) [0.1, 2.5]', 
#          xlabel='cm',
#          ylabel='cm')
# ax14.set(title='(d) [0.1, 0.1]')



# fig1.colorbar(ax11.imshow(Greens_array1, extent=Interval_List))
# fig1.colorbar(ax12.imshow(Greens_array2, extent=Interval_List))
# fig1.colorbar(ax13.imshow(Greens_array3, extent=Interval_List))
# fig1.colorbar(ax14.imshow(Greens_array4, extent=Interval_List))

# #%% Task 4

# ex4_Greens1 = a.function_val(walk_number, [5, 5], function)
# ex4_Greens2 = a.function_val(walk_number, [2.5, 2.5], function)
# ex4_Greens3 = a.function_val(walk_number, [0.1, 2.5], function)
# ex4_Greens4 = a.function_val(walk_number, [0.1, 0.1], function)
'''
time to cut losses me thinks?
just look at uncertainty next day and other stuffs
'''