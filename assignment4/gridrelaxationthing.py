# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 17:01:06 2025

@author: Ben
"""

import numpy as np
import matplotlib.pyplot as plt

from matplotlib import cm


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
        while change > 1e-5/self.N**2:         
            previous_grid = grid_array*1
            for x_counter in range (1, self.N+1):
                for y_counter in range (1, self.N+1):
                    part1 = grid_array[x_counter+1][y_counter]+grid_array[x_counter-1][y_counter]+grid_array[x_counter][y_counter+1]+grid_array[x_counter][y_counter-1]
                    part2 = h**2* function(X[x_counter-1], Y[y_counter-1])
                    # grid_array[x_counter][y_counter] = 0.25*(part1-part2)
                    
                    #overrelaxed_version
                    grid_array[x_counter][y_counter] = 0.25*omega*(part1-part2)+(1-omega)*grid_array[x_counter][y_counter]
                    
            change = np.max(np.abs((previous_grid[1:self.N+1,1:self.N+1]-grid_array[1:self.N+1,1:self.N+1])/previous_grid[1:self.N+1,1:self.N+1]))
            #print('change value', change)
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

    def walker(self, walk_numb, start_point, function_grid):
        grid_for_counts = self.grid_array*0
        h = (self.interval[1]-self.interval[0])/self.N
        function_counter = 0

        # location of walker at any time is defined by the variable position
        # the input is currently an index of the location of grid (unitless)
        # ngl I think my walker function is currently pretty fucked 
        
        for i in range(walk_numb):
            position = start_point
            while position[0] in range(1, self.N+1) and position[1] in range(1, self.N+1):
                function_counter += function_grid[position[0]-1, position[1]-1]  
                direction = np.random.choice([1, 2, 3, 4], 
                                             p=[0.25, 0.25, 0.25, 0.25], 
                                             size=(1))
                if direction==1:
                    position = [position[0]+1, position[1]]                    
                elif direction==2:
                    position = [position[0], position[1]+1]
                elif direction==3:
                    position = [position[0]-1, position[1]]
                else:
                    position = [position[0], position[1]-1]
                  
            grid_for_counts[position[0], position[1]] += 1
                
            
        distribution_list = grid_for_counts[self.boundry_index]
        # this is probability of a walker ariving at a certain point on the boundry
        Green_function = 1/walk_numb * distribution_list
        # this is probabitiy of ariving at certain points on the grid times h^2
        # it will be used for the greens grid function
        Green_grid_function = h**2/walk_numb * function_counter
        return Green_function, Green_grid_function # use boundry index to go to physical position on grid
    
    def Function_at_point(self, walk_numb, point, function):
        # using walkers to find greens function at a point and hace the value
        # of a function at a certain point.
        # x and y are the spacial values for the grid 
        X = np.linspace(self.interval[0], self.interval[1], self.N)
        X_outer_product = np.outer(X, np.ones(len(X)))
        grid_function_vals = function(X_outer_product, np.transpose(X_outer_product))
        
        #1. calculate the greens function at each point
        green_func, green_grid = self.walker(walk_numb, point, grid_function_vals)
        # boundry greens func mutiplyed by the value at the boundry
        point1 = np.sum(green_func*self.grid_array[self.boundry_index])
        point2 = green_grid
        #2 use along with values at boundry conditions to figure out the value
        #   at a certain point.
        
        #3. profit
        
        return point1+point2

def Funciton(x, y):
    return 0*x+0*y #x*2+y**2




N_val = 25
initial_value = 5
bounbry_value = 0.1
interval = [-10, 10]

point_of_interest = [10, 10]

# now for the trouble shooting :((


a = gridthing(N_val, initial_value, bounbry_value, interval)
a.grid_array[:,0]=2
#b = a.walker(1000, [2,2])
b = a.Function_at_point(5000, point_of_interest, Funciton)

# # a.grid_array[N_val][N_val]=2
a.intergrator(Funciton)
print('over-relaxed method:', a.grid_array[point_of_interest[0], point_of_interest[1]])
print('walker method:', b)
a.plot()

'''
am i comparing the right points from the different intergration methods?

for the over relaxed method I am taking the valu from the grid array wich is
of dimentions (N+2)x(N+2)

for the walker method where am i taking it from?

'''
