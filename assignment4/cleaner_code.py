# -*- coding: utf-8 -*-
"""
Created on Wed May 14 12:25:00 2025

@author: Ben
cleaning code
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


class GridThing():
    '''
    How to use:
        to start: when class is called it sets up the array that is used for calculations
        to add more complicated boundry condition can change values in the
        grid_array
        
        intergrator()
        solves the equation over all of the spesifyed space (deterministic solve)
        
        this can then be plotted using plot()
        
    To Calculate at spesific point:
        initallise (if not already)
        use greens_calc() to find the greens function for a given point
        
        Use function_val() with the calculated greens functinon to find a value 
        for the equation at a given point
    '''

    def __init__(self, n_val, start_val, boundry_val, interval):
        '''
        This sets up an array for solving a grid of size n_valxn_val with a boundries 
        around that which can be used to add boundry condisions
        
        Start value is the initial value for all possitions on the grid
        value for this isn't particularly important
        
        Boundry value is the fixed value at the boundry of the area that 
        is being intergrated over
        
        Interval is the  real space that the grid takes up
        '''
        self.n_val = n_val
        self.boundry_val = boundry_val
        self.interval = interval
        # makeing up the grid
        grid_array = np.zeros((n_val+2, n_val+2))
        n_grid = np.ones((n_val, n_val)) * start_val
        boundrys = np.ones(n_val) * boundry_val
        grid_array[1:n_val+1,1:n_val+1] = n_grid
        grid_array[1:n_val+1,0], grid_array[1:n_val+1,n_val+1] = boundrys, boundrys
        grid_array[0][1:n_val+1], grid_array[n_val+1][1:n_val+1] = boundrys, boundrys
        self.grid_array = grid_array
        self.start_copy = self.grid_array*1
        self.boundry_index = np.where(grid_array==boundry_val)
        self.h_val = (self.interval[1]-self.interval[0])/self.n_val


    def intergrator(self, function):
        '''
        Currently using the Guaus-Seidel relaxation also 
        a line comented out which does the over-relaxation to be made
        
        updates self.grid_array to have the values for the solved intergral
        
        Also prints out the number or iterations it takes and also
        the smallest largest change in values from the previous iteration

        '''
        def gaus_seidel():
            # equation is split up to make pylint happy
            part1a = self.grid_array[y_counter+1][x_counter]
            part1b = self.grid_array[y_counter-1][x_counter]
            part1c = self.grid_array[y_counter][x_counter+1]
            part1d = self.grid_array[y_counter][x_counter-1]
            part1 = part1a + part1b + part1c + part1d

            part2 = self.h_val**2* function(x_array[x_counter-1],
                                            y_array[y_counter-1])

            self.grid_array[y_counter][x_counter] = 0.25*(part1+part2)
            ## overrelaxed_version
            # part1 = 0.25*omega*(part1+part2)
            # part2 = (1-omega)*grid_array[y_counter][x_counter]
            # grid_array[y_counter][x_counter] = part1+part2

        x_array = np.linspace(self.interval[0], self.interval[1], self.n_val)
        y_array = x_array * 1
        change = 1.0
        counter=1
        # over relaxed equation for omega from notes.
        omega = 2/(1+np.sin(np.pi/self.n_val))

        while change > 1e-4/self.n_val**2:
            previous_grid = self.grid_array*1
            for y_counter in range (1, self.n_val+1):
                for x_counter in range (1, self.n_val+1):
                    gaus_seidel()

            if counter>10: # this is just to try and avoid divide by 0 issue wth this
                part1 = previous_grid[1:self.n_val+1,1:self.n_val+1]
                part2 = self.grid_array[1:self.n_val+1,1:self.n_val+1]
                change = np.max(np.abs((part1-part2)))

            counter +=1
        print('largest change from previous iteration', change)
        print('number of iterations', counter)


    def plot(self):
        '''
        Plots the solved intergral in 3D
        '''
        # sets up the axes
        ax0 = plt.axes(projection='3d')
        ax0.set_xlim3d(self.interval[0], self.interval[1])
        ax0.set_ylim3d(self.interval[0], self.interval[1])
        # insureing that the plot is over the interval
        x_array = np.linspace(self.interval[0], self.interval[1], self.n_val)
        y_array = x_array * 1
        x_array, y_array = np.meshgrid(x_array, y_array)
        n_grid = self.grid_array[1:self.n_val+1,1:self.n_val+1]
        # plotting the values
        ax0.plot_surface(x_array,y_array, n_grid, cmap=cm.coolwarm)

    def coord_to_grid_index(self, coordinat):
        '''
        used to switch a coordinate 
        '''
        start_index = (coordinat-self.interval[0])//self.h_val
        start_index = (start_index+1).astype(int) # the plus 1 to be index of grid_array
        start_index = [start_index[1], start_index[0]] # flipping the order
        return start_index


    def greens_calc(self, walk_numb, start_value):
        '''
        uses random walks to create a greens function array and varience array
        from a spesifyed starting point on the grid.
        
        walk_numb is the number of random walks to complete (interger)
        start_value is the starting coordinats of the random walker (numpy_array)
        
        returns the greens function and its varience at each point
        '''

         # grid spacing
        print('h_val value', self.h_val) # was using for troublesh_valooting
        correction_val = self.h_val**2*0.25
        # input is coordinats in the form (x,y) need to also switch order to
        # get into indexing format

        start_index = self.coord_to_grid_index(start_value)

        # sumations of places the random walk has gone
        global_counts = self.grid_array*0
        # going to be needed to calculate the varience
        global_counts_squared = self.grid_array*0

        step_conter = 0 # used to see on average how many steps were taken per walk
        for _ in range (0, walk_numb):
            # current position and array to save the walkers path
            position = start_index*1
            walker_counts = self.grid_array*0
            # walks stops when position is outside the n_valxN grid
            while position[0] in range(1, self.n_val+1) and position[1] in range(1, self.n_val+1):
                # choses direction at random
                direction = np.random.choice([0, 1, 2, 3],
                                              p=[0.25, 0.25, 0.25, 0.25],
                                              size=1)

                # takes a step in a direction and is recorded
                if direction==1:
                    position = [position[0]+1, position[1]] # +1 in y direction
                elif direction==2:
                    position = [position[0], position[1]+1] # +1 in x direction
                elif direction==3:
                    position = [position[0]-1, position[1]] # -1 in y direction
                else:
                    position = [position[0], position[1]-1] # -1 in x direction

                walker_counts[position[0], position[1]] += 1 # adding a step to the walk
                step_conter += 1

            # when a walk has reached the boundry update the global arrays
            # for greens function

            # in_grid_part = walker_counts[1:self.n_val+1,1:self.n_val+1]
            global_counts += walker_counts
            # global_counts[position[0], position[1]] += 1

            # updating the array for varience stuffs
            global_counts_squared[1:self.n_val+1,1:self.n_val+1] += (in_grid_part*correction_val)**2
            # global_counts_squared[position[0], position[1]] += 1 # 1 squared is still 1

        # All walks are now finished
        # Finding greens function at sites and at boundrys
        greens_function = global_counts/(walk_numb)
        # Calculating varience arrays
        varience_step = global_counts_squared/walk_numb
        varience = (varience_step-greens_function**2)
        return greens_function, varience


    def function_val(self, greens_function, greens_varience, function):
        '''
        calcuulates a value of a function at a certain point using the
        greens function
        
        greens_function is numpy array containing values for greens function
        
        greens_varience is numpy array containing variences for greens funciton
        
        function is the function that the poisson equation is equal to for the 
        equation that is being solved
        
        returns the value for the function at the point spesifyed when
        calculating the greens function
        '''
        y_array = np.linspace(self.interval[0], self.interval[1], self.n_val)
        y_outer_product = np.outer(y_array, np.ones(len(y_array)))
        x_outer_product = np.transpose(y_outer_product)

        # grid of function values and including the boundry conditions
        potential_grid = self.grid_array*1
        potential_grid[1:self.n_val+1,1:self.n_val+1] = function(x_outer_product,
                                                                 y_outer_product)


        function_at_point = np.sum(greens_function*potential_grid)
        # not 100% if this step is correct
        varience_combined_with_potential = greens_varience * potential_grid**2
        self.temp = greens_function*potential_grid
        uncertainty_in_ans = np.sqrt(np.sum(varience_combined_with_potential))
        print(function_at_point, '+/-', uncertainty_in_ans)
        return function_at_point



#%% out of class functions

def function0(x_val, y_val): # this one is just for testing
    '''
    function for laplace equation
    '''
    return 10*x_val+0*y_val

def function_a(x_val, y_val):
    '''
    function for exersise 4-5
    '''
    return 10


def function_b(x_val, y_val):
    '''
    function for exersise 4-5
    '''
    return y_val/10e-2+0*x_val


def function_c(x_val,y_val):
    '''
    function for exersise 4-5
    '''
    
    center = np.array([5,5])
    r_val = np.sqrt((x_val-center[0])**2 + (y_val-center[1])**2)
    return np.exp(-2000*r_val)


def boundrys_b(grid_class, n_val):
    '''
    
    '''
    grid_class.grid_array[1:n_val+1,0]=-1
    grid_class.grid_array[1:n_val+1,n_val+1]=-1

    grid_class.grid_array[0, 1:n_val+1] = 1
    grid_class.grid_array[n_val+1, 1:n_val+1] = 1


def boundrys_c(grid_class, n_val):
    grid_class.grid_array[1:n_val+1,0]= 2 # left
    grid_class.grid_array[1:n_val+1,n_val+1]=-4 # right

    grid_class.grid_array[0, 1:n_val+1] = 2 #top
    grid_class.grid_array[n_val+1, 1:n_val+1] = 0 # bottem

#%% 















       