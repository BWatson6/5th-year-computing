# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 11:23:49 2025

@author: Ben
"""

import numpy as np


def MontyCarloish(function, interval, numb_of_samples, dimentions):
    
    random_inputs = np.random.uniform(interval[0], interval[1], 
                                      size=(numb_of_samples*dimentions))
    
    random_inputs = np.resize(random_inputs, (numb_of_samples, dimentions))
    
    sampled_function = function(random_inputs)
    
#    Intergral = (interval[1]-interval[0])/numb_of_samples * np.sum(sampled_function)
    Intergral = np.sum(sampled_function)/numb_of_samples
    return Intergral*2**dimentions # area/volume whatever of round thing




def FillFration(trial_coordinates):
    radius = 1
    point_dist_from_origin = np.sum(trial_coordinates**2, axis=1)
    points_in_circle_index = np.where(point_dist_from_origin<radius)
    return len(points_in_circle_index[0])
    
def originDistance(trial_coordinates):
       point_dist_from_origin = np.sqrt(np.sum(trial_coordinates**2, axis=1))
       return point_dist_from_origin
  
    
test = MontyCarloish(FillFration, [-1, 1], 2000000, 2)

