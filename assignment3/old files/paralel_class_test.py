# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 16:57:41 2025

@author: Ben
"""
import numpy as np
from mpi4py import MPI

class MontysCarloss():
    
    def __init__(self, a):
        self.a = a
        
    def __add__(self, other_object):
        self.addition = self.a + other_object.a
        self.mean = self.addition/2
        return MontysCarloss(self.addition)

    def __str__(self):
        return f'({self.a}, {self.mean})'

def paking_fraction(dimentions, numb_of_samples):
    
    # test_coords is the random coordinates in dimentional space, but bounded within a box -1 to 1 in each direction
    test_coords = np.random.uniform(-1, 1, size=(dimentions*numb_of_samples))
    test_coords = np.resize(test_coords, (dimentions, numb_of_samples))
    # is used to count the nuber of points within the radius
    # I could do this a lot faster if I just used a np.where, should implement later
    counter = 0.0
    radius = 1.0
    for i in range (numb_of_samples):
        if np.sum((test_coords[i])**2) < radius:
            counter +=1
    fraction = counter/numb_of_samples
    return fraction

    
comm = MPI.COMM_WORLD
numb_of_ranks = comm.Get_size()

if comm.Get_rank()==0:
    test = MontysCarloss(1)
    comm.send(test, dest=1)
        
    
if comm.Get_rank()==(numb_of_ranks-1):
    test2 = MontysCarloss(4)
    recive = comm.recv(source=(comm.Get_rank()-1))
    print(test2+recive)
    
else:
    recive = comm.recv(source=(comm.Get_rank()-1))
    test = MontysCarloss(2)
    comm.send(test+recive, dest=1)    

