# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 14:56:29 2025

@author: Ben

assignment 1
calculating pi 
advanced computing
"""

from mpi4py import MPI


comm = MPI.COMM_WORLD # begining paralel computing

# number of ranks
nproc = comm.Get_size()
# print(nproc)
nworkers = nproc - 1

N=1000000
Delta = 1.0/N

Intergral = 0.0

def integrand(x):
    return 4.0/(1.0+x**2)

samples_per_rank = N//(nproc-1) # split between workers
samples_for_rank0 = N%(nproc-1) # leftovers for the leader

if comm.Get_rank()==0:
    print('i am rank 0')
    if samples_for_rank0>0:
        for i in range(0, samples_for_rank0):
            x_value = (i+0.5)*Delta
            integral_part = integrand(x_value) * Delta
            Intergral += integral_part    
    for j in range (1, nproc):
        Integral_parts = comm.recv(source=j)
        Intergral += Integral_parts
    print(Intergral)
#gets stuck as im not sending anythin yet
else:
    rank = comm.Get_rank()
#    print('i am rank', rank)
    Rank_contribution = 0.0
    sample_range_start = samples_for_rank0 + (rank-1)*samples_per_rank
    sample_range_end = samples_for_rank0 + rank * samples_per_rank
    for i in range(sample_range_start, sample_range_end):
        x_value = (i+0.5) * Delta
        integral_part = integrand(x_value) * Delta
        Rank_contribution += integral_part 
    comm.send(Rank_contribution, dest=0)