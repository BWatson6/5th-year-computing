# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 14:56:29 2025

@author: Ben
in discussion with Eamonn McHugh and Jack MacKenzie

assignment 1
calculating pi 
advanced computing

When ran returns a value of pi to 15 significant figures
"""

from mpi4py import MPI
import mpmath as mp

# begining paralel computing
comm = MPI.COMM_WORLD

# number of ranks
nproc = comm.Get_size()
# rank 0 will be used as leader
nworkers = nproc - 1

# number of samples
N=10000000
DELTA = 1.0/N


INTERGRAL = 0.0

def integrand(x_input):
    """
    A function to be intergrated to find pi

    Parameters
    ----------
    x_input : float
        input value for the function.

    Returns
    -------
    float
        numeric value for function for a given value of x.

    """
    return 4.0/(1.0+mp.fmul(x_input, x_input, exact=True))

# samples are divideed up between the nodes and remainder samples are sumed by the leader

# number of samples for each worker
samples_per_rank = N//(nproc-1)
# any remainder samples are calculated by the leader rank
samples_for_rank0 = N%(nproc-1)

# code for leader
if comm.Get_rank()==0:
    # for any remainder samples, the area for each are calculated and added to the intergral
    if samples_for_rank0>0:
        for i in range(0, samples_for_rank0):
            x_value = mp.fmul((i+0.5), DELTA, exact=True)
            integral_part = mp.fmul(integrand(x_value), DELTA, exact=True)
            INTERGRAL = mp.fadd(INTERGRAL, integral_part, exact=True)

    # the sums calculated on each of the other nodes are recived and added to the total intergral
    for j in range (1, nproc):
        Integral_parts = comm.recv(source=j)
        INTERGRAL = mp.fadd(INTERGRAL, Integral_parts, exact=True)
    mp.nprint(INTERGRAL, 16)

# code for workers
else:
    rank = comm.Get_rank()
    RANK_CONTRIBUTION = 0.0
    # range of values that the node will calculate for are defined by a start and end value
    sample_range_start = samples_for_rank0 + (rank-1)*samples_per_rank
    sample_range_end = samples_for_rank0 + rank * samples_per_rank

    # part of the intergral is calculated and sent to the leader node
    for i in range(sample_range_start, sample_range_end):
        x_value = mp.fmul((i+0.5), DELTA, exact=True)
        integral_part = mp.fmul(integrand(x_value), DELTA, exact=True)
#        RANK_CONTRIBUTION += integral_part
        RANK_CONTRIBUTION = mp.fadd(RANK_CONTRIBUTION, integral_part, exact=True)
    comm.send(RANK_CONTRIBUTION, dest=0)
