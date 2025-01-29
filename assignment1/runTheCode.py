# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 15:15:59 2025

@author: Ben
used to run the code for assignment 1
"""
import numpy as np
from IPython import get_ipython
ip = get_ipython()
ip.run_cell("!mpiexec -n 6 python piCalculation.py")


print(np.pi, 'from numpy')
