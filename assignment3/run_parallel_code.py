# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 17:15:07 2025

@author: Ben
to test paralell
"""


from IPython import get_ipython

# nice and easy yay
ip = get_ipython()
#ip.run_cell("!mpiexec -n 4 python paralel_task1_try.py")
# im not variencing right i think
ip.run_cell("!mpiexec -n 8 python assignment3_final.py")

#ip.run_cell("!mpiexec -n 6 python Task3.py")