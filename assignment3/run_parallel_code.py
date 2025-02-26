# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 17:15:07 2025

@author: Ben
to test paralell
"""


from IPython import get_ipython

# nice and easy yay
ip = get_ipython()
ip.run_cell("!mpiexec -n 5 python paralel_task1_try.py")