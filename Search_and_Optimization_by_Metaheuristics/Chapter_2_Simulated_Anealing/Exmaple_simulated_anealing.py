
import numpy as np 
from Simulated_anealing import simulatedAnealing

def easom_func(x):
    val  = -np.cos(x[0]) * np.cos(x[1]) * np.exp(-(x[0]- np.pi)**2 - (x[1]-np.pi)**2)
    return val 


point, val, _ ,_= simulatedAnealing(func = easom_func,
                              search_space = [[-10,10],[-10,10]], initial_point = [-0.5,0.5], 
                              max_iteration= 1000000,
                              initial_temperature=1)


### Search and optimization by methuristic book example (2.1)
