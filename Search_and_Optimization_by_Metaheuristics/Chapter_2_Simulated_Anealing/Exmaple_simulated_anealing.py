
import numpy as np 
from Simulated_anealing import simulatedAnealing

def easom_func(x):
    val  = -np.cos(x[0]) * np.cos(x[1]) * np.exp(-(x[0]- np.pi)**2 - (x[1]-np.pi)**2)
    return val 

def Ackley(x, dims = 5):
    a = 20
    b = 0.2
    c = np.pi * 2
    
    
    
    sum1 = -b * np.sqrt(np.mean(np.square(x)))
    right = -a * np.exp(sum1)
    
    sum2 = np.mean(np.cos(c * x))
    left = -np.exp(sum2) + a + np.exp(1)
    
    return right + left 

# point, val = simulatedAnealing(func = easom_func,
#                               search_space = [[-10,10],[-10,10]], initial_point = [-0.5,0.5])


### Search and optimization by methuristic book example (2.1)
point, val = simulatedAnealing(func = Ackley,
                              search_space = [[-10,10]]*5, initial_point = np.random.rand(5), 
                              initial_temperature = 100)
