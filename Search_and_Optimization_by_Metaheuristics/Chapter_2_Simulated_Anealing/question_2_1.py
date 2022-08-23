import matplotlib.pyplot as plt 

import numpy as np 
from Simulated_anealing import simulatedAnealing


def Ackley(x, dims = 5):
    a = 20
    b = 0.2
    c = np.pi * 2
    
    
    
    sum1 = -b * np.sqrt(np.mean(np.square(x)))
    right = -a * np.exp(sum1)
    
    sum2 = np.mean(np.cos(c * x))
    left = -np.exp(sum2) + a + np.exp(1)
    
    return right + left 


### Search and optimization by methuristic book example (2.1)
all_values = []# np.empty(shape = (10,1000))
for i in range(10):
    print(i)
    point, val, best_so_far, exit_reason  = simulatedAnealing(func = Ackley,
                                  search_space = [[-10,10]]*5, initial_point = np.random.rand(5), 
                                  initial_temperature = 100, max_iteration = 100000)
    all_values.append(best_so_far)
    print(exit_reason)
    
for b in all_values:
    plt.plot(b)