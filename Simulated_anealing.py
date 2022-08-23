
import numpy as np 
from threading import Thread
import sys 

def check_domains(x, min_values, max_values, d):
    for i in range(d):
        if x[i] > max_values[i]:
            x[i] = max_values[i]
            
        if x[i] < min_values[i]:
            x[i] = min_values[i]
            
    return x

def simulatedAnealing(func, search_space, initial_point):
    """
    func - function to minimize
    search_space : The domain of search [[min value, max_value],,....,d]
    initial_point: Initial point to strat from (size of the state space) \in R^d [x_1,...,x_d]
    """
    d = len(initial_point)
    state = np.asarray(initial_point)
    min_values = []
    max_values = []
    
    for i in search_space:
        min_values.append(i[0])
        max_values.append(i[1])
        
    best_solution_found = sys.maxsize
    best_point = None
    T = 10
    val  = func(state)
    sigma = 1.
    alpha = 0.8
    accepted_solution_fraction =1. 
    
    while T> 0.0001:
        if T < 0.01 :
            sigma *=0.5
        iteration = 0
        while accepted_solution_fraction > 0.2 and iteration < 10:
            accepted_solution = 0
            iteration += 1
            for _ in range(100):
                # Apply random pertubations to the state x = x+del_x
                x = state + np.random.normal(loc = 0.0, scale = sigma, size=(d))
                
                x = check_domains(x, min_values, max_values, d)
        
                # Evaluation 
                val_temp = func(x)
                del_E = val_temp - func(state)
                # print(del_E)
                if min(1.0, np.exp(-del_E/T)) >= np.random.random():
                    state = x
                    accepted_solution += 1
                    val = val_temp
                    
                if val < best_solution_found:
                    best_solution_found = val 
                    best_point = state 
                    
            accepted_solution_fraction = accepted_solution/100
            
        # print(accepted_solution_fraction)
        # print("Best function value:", best_solution_found, "at:", best_point)
        T  = alpha * T
        # print(T)
    return best_point , best_solution_found




        
        
            
            
            
            


