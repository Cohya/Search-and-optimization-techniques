
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

def simulatedAnealing(func, search_space, initial_point, initial_temperature = 10,
                      max_iteration = 1000):
    """
    func - function to minimize
    search_space : The domain of search [[min value, max_value],,....,d]
    initial_point: Initial point to strat from (size of the state space) \in R^d [x_1,...,x_d]
    initial_temperature -> The initial temperature value 
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
    T = initial_temperature
    val  = func(state)
    sigma = 1.
    alpha = 0.95
    accepted_solution_fraction =1. 
    total_iteration = 0
    best_so_far = []
    best_point = state
    while T> 1e-7 and total_iteration <= max_iteration :
        state = best_point
        if T < 0.1 :
            # sigma *=0.99
            sigma =min(2,T *100)
        iteration = 0
        
        while accepted_solution_fraction > 0.2 and iteration < 10:
            accepted_solution = 0
            iteration += 1
            
            for _ in range(100):
                total_iteration += 1
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
                    
                if  val < best_solution_found:
                    best_solution_found = val
                    best_point = state 
                
                if total_iteration > max_iteration:
                    exit_reason  ="max_iteration"
                    return best_point , best_solution_found,  best_so_far, exit_reason 
                else:
                    best_so_far.append( best_solution_found)
                
            accepted_solution_fraction = accepted_solution/100#iteration

            
            
            # print(accepted_solution_fraction)
        # print("Best function value:", best_solution_found, "at:", best_point)
        T  = max(alpha * T,1e-8)
        # print(T,"sd")
    exit_reason  = "T is lower than " + str(1e-7)
    print(sigma)
    return best_point , best_solution_found, best_so_far, exit_reason


def simulatedAnealingSimpler(func, search_space, initial_point, initial_temperature = 10,
                      max_iteration = 1000):
    """
    func - function to minimize
    search_space : The domain of search [[min value, max_value],,....,d]
    initial_point: Initial point to strat from (size of the state space) \in R^d [x_1,...,x_d]
    initial_temperature -> The initial temperature value 
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
    T = initial_temperature
    val  = func(state)
    alpha = 0.98 # best prams 0.85-0.95

    total_iteration = 0
    best_so_far = []
    best_point = state
    while T> 1e-7 and total_iteration <= max_iteration :
        state = best_point
        total_iteration += 1
        # Apply random pertubations to the state x = x+del_x
        x = state + np.random.normal(loc = 0.0, scale = T, size=(d))
        x = check_domains(x, min_values, max_values, d)
        
        # Evaluation 
        val_temp = func(x)
        del_E = val_temp - func(state)
        # print(del_E)
        if min(1.0, np.exp(-del_E/T)) >= np.random.random():
            state = x
            val = val_temp
            
        if  val < best_solution_found:
            best_solution_found = val
            best_point = state 
        
        if total_iteration > max_iteration:
            exit_reason  ="max_iteration"
            return best_point , best_solution_found,  best_so_far, exit_reason 
        else:
            best_so_far.append( best_solution_found)

        T  = max(alpha * T,1e-8)
    exit_reason  = "T is lower than " + str(1e-7)
    return best_point , best_solution_found, best_so_far, exit_reason

# def fuc(x):
#     return x**2

# point,_,_,_=simulatedAnealingSimpler(fuc, search_space=[[-10,10]],
#                          initial_point=[2])
        
        
            
            
            
            


