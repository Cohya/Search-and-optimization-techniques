import matplotlib.pyplot as plt  
from scipy.stats import truncnorm 
import time 
import tensorflow as tf 
import numpy as np 
from GA_threads import Genetic_algorithm

def func(x):

    y = (x[0])**2 + (x[1])**2 + (x[2])**2
    return  -y  

if __name__ =='__main__':
    # ga = Genetic_algorithm(parameters=['discrete', 'discrete', 'continouse'],
    #                       search_range = [[i- 50 for i in range(100)],[i - 50 for i in range(100)], [-100,100]] )
    
    t0 = time.time()
    ga = Genetic_algorithm()
    pop, val, best_res, best_solution, all_results = ga.optimize(parameters_types = ['continouse','continouse', 'continouse'],
                                                                 search_range = [[-5,5],[-100,100], [-100,100]],
                                                                 fitness_fucntion=func , population_size = 100, 
                                                                 operation = 'maximization', max_iteration= 600,
                                                                 verbose = False)
    # print("res:", val)#, best_res, best_solution)
    print("Time:", time.time() - t0)
    plt.figure()
    plt.plot(all_results)
    plt.show()
    
    # print("all results:", all_results)
            