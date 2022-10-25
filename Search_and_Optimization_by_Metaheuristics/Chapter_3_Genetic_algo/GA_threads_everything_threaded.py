
import numpy as np 
from threading import Thread, Lock
# import multiprocessing
import pickle 
import os 
import sys 
import matplotlib.pyplot as plt  
from scipy.stats import truncnorm 
import time 

"""With mutex 183 sec for simple function x1**2 + x2**2 + x3**2, without 150 sec"""
mutex = Lock()

            
class Genetic_algorithm():
    def __init__(self, initialization_function = None, options = {}):

        self.initialization = initialization_function
        
        self.options = options
        
        self.options['technique'] = self.options.get('technique','RoulleteWheel')
        
        
    def create_chromosome(self):
        chromosome = []
        for i in range(self.number_of_genes):
            type_i = self.parameters[i]
            
            if type_i == "discrete":
                gene = np.random.choice(self.search_range[i])
            
            else:
                # continuse
                minimum, maximum = self.search_range[i]
                
                gene = np.random.random() * (maximum - minimum) + minimum
                
            chromosome.append(gene)
            
        return chromosome
    
    def create_population(self, population_size):
        population = []
        for p in range(population_size):
            chromosome = self.create_chromosome()
            population.append(chromosome)
        
        return population

                
    def create_mutation_thread(self,population, eps = 0.2, std = 1.0):
        index= np.random.choice(self.population_size)
        chromosome = list(population[index])
        
        for i in range(self.number_of_genes):
            typ_i = self.parameters[i]
            domain = self.search_range[i]
            
            if typ_i == "discrete":
                if np.random.random()<eps:
                    chromosome[i] = np.random.choice(domain)
                else:
                    continue
            else: # for continues params 
                    
                min_value,max_value = domain
                mean = chromosome[i]
                mutant_value = np.random.normal(scale = std) + mean 
                
                mutant_value = min(max_value, mutant_value)
                mutant_value = max(min_value, mutant_value)
                chromosome[i] = mutant_value
        
        self.mutations.append(chromosome)    
        
    def update_population(self, population, vals,ids,eps = 0.2,std = 1.0):
        
        
        ### mutation 20%
        ## crossOver_60%
        ## Elits 20%
        
        amount_of_elits = int(self.population_size * self.elits)
        amount_of_mutation = int(self.population_size * self.mutation)
        amount_of_crossover = self.population_size - amount_of_elits - amount_of_mutation
        
        if amount_of_elits == 0:
            amount_of_elits = 1
            amount_of_mutation = 1
            amount_of_crossover = self.population_size - 2
        # orgenized from small to big 

        vals, population,ids = zip(*sorted(zip(vals,population,ids), key = lambda x: x[0]))
        vals = list(vals)
        population = list(population)
        ids = list(ids)
        old_pop = population.copy()
        #Update best result
        
        if self.operation == 'maximization':
            if vals[-1] > self.best_result:
                self.best_result = vals[-1]
                self.best_chromosome = population[-1]
        else:
            if vals[0] < self.best_result:
                self.best_result = vals[0]
                self.best_chromosome = population[0]

        self.all_results.append(self.best_result)
        # Collect Elits 
        # print(population, type(population), vals, type(vals))
        if self.operation == 'maximization':
            elists = list(old_pop[-amount_of_elits:])
            #probs = vals

        else:

            elists = list(old_pop[:amount_of_elits])
        probs = vals           
       #probs = [-value_i for value_i in vals]
        # eps should decrease with time steps 

        new_population = list(elists) 
        self.mutations = []
        
        all_threads = []
        for i in range(amount_of_mutation):
            t = Thread(target= self.create_mutation_thread, args=(population,eps, std))
            t.start()
            all_threads.append(t)
            
        for k in all_threads:
            k.join()
            
        # Create crossover
        self.create_cross_over(population, probs, amount_of_crossover)
        crossovers = self.cross_over_list.copy()
        
        # concatante 
        new_population += list(self.mutations) 
        new_population += list(crossovers)

        return new_population
    
    def create_cross_over(self, population, vals, amount_of_crossover):
        n = self.population_size
        sum_vals = np.sum(vals)
        probs = [val / sum_vals for val in vals] 
        
        if self.operation == 'maximization':
            pass
        else:
            probs1 = [1 - p for p in probs]
            sum_p = sum(probs1) 
            probs = [p / sum_p for p in probs1]
            
 
        if self.options['technique'] == 'RankingSelection':
            beta = 0 # can be selected from [0,2] # 0 be proporation to the rank 2 is for the opposite 
            
            if self.operation == 'maximization':
                probs_rank = np.arange(self.population_size) + 1
            else:
                probs_rank = np.arange(self.population_size,0, -1)
                
                
            probs = 1/self.population_size * (beta - 2* (beta -1) * (probs_rank  -1)/(self.population_size-1)) 
            
            
        self.cross_over_list = []
        t_list = []
        for i in range(amount_of_crossover):
            t = Thread(target = self.create_single_co, args=(population, n,probs))
            t.start()
            t_list.append(t)
        for t in t_list:
            t.join()
            
   
    def create_single_co(self,population, n, probs):
        indexs = np.random.choice(n, size = 2, p = probs, replace = False)
        chrom1 = population[indexs[0]]
        chrom2 = population[indexs[1]]
        chromosomes = [chrom1, chrom2]#chromosome1, chromosome2
        
        sum_p1_p2 = probs[indexs[0]] + probs[indexs[1]]
        p1 = probs[indexs[0]]/sum_p1_p2
        p2 = 1 - p1
        
        cross_i = []
        
        for g in range(self.number_of_genes):
            if self.parameters[g] == 'discrete':
                index = np.random.choice([0,1], size = 1, p = [p1,p2])[0]
    
                cross_i.append(chromosomes[index][g])
                
            else: # for continues
                Beta = 0.
                alpha = (1 + 2*Beta)*np.random.random() - Beta;
                # alpha = 0.1 + np.random.random()*(sensitivity-0.1)
                
                value_g = chromosomes[0][g] + alpha*(chromosomes[1][g] - chromosomes[0][g]) 
                cross_i.append(value_g)
                
        self.cross_over_list.append(cross_i)
            

        
    def update_std(self, std, step):
        # go down to 0.02 - in linear fashion 
        std = (0.02 - self.std_0) / self.max_iteration *step + self.std_0
        std = max(0.02, std)
        return std
        
    def udate_eps(self, eps,step):
        # goes linearly to 0.02
        eps = (0.02 - self.eps_0) / self.max_iteration *step + self.eps_0
        eps = max(0.02, eps)
        return eps
    
  
    def optimize(self, parameters_types, search_range, fitness_fucntion, max_iteration = 2, population_size = 3,
                 elits = 0.2, mutation = 0.2, operation = 'maximization',
                 verbose = False):
        """
        parameters = a list of types, can be discrete or continue (gene tpyes)
        search_range = is a list of a list
                        is the parameter is a continue type, then the list will be consist of 
                        [min value, max balue] and if it is discreate then it is the set of options
    

        """
        self.fitness_fucntion = fitness_fucntion
        self.parameters = parameters_types
        self.number_of_genes = len(parameters_types)
        self.search_range = search_range
        self.operation = operation
        
        if operation == 'maximization':
            self.best_result = -sys.maxsize
        else:
            # for minimization
            self.best_result = sys.maxsize
            
            
        self.elits = elits
        self.mutation = mutation
        
        self.population_size = population_size
        # print(population_size)
        self.max_iteration = max_iteration
        
        population = self.create_population(population_size = self.population_size)
        self.step = 1
        eps = 1.0
        std = 2.0
        self.all_results = []
        ids = []
        self.std_0 = std
        self.eps_0 = eps
        self.vals = [None for i in range(self.population_size)]
        
        while self.step <= self.max_iteration:
        # Activate the workers
            self.last_population = population.copy()
            # self.last_vals = self.vals.copy()
            if self.step > 2:
                # old_best_res = float(self.best_result)
                population = self.update_population(population,self.vals,ids, 
                                                    eps = self.udate_eps(eps,self.step), 
                                                    std = self.update_std(std,self.step))
                
            workers = []
            for i in range(population_size):
                data = population[i]
                worker = Thread(target = self.worker , args = (data,i))
                worker.start()
                workers.append(worker)
                # print("Thread %d tarted" % i)
                ids.append(i)
            for worker in workers:
                worker.join()
                
            if (self.step) % 50 == 0 and verbose == True:
                print("Iteration:", self.step, "Best result:", self.best_result, "Best chromosome:", self.best_chromosome)            
                # for i, j in zip(population, self.vals):
                #     print("Pop:", i, "value:", j)
            self.step += 1
      
        
        
        return self.last_population, self.vals, self.best_result, self.best_chromosome, self.all_results
        
    def worker(self, data, i):
        # i is for changing the coresponding location in the array which belong to the specific chromosom
        value = self.fitness_fucntion(data)
        
        mutex.acquire()
        self.vals[i] = value
        mutex.release()
           
def func(x):

    y = (x[0])**2 + (x[1])**2 + (x[2])**2
    return  -y  

if __name__ =='__main__':

    t0 = time.time()
    ga = Genetic_algorithm()
    options = {}
    options['technique'] = 'RankingSelection'
    ga = Genetic_algorithm(options=options)
    pop, val, best_res, best_solution, all_results = ga.optimize(parameters_types = ['continouse','continouse', 'continouse'],
                                                                 search_range = [[-5,5],[-100,100], [-100,100]],
                                                                 fitness_fucntion=func , population_size =50, 
                                                                 operation = 'maximization', max_iteration= 100,
                                                                 verbose = False)
    # print("res:", val)#, best_res, best_solution)
    print("Time:", time.time() - t0)
    print("Best Solution:", best_solution)
    plt.figure()
    plt.plot(all_results)
    plt.show()
    
    # print("all results:", all_results)
            
    