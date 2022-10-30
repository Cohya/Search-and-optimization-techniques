
import numpy as np 
from multiprocessing import Process, Queue, Lock
import multiprocessing
import pickle 
import os 
import sys 
import matplotlib.pyplot as plt  
from scipy.stats import truncnorm 
import time 

"""With mutex 183 sec for simple function x1**2 + x2**2 + x3**2, without 150 sec"""
mutex = Lock()
class Worker():
    def __init__(self,func,
                 q_to_worker = [],
                 q_to_global = [],
                 max_generations = 10,
                 i_d = 34534):
        
        self.i_d = i_d
        self.q_to_worker = q_to_worker
        self.q_to_global = q_to_global
        self.max_generations = max_generations
        self.func = func
        
        
    def run(self):
        
        while True:
            delivery = self.q_to_worker.get()
            chromosome, step = delivery
            
            if step > self.max_generations:
                print("Process %d Finished!" % self.i_d)
                mutex.acquire()
                self.q_to_global.put(None)
                mutex.release()
                return 
            
            val = self.func(chromosome)
            
            delivery = [chromosome, val, self.i_d]
            mutex.acquire()
            self.q_to_global.put(delivery)
            mutex.release()
            
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
    
    def create_mutation(self,population, eps = 0.2, std = 1.0):
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
        
        return chromosome
                
            
           
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
            

        else:

            elists = list(old_pop[:amount_of_elits])
        # eps should decrease with time steps 
        probs = vals
        new_population = list(elists) 
        mutations = []

        for i in range(amount_of_mutation): 
            mutant = self.create_mutation(population,eps = eps, std = std)
            
            mutations.append(mutant)

        # Create crossover
        crossovers = self.create_cross_over(population, probs, amount_of_crossover)
        
        
        # concatante 
        new_population += list(mutations) 
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
            
            
        cross_over_list = []
 
        for i in range(amount_of_crossover):
            # time.sleep(3)
            indexs = np.random.choice(n, size = 2, p = probs, replace = False)

            chrom1 = list(population[indexs[0]])
            chrom2 = list(population[indexs[1]])
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
                    
            cross_over_list.append(cross_i)
            
        return cross_over_list
                    
            
                
        
        
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
    
    def manager_worker(self):
        population = self.create_population(population_size = self.population_size)
        step = 0
        eps = 1.0
        std = 2.0
        self.all_results = []
        ids = []
        self.std_0 = std
        self.eps_0 = eps
        vals = [None for i in range(self.population_size)]
        # print(population,"Original")
        while True:
            self.last_population = population.copy()
            self.last_vals = vals.copy()
            if step > 1:
                # old_best_res = float(self.best_result)
                population = self.update_population(population,vals,ids, 
                                                    eps = self.udate_eps(eps,step), 
                                                    std = self.update_std(std,step))

                
            # Sending the workers job 
            mutex.acquire()
            for i in range(self.population_size):
                delivery = [population[i], step]
                self.q_to_worker.put(delivery)
            mutex.release()
            

            population = []
            vals = []
            ids = []
            # waiting until all workers finished
            counter = 0
            
            while True:
                mutex.acquire()
                check =self.q_to_global.qsize()
                mutex.release()
                if check ==  self.population_size :
                    break
                counter += 1
                 
            
            amount = self.q_to_global.qsize()
            while amount  != 0: 
                mutex.acquire()
                delivery = self.q_to_global.get()
                mutex.release()
                if step > self.max_iteration or delivery is None:
                    # print(step, delivery)
                    with open('results.pickle', 'wb') as file:
                        pickle.dump((self.last_population,self.last_vals,  self.best_result,
                                     self.best_chromosome,self.all_results), file)
                    return  
                
                chromosome, val, i_d = delivery
                # print("recieved:", chromosome, val, i_d )
                population.append(chromosome)
                vals.append(val)
                ids.append(i_d)
                amount -= 1
            
            step += 1
            
            if step % 50 == 0 and  self.verbose == True:
                print("Iteration:", step, "Best result:", self.best_result, "Best chromosome:", self.best_chromosome)
                
    def optimize(self, parameters_types, search_range, fitness_fucntion, max_iteration = 2, population_size = 3,
                 elits = 0.2, mutation = 0.2, operation = 'maximization',
                  verbose = False):
        """
        parameters = a list of types, can be discrete or continue (gene tpyes)
        search_range = is a list of a list
                        is the parameter is a continue type, then the list will be consist of 
                        [min value, max balue] and if it is discreate then it is the set of options
    

        """
        self.verbose = verbose
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
        
        max_num_of_proccesors = int(multiprocessing.cpu_count()/2 - 1)
        if max_num_of_proccesors < population_size:
            print("Sorry but the number of processors exceeds the number of available processors!")
            print("Try with maximum processors of %d or lower!" % max_num_of_proccesors)

        self.q_to_worker = Queue(maxsize = population_size)  
        self.q_to_global = Queue(maxsize= 1000)
        # Starting manager processor!!
        p_global = Process(target = self.manager_worker, args = ()) 
        p_global.start() 
        
        workers = []
        for i in range(population_size):
            worker = Worker(func= fitness_fucntion,
                            q_to_worker = self.q_to_worker,
                            q_to_global = self.q_to_global,
                            max_generations = max_iteration,
                            i_d = i)
            
            
            workers.append(worker)
        
        # Activate the workers
        for i in range(population_size):
            worker = workers[i]
            p = Process(target = worker.run , args = ())
            p.start()
            print("Process %d tarted" % worker.i_d)
      
        for worker in workers:
            p.join()
            
            
        p_global.join()
        with open('results.pickle', 'rb') as file:
            (self.last_population, self.last_vals,  self.best_result,
                self.best_chromosome, self.all_results) = pickle.load(file)
            
        os.remove('results.pickle')
        return self.last_population, self.last_vals, self.best_result, self.best_chromosome, self.all_results
        
           

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
            
    