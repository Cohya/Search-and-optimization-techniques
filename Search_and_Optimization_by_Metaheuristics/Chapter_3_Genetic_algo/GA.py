
import numpy as np 
from threading import Thread

class Genetic_algorithm():
    def __init__(self, parameters, search_range ):
        """
        parameters = a list of types, can be discreate or continue (gene tpyes)
        search_range = is a list of a list
                        is the parameter is a continue type, then the list will be consist of 
                        [min value, max balue] and if it is discreate then it is the set of options
    

        """
        self.parameters = parameters
        self.number_of_genes = len(parameters)
        self.search_range = search_range
        
    
    def create_chromosome(self):
        chromosome = []
        for i in range(self.number_of_genes):
            type_i = self.parameters[i]
            
            if type_i is "discrete":
                gene = np.random.choice(self.search_range[i])
            
            else:
                # continuse
                minimum, maximum = self.search_range[i]
                
                gene = np.random.random() * (maximum - minimum) + minimum
                
            chromosome.append(gene)
            
        return chromosome
    
    def create_a_population(self, population_size):
        
        for p in range(population_size):
            
        