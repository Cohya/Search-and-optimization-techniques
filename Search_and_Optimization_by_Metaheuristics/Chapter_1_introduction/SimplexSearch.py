
import numpy as np 
import sys 



class SimplexSearch(object):
    def __init__(self):
        
        pass
        
    def optimize(self, cost_function, problemDims, numberOfParams, 
                 domainminValue, domainmaxValue, collect_results = False,maxIteration = 100):
        
        self.cost_function = cost_function
        self.numberOfParams = numberOfParams
        self.n =  problemDims 
        self.domainmaxValue = domainmaxValue
        self.domainminValue = domainminValue
        self.initializationTechnique = 'uniform'
        
        self.allSolution_per_iteration = []
        if self.initializationTechnique == 'uniform':
            
            points = np.random.uniform(low= self.domainminValue, 
                                   high= self.domainmaxValue, size = (self.n+1, self.n))
            
        alpha = 1
        beta = 2
        gamma = -0.5 
        delta = 0.5
        
        results = [0 for i in range(self.n+1)]
        best = sys.maxsize
        worst = 0
        for i in range(maxIteration):
            worst = -sys.maxsize
                
            for j in range(self.n + 1):
                p = points[j]
                
                res = cost_function(p)
                results[j] = res
                if res < best:
                    best = res
                    bestIndex = int(j) 
                    
                if res > worst:
                    worst = res
                    worstIndex = int(j) 
             
                    
            # calculating the centroid 
            count = 1
            for k in range(self.n + 1):
                 
                if k == worstIndex:
                    continue
                
                elif count == 1:
                    centroid = points[k]
                else:
                    centroid += points[k]
                count += 1
            
            centroid = centroid/self.n

            # Enter reflection point 
            xr = centroid + alpha*(centroid - points[worstIndex])

            
            resXr = self.cost_function(xr)
            if best< resXr and resXr < worst:
                points[worstIndex] = xr
                
            elif resXr < best:
                # Enter Expansion mode
                xe = centroid + beta*(centroid - points[worstIndex])
                resxe = self.cost_function(xe)
                
                if resxe < best :
                    points[worstIndex] = xe
                    
                else:

                    points[worstIndex] = xr
                    
            elif [resXr > results[i] for i in range(self.n+1) if i != worstIndex]:
                # enter contraction mode 
                Xc = centroid + gamma*(centroid - points[worstIndex])
                
                resXc = self.cost_function(Xc)
                
                if resXc < worst:
                    
                    points[worstIndex] = Xc
                    
                else:
                    # enter shrinking mode 
                    for i in range(self.n + 1):
                       if i != bestIndex:   
                           points[i] = points[bestIndex] + delta * (points[i] - points[bestIndex])
                        
                    
            
           
            if collect_results:
                
                resNumpy = []
                for i in results:
                    resNumpy.append(i)
                self.allSolution_per_iteration.append([np.array(points),resNumpy])
            print(points, "best:", bestIndex, "bestVal:", best)   
        return points[bestIndex], best
            
            
            
            
        



        
            
        
            
