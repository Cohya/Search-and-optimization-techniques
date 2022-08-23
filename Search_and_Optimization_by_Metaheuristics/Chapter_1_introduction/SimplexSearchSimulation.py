from SimplexSearch import SimplexSearch
import numpy as np 
from PlotSimulation import SimulationOverContoufAndScatter, SimulationOver3DplotAndScatter,SimulationOverContoufAndScatterAndLine

def costi(x, y):
    return x**2 + y**2
# def costFnc(X, expand = True):
#     if expand:
#         X = np.expand_dims(X, axis = 1)
#     res = np.matmul(np.transpose(X),X)
#     return res

def costFnc(X, expand = True):
    x = X[0]
    y =X[1]
    return x**2 + y**2

# def costFnc(X, expand = True):
#     x = X[0]
#     y = X[1]
#     res = BealeFunc(x,y)
#     return res

# def costi(x, y):
#     res =  BealeFunc(x, y)
#     return res

# def BealeFunc(x, y):
#     return (1.5 - x + x* y)**2 + (2.25 - x +x*y**2)**2 + (2.625 - x + x*y**3)**2

ss = SimplexSearch()
res2 = ss.optimize(costFnc, 2,2, [-3,-3], [3,3], collect_results= True, maxIteration= 50)

allRes = ss.allSolution_per_iteration

x = []
y = []
z = [] # results 
for res in allRes:
    xTemp = []
    yTemp = []
    zTemp = []
    # print(res)
    for j in res[0]:
        xTemp.append(j[0])
        yTemp.append(j[1])
    
    for j in res[1]:
        zTemp.append(j)
        
    x.append(xTemp)
    y.append(yTemp)
    z.append(zTemp)
    
    
x = np.transpose(np.array(x))
y = np.transpose(np.array(y))
z = np.transpose(np.array(z))



N = 500
xs = np.linspace(-3, 3,N)
ys = np.linspace(-3,3, N)

def contourFunc(X,Y):
    return np.exp(X*Y)

def Rosenbrock_fnc2D(x1,x2):
    res = 100*(x2 - x1*x1)** 2 + (1-x1)**2
    return res

def EggholderFunc(x, y):
    res = -(y + 47)*np.sin(np.sqrt(np.abs(x/2 + (y+47)))) - x*np.sin(np.sqrt(np.abs(x - (y+47))))
    return res

# x,y = np.meshgrid(xs, ys)
# Z = Rosenbrock_fnc2D(x, y)

# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# surf = ax.plot_surface(x, y, Z,
#                        linewidth=0, antialiased=False)

# In your Algorithms I should return the:
    # P1 = the vector which gives the best results 
    # p2 = the value of the cost function 
    # p3 = a matrix of all generation and dims = (number of individuals at each iteration * Dimenxtion size) * number of genearion
    # p4 = a matrix of all results at each iteration / generation , 
    # dimension = number of individual at each iteration * cost fucntion value(1D) * nuber of iteration 
    
    

s = SimulationOverContoufAndScatterAndLine(xScatter = x,
                                    yScatter = y,zScatter= z,
                                    contourfFunc = costi, xRangeForContour = xs,
                                    yRangeForContour = ys, 
                                    include_min_value=True )

# 
s.simulate(cmapp = 'hot', colori = 'w', record = True)
    # 
    
    


            