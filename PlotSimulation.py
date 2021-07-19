
import numpy as np 
import matplotlib.pyplot as plt 
from  matplotlib.animation import FuncAnimation
from matplotlib import animation

#This function is to show a simulation of scatter points over contourf 

class SimulationOverContoufAndScatter(object):
    
    
    def __init__(self,xScatter, yScatter, zScatter, contourfFunc, xRangeForContour, yRangeForContour,include_min_value = False):
        
        # xScatter - is the location of the points in the axes direction [x(t=0), x(t=1), x(t=2)] where x(t=i) \in R^n,
        #   n - number of points per iteration , size = (numberOfPoints, numberOfIteration)
        
        # yScatter - same as xScatter just for Y , size = (numberOfPoints, numberOfIteration)
        
        # zScatter - The values of each point at t!! , size = (numberOfPoints, numberOfIteration)
        
        # contourfFunc - The function of the contourf
        # xRangeForContour, yRangeForContour - the ranges which we want to plot of the contour function 
        # Remember that xRangeForContour and yRangeForContour are numpay array  
        self.x = xScatter
        self.y = yScatter
        if len(zScatter) == 0:
            self.z = contourfFunc(self.x, self.y)
        else:
            self.z = zScatter
            
        # self.z =  zScatter

        self.xRangeForContour = xRangeForContour
        self.yRangeForContour = yRangeForContour
        self.contourfFunc = contourfFunc
        self.include_min_value = include_min_value
        

    def find_min(self,z):
        
        n = len(z)
        min_val = np.inf
        index = None
        for i in range(n):
            
            val = z[i]
            
            if val < min_val:
                min_val = val 
                index = i 
                
        return min_val, index
            
            
    def update3(self, frame):
        array = np.stack((self.x[:,frame], self.y[:,frame]))
        # array = array.reshape(len(self.x[:,frame]), 2)
        self.sc.set_offsets(array.T)
        if self.include_min_value:
            zmin, index = self.find_min(self.z[:, frame])

            string_for_title = ("Min Value: " +  str("%.4f,"% zmin) +  "  x1: " + str("%.4f," % self.x[index,frame]) +
            "  x2: " +  str("%.4f" %self.y[index, frame]))
            plt.title(string_for_title)
            
        return self.sc,
                
    def simulate(self, record = False):
        self.fig = plt.figure()
        # plt.title("df")
        
        self.ax = plt.subplot(1,1,1)
        # self.text = self.ax.text(0,0,0)
        plt.rcParams["font.family"] = "Times New Roman"
        self.ax.set_xlim([min(self.xRangeForContour), max(self.xRangeForContour)])
        self.ax.set_ylim([min(self.yRangeForContour), max(self.yRangeForContour)])
        
        
        X,Y = np.meshgrid(self.xRangeForContour, self.yRangeForContour)
        Z = self.contourfFunc(X,Y)
        cs  = self.ax.contourf(X,Y,Z, cmap ="hot")
        plt.colorbar(cs)
    
        self.sc = self.ax.scatter(self.x[:,0] , self.y[:,0], color = 'r')
        
        self.ani = FuncAnimation(self.fig, self.update3, frames=np.arange(self.x.shape[1]),
                interval=200,
                blit = False ,repeat=True)

        plt.show()
        
        if record :
            self.ani.save('sdf.mp4', dpi = 150, fps = 30, writer='ffmpeg')
                
                
            
            

class SimulationOver3DplotAndScatter(object):
    
    
    def __init__(self,xScatter, yScatter, contourfFunc, xRangeForContour, yRangeForContour,include_min_value = False):
        
        # xScatter - is the location of the points in the axes direction [x(t=0), x(t=1), x(t=2)] where x(t=i) \in R^n,
        #   n - number of points per iteration , size = (numberOfPoints, numberOfIteration)
        
        # yScatter - same as xScatter just for Y , size = (numberOfPoints, numberOfIteration)
        
        # zScatter - The values of each point at t!! , size = (numberOfPoints, numberOfIteration)
        
        # contourfFunc - The function of the contourf
        # xRangeForContour, yRangeForContour - the ranges which we want to plot of the contour function 
        # Remember that xRangeForContour and yRangeForContour are numpay array  
        self.x = xScatter
        self.y = yScatter
        self.z = contourfFunc(self.x, self.y)
        self.xRangeForContour = xRangeForContour
        self.yRangeForContour = yRangeForContour
        self.contourfFunc = contourfFunc
        self.include_min_value = include_min_value
        

    def find_min(self,z):
        
        n = len(z)
        min_val = np.inf
        index = None
        for i in range(n):
            
            val = z[i]
            
            if val < min_val:
                min_val = val 
                index = i 
                
        return min_val, index
            
    def update3(self, frame):
        array = np.stack((self.x[:,frame], self.y[:,frame], self.z[:,frame]))

        self.sc._offsets3d = array 

        """For rotating"""
        # self.ax.view_init(azim=int(360/self.framess) * frame)
        if self.include_min_value:
            zmin, index = self.find_min(self.z[:, frame])

            string_for_title = ("Min Value: " +  str("%.4f,"% zmin) +  "  x1: " + str("%.4f," % self.x[index,frame]) +
            "  x2: " +  str("%.4f" %self.y[index, frame]))
            plt.title(string_for_title)
            
        return self.sc,
                
    def simulate(self, record = True):
        
        plt.style.use('dark_background')
        self.fig = plt.figure(1)
        # plt.title("df")
        
        self.ax = plt.subplot(111,projection='3d')
        
        # self.text = self.ax.text(0,0,0)
        plt.rcParams["font.family"] = "Times New Roman"
        self.ax.set_xlim([min(self.xRangeForContour), max(self.xRangeForContour)])
        self.ax.set_ylim([min(self.yRangeForContour), max(self.yRangeForContour)])
        # self.ax = p3.Axes3D(self.fig)
        self.ax.grid(False)
        
        X,Y = np.meshgrid(self.xRangeForContour, self.yRangeForContour)
        Z = self.contourfFunc(X,Y)
        cs  = self.ax.plot_surface(X,Y,Z, cmap = "viridis", zorder=-1, alpha = 0.5)# cmap ="cool",
        plt.colorbar(cs)

        self.sc = self.ax.scatter(self.x[:,0] , self.y[:,0],self.z[:,0] ,color = 'k',
                                  alpha = 1, s =30 ,linewidth=5, zorder=50)
        self.framess = len(np.arange(self.x.shape[1]))
        self.ani = FuncAnimation(self.fig, self.update3, frames= np.arange(self.x.shape[1]),
                interval=100,
                blit = False ,repeat=True)

        plt.show()
        
        if record :
            # writergif = animation.PillowWriter(fps=30)
            # self.ani.save('filename.gif',writer=writergif)
            self.ani.save('sdf.mp4', dpi = 150, fps = 30, writer='ffmpeg')
                     
        
        
class SimulationOverContoufAndScatterAndLine(object):
   
   
   def __init__(self,xScatter, yScatter, zScatter, contourfFunc, xRangeForContour, yRangeForContour,include_min_value = False):
       
       # xScatter - is the location of the points in the axes direction [x(t=0), x(t=1), x(t=2)] where x(t=i) \in R^n,
       #   n - number of points per iteration , size = (numberOfPoints, numberOfIteration)
       
       # yScatter - same as xScatter just for Y , size = (numberOfPoints, numberOfIteration)
       
       # zScatter - The values of each point at t!! , size = (numberOfPoints, numberOfIteration)
       
       # contourfFunc - The function of the contourf
       # xRangeForContour, yRangeForContour - the ranges which we want to plot of the contour function 
       # Remember that xRangeForContour and yRangeForContour are numpay array  
       self.x = xScatter
       self.y = yScatter
       if len(zScatter) == 0:
           self.z = contourfFunc(self.x, self.y)
       else:
           self.z = zScatter
           
       # self.z =  zScatter

       self.xRangeForContour = xRangeForContour
       self.yRangeForContour = yRangeForContour
       self.contourfFunc = contourfFunc
       self.include_min_value = include_min_value
       

   def find_min(self,z):
       
       n = len(z)
       min_val = np.inf
       index = None
       for i in range(n):
           
           val = z[i]
           
           if val < min_val:
               min_val = val 
               index = i 
               
       return min_val, index
           
           
   def update3(self, frame):
       array = np.stack((self.x[:,frame], self.y[:,frame]))
       
       x = self.x[:,frame]
       x = np.append(x, x[0])
       y = self.y[:,frame]
       y = np.append(y, y[0])
       # array = array.reshape(len(self.x[:,frame]), 2)
       self.sc.set_data(x, y)
       if self.include_min_value:
           zmin, index = self.find_min(self.z[:, frame])

           string_for_title = ("Min Value: " +  str("%.4f,"% zmin) +  "  x1: " + str("%.4f," % self.x[index,frame]) +
           "  x2: " +  str("%.4f" %self.y[index, frame]))
           plt.title(string_for_title)
           
       return self.sc,
               
   def simulate(self, record = False, cmapp = 'hot', colori = 'r'):
       self.fig = plt.figure()
       # plt.title("df")
       
       self.ax = plt.subplot(1,1,1)
       # self.text = self.ax.text(0,0,0)
       plt.rcParams["font.family"] = "Times New Roman"
       self.ax.set_xlim([min(self.xRangeForContour), max(self.xRangeForContour)])
       self.ax.set_ylim([min(self.yRangeForContour), max(self.yRangeForContour)])
       
       
       X,Y = np.meshgrid(self.xRangeForContour, self.yRangeForContour)
       Z = self.contourfFunc(X,Y)
       cs  = self.ax.contourf(X,Y,Z, cmap = cmapp)
       plt.xlabel("$x_1$")
       plt.ylabel("$x_2$")
       plt.colorbar(cs)
   
       self.sc, = self.ax.plot(self.x[:,0] , self.y[:,0],  '--s', color = colori )
       
       self.ani = FuncAnimation(self.fig, self.update3, frames=np.arange(self.x.shape[1]),
               interval=500,
               blit = False ,repeat=True)

       plt.show()
       
       if record :
           self.ani.save('sdf.mp4', dpi = 150, fps = 3, writer='ffmpeg')
                  