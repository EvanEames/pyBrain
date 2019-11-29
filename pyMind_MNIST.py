import numpy as np
import random
import math
import matplotlib.pyplot as plt
import sys
from mlxtend.data import loadlocal_mnist
np.set_printoptions(threshold=sys.maxsize)


#===========================================================================================================
# Initialize Parameters
#===========================================================================================================

n = 2000 #Number of neurons
timeSteps = 4 #Number of time steps for each training step
attenuation = 0.5 #Attenuation factor for the activations after each time step (1 = no attenuation, 0.5 = attenuate by half each step)
alpha = 2 #Axon training factor: How much you want to adjust each axon per time step (bigger = more adjusting)
actionPotential = 4

#===========================================================================================================
# Read in MNIST data set
#===========================================================================================================

X, y = loadlocal_mnist(
        images_path='/home/evan/Desktop/ML_Stuff/PyMind/train-images.idx3-ubyte', 
        labels_path='/home/evan/Desktop/ML_Stuff/PyMind/train-labels.idx1-ubyte')


X.astype(float)
#X /= 255.

xcoords = np.random.rand(n) - 0.5
ycoords = np.random.rand(n) - 0.5
axons = np.zeros((n,n))

#Randomly distribute initial axon strengths
for i in range (0,n):
	for j in range (0,n):
		if (i != j):
			axons[i][j] = random.random()

#Train the neurons on various different input node activations
for train in range (0,np.shape(X)[0]):
	#Activate the inital activation nodes by inputting a digit
	activations = np.zeros(n)
	activations[0:np.shape(X)[1]] = X[train]/255.
	tmpActivations = activations

	#Now we adjust the axons based on the signals that passed through them
	(xx,yy) = np.meshgrid(tmpActivations,tmpActivations)
	toAdjust = np.ones((n,n))
	toAdjust[np.where(xx*yy != 0)] = alpha
	axons = axons*toAdjust

#Iterative time steps
for i in range(0,timeSteps):
	activations[np.where(activations > 1)] = 1
	activations = np.dot(activations,axons)
	activations*=attenuation
	activations[np.where(activations < actionPotential)] = 0
	#Print them out
	plt.subplot(5,5,train*5+2+i)
	sc = plt.scatter(xcoords, ycoords, s=activations*50+1, c=activations*1, alpha=0.5, vmin=0, vmax=2)

#plt.show()
