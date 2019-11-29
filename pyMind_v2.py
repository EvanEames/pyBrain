import numpy as np
import random
import math
import matplotlib.pyplot as plt
import sys
np.set_printoptions(threshold=sys.maxsize)

n = 2000 #Number of neurons
timeSteps = 4 #Number of time steps for each training step
trainingSteps = 2 #Number of traning steps
toActivate = 10 #Number of activation nodes
inputNodes1 = np.random.choice(n,toActivate,replace=False) #Number of activation neurons within the larger neuron cluster
inputNodes2 = np.random.choice(n,toActivate,replace=False) #To compare with another random activation
inputNodes3 = np.random.choice(n,toActivate,replace=False) #To compare with another random activation
attenuation = 0.5 #Attenuation factor for the activations after each time step (1 = no attenuation, 0.5 = attenuate by half each step)
alpha = 2 #Axon training factor: How much you want to adjust each axon per time step (bigger = more adjusting)
actionPotential = 4

xcoords = np.random.rand(n) - 0.5
ycoords = np.random.rand(n) - 0.5
axons = np.zeros((n,n))

#Randomly distribute initial axon strengths
for i in range (0,n):
	for j in range (0,n):
		if (i != j):
			axons[i][j] = random.random()

#Train the neurons on various different input node activations
for train in range (0,trainingSteps):
	#Activate the inital activation nodes
	activations = np.zeros(n)
	if (train == 0):
		activations[inputNodes1] = 1
	else:
		activations[inputNodes2] = 1
	plt.subplot(5,5,train*5+1)
	sc = plt.scatter(xcoords, ycoords, s=activations*50+1, c=activations*1, alpha=0.5, vmin=0, vmax=2)
	tmpActivations = activations

	#Iterative time steps
	for i in range(0,timeSteps):
		activations[np.where(activations > 1)] = 1
		activations = np.dot(activations,axons)
		activations*=attenuation
		activations[np.where(activations < actionPotential)] = 0
		#Print them out
		plt.subplot(5,5,train*5+2+i)
		sc = plt.scatter(xcoords, ycoords, s=activations*50+1, c=activations*1, alpha=0.5, vmin=0, vmax=2)
	#Now we adjust the axons based on the signals that passed through them
	(xx,yy) = np.meshgrid(tmpActivations,tmpActivations)
	toAdjust = np.ones((n,n))
	toAdjust[np.where(xx*yy != 0)] = alpha
	axons = axons*toAdjust

#Test how the neurons react after being trained on both
for test in range (0,trainingSteps):
	#Activate the inital activation nodes
	activations = np.zeros(n)
	if (test == 0):
		activations[inputNodes1] = 1
	else:
		activations[inputNodes2] = 1
	plt.subplot(5,5,10+test*5+1)
	sc = plt.scatter(xcoords, ycoords, s=activations*50+1, c=activations, alpha=0.5, vmin=0, vmax=2)

	#Iterative time steps
	for i in range(0,timeSteps):
		activations[np.where(activations > 1)] = 1
		activations = np.dot(activations,axons)
		activations*=attenuation
		activations[np.where(activations < actionPotential)] = 0
		plt.subplot(5,5,10+test*5+2+i)
		sc = plt.scatter(xcoords, ycoords, s=activations*50+1, c=activations, alpha=0.5, vmin=0, vmax=2)


#Activate the inital activation nodes
activations = np.zeros(n)
activations[inputNodes3] = 1
plt.subplot(5,5,20+1)
sc = plt.scatter(xcoords, ycoords, s=activations*50+1, c=activations, alpha=0.5, vmin=0, vmax=2)

#Iterative time steps
for i in range(0,timeSteps):
	activations[np.where(activations > 1)] = 1
	activations = np.dot(activations,axons)
	activations*=attenuation
	activations[np.where(activations < actionPotential)] = 0
	plt.subplot(5,5,20+2+i)
	sc = plt.scatter(xcoords, ycoords, s=activations*50+1, c=activations, alpha=0.5, vmin=0, vmax=2)

plt.show()
