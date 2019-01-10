import numpy as np
import random
import math
import matplotlib.pyplot as plt

n = 5000 #Number of neurons
timeSteps = 4 #Number of time steps for each training step
trainingSteps = 2 #Number of traning steps
coverage = 10 #Percentage of boxWidth that neurons can reach
toActivate = 50 #Number of activation nodes
inputNodes1 = np.random.choice(n,toActivate,replace=False) #Number of activation neurons within the larger neuron cluster
inputNodes2 = np.random.choice(n,toActivate,replace=False) #To compare with another random activation
inputNodes3 = np.random.choice(n,toActivate,replace=False) #To compare with another random activation
attenuation = 0.8 #Attenuation factor for each time step
alpha = 0.5 #Axon training factor: How much you want to adjust each axon per time step (1 <= alpha < 0, bigger = less adjusting)

xcoords = np.random.rand(n) - 0.5
ycoords = np.random.rand(n) - 0.5
axons = np.zeros((n,n))

#Randomly distribute initial axon strengths
for i in range (0,n):
	for j in range (0,i):
		dist = math.sqrt((xcoords[i] - xcoords[j])**2 + (ycoords[i] - ycoords[j])**2)
		if dist <= coverage/100. and i != j and axons[j][i] == 0:
			axons[i][j] = random.random()
convolve = np.array(axons)
convolve[np.where(convolve != 0)] = 1

#Train the neurons on various different input node activations
for train in range (0,trainingSteps):
	#Activate the inital activation nodes
	activations = np.zeros(n)
	sc = plt.scatter(xcoords, ycoords, s=activations*50+1, c=activations, alpha=0.5)
	if (train == 0): plt.title("Training activation 1, step = 0")
	if (train == 1): plt.title("Training activation 2, step = 0")
	plt.colorbar(sc)
	plt.show()
	if (train == 0):
		activations[inputNodes1] = 1
	else:
		activations[inputNodes2] = 1
	activations = activations*0.5#np.random.rand(n)
	sc = plt.scatter(xcoords, ycoords, s=activations*50+1, c=activations, alpha=0.5)
	if (train == 0): plt.title("Training activation 1, step = 1")
	if (train == 1): plt.title("Training activation 2, step = 1")
	plt.colorbar(sc)
	plt.show()

	#Iterative time steps
	for i in range(0,timeSteps):
		tmpActivations = activations
		activations = np.dot(activations,axons)
		activations = activations/np.max(activations)
		#Now we adjust the axons based on the signals that passed through them
		axons = (axons.T + ((1-alpha)/alpha)*tmpActivations).T * convolve
		tmpActivations[np.where(tmpActivations != 0)[0]] = alpha
		tmpActivations[np.where(tmpActivations == 0)[0]] = 1
		axons = (axons.T * tmpActivations).T
		#Print them out
		sc = plt.scatter(xcoords, ycoords, s=activations*50+1, c=activations, alpha=0.5)
		if (train == 0): title = "Training activation 1, step = " + str(2+i)
		if (train == 1): title = "Training activation 2, step = " + str(2+i)
		plt.title(title)
		plt.colorbar(sc)
		plt.show()

#Test how the neurons react after being trained on both
for test in range (0,trainingSteps):
	#Activate the inital activation nodes
	activations = np.zeros(n)
	sc = plt.scatter(xcoords, ycoords, s=activations*50+1, c=activations, alpha=0.5)
	if (test == 0): plt.title("Testing activation 1, step = 0")
	if (test == 1): plt.title("Testing activation 2, step = 0")
	plt.colorbar(sc)
	plt.show()
	if (test == 0):
		activations[inputNodes1] = 1
	else:
		activations[inputNodes2] = 1
	activations = activations*0.5#np.random.rand(n)
	sc = plt.scatter(xcoords, ycoords, s=activations*50+1, c=activations, alpha=0.5)
	if (test == 0): plt.title("Testing activation 1, step = 1")
	if (test == 1): plt.title("Testing activation 2, step = 1")
	plt.colorbar(sc)
	plt.show()

	#Iterative time steps
	for i in range(0,timeSteps):
		tmpActivation = activations
		activations = np.dot(activations,axons)
		activations = activations/np.max(activations)
		sc = plt.scatter(xcoords, ycoords, s=activations*50+1, c=activations, alpha=0.5)
		if (test == 0): title = "Testing activation 1, step = " + str(2+i)
		if (test == 1): title = "Testing activation 2, step = " + str(2+i)
		plt.title(title)
		plt.colorbar(sc)
		plt.show()


#Activate the inital activation nodes
activations = np.zeros(n)
sc = plt.scatter(xcoords, ycoords, s=activations*50+1, c=activations, alpha=0.5)
plt.title("Testing activation 3, step = 0")
plt.colorbar(sc)
plt.show()
activations[inputNodes3] = 1
activations = activations*0.5#np.random.rand(n)
sc = plt.scatter(xcoords, ycoords, s=activations*50+1, c=activations, alpha=0.5)
plt.title("Testing activation 3, step = 1")
plt.colorbar(sc)
plt.show()

#Iterative time steps
for i in range(0,timeSteps):
	tmpActivation = activations
	activations = np.dot(activations,axons)
	activations = activations/np.max(activations)
	sc = plt.scatter(xcoords, ycoords, s=activations*50+1, c=activations, alpha=0.5)
	title = "Testing activation 3, step = " + str(2+i)
	plt.title(title)
	plt.colorbar(sc)
	plt.show()
