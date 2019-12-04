import numpy as np
import random
import math
import matplotlib.pyplot as plt
import sys
import imageio
import cv2
from mlxtend.data import loadlocal_mnist
np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(formatter={'float': lambda x: "{0:5.3f}".format(x)})

#===========================================================================================================
# Initialize Parameters
#===========================================================================================================

n = 784 #Number of neurons (equal to the number of MNIST pixels)
eta = 0.01 #Learning rate (see https://en.wikipedia.org/wiki/Generalized_Hebbian_Algorithm)
actionPotential = 4.3/(n-1)

#===========================================================================================================
# Read in MNIST data set
#===========================================================================================================

X, y = loadlocal_mnist(
        images_path='/home/evan/Desktop/ML_Stuff/PyMind/train-images.idx3-ubyte', 
        labels_path='/home/evan/Desktop/ML_Stuff/PyMind/train-labels.idx1-ubyte')

X.astype(float)
average_power = np.sum(X)/np.shape(X)[0]/255.

#===========================================================================================================
# Set up neurons and axons
#===========================================================================================================

xcoords = np.arange(0,784)%28 #np.random.rand(n) - 0.5
ycoords =  np.repeat(np.arange(0,28,1),28)[::-1] #np.random.rand(n) - 0.5
axons = np.zeros((n,n))

#Randomly distribute initial axon strengths
for i in range (0,n):
	for j in range (0,n):
		if (i != j):
			axons[i][j] = np.random.random()

#===========================================================================================================
# Train on MNIST images
#===========================================================================================================

#Train the neurons on various different input node activations
for train in range (0,1000):#np.shape(X)[0]):
	#Activate the inital activation nodes by inputting a digit
	activations = X[train]/255.
	#Now we adjust the axons based on the signals that passed through them
	#x = (activations*np.ones((n,n))).transpose()/(n-1) #How much is coming along each neuron along path ij
	#np.fill_diagonal(x,0)
	#y = np.sum(x*axons,axis=0) #How much is coming to each neuron j (the sum of all paths)
	#dw = eta*(y*x - (y**2)*axons) #Generalized Hebbian Algorithm
	diff = np.abs((activations*np.ones((n,n))).transpose()-activations)
	if (train%1000 == 0): eta/=2
	dw = eta*(np.outer(activations,activations) - diff*np.square(axons))
	np.fill_diagonal(dw,0)
	axons += dw
	#sc = plt.scatter(xcoords, ycoords, s=activations*50+1, c=activations*1, alpha=0.8, vmin=0, vmax=1)
	#max_dw = np.max(dw)
	#for i in range(0,n):
	#	for j in range(0,n):
	#		if (dw[i][j] == max_dw):
	#			plt.plot((xcoords[i],ycoords[i]), 'r--')
	#plt.show()
	print(train)
	#print(np.mean(dw))
	#print(np.max(dw))

#===========================================================================================================
# Test
#===========================================================================================================

vmax = 0.08**2 #max plotting value
howbig = 15

activations = X[-1265]/255.
activations *= average_power/np.sum(activations)
plt.subplot(2,4,1)
sc = plt.scatter(xcoords, ycoords, s=activations*50+1, c=activations*1, alpha=0.8, vmin=0, vmax=vmax)
activations = np.dot(activations/(n-1),axons) #The electric potential must be divided up by how many nodes it is sent to
activations = np.square(activations)
activations[np.where(activations < actionPotential)] = 0
plt.subplot(2,4,2)
sc = plt.scatter(xcoords, ycoords, s=activations*50*howbig+1, c=activations*1, alpha=0.8, vmin=0, vmax=vmax)

activations = X[-6]/255.
activations *= average_power/np.sum(activations)
plt.subplot(2,4,3)
sc = plt.scatter(xcoords, ycoords, s=activations*50+1, c=activations*1, alpha=0.8, vmin=0, vmax=vmax)
activations = np.dot(activations/(n-1),axons) #The electric potential must be divided up by how many nodes it is sent to
activations = np.square(activations)
activations[np.where(activations < actionPotential)] = 0
plt.subplot(2,4,4)
sc = plt.scatter(xcoords, ycoords, s=activations*50*howbig+1, c=activations*1, alpha=0.8, vmin=0, vmax=vmax)

activations = X[-7]/255.
activations *= average_power/np.sum(activations)
plt.subplot(2,4,5)
sc = plt.scatter(xcoords, ycoords, s=activations*50+1, c=activations*1, alpha=0.8, vmin=0, vmax=vmax)
activations = np.dot(activations/(n-1),axons) #The electric potential must be divided up by how many nodes it is sent to
activations = np.square(activations)
activations[np.where(activations < actionPotential)] = 0
plt.subplot(2,4,6)
sc = plt.scatter(xcoords, ycoords, s=activations*50*howbig+1, c=activations*1, alpha=0.8, vmin=0, vmax=vmax)


#Also read in a letter to cross check
letter = imageio.imread("symbol.png",as_gray=True)/255.
letter = cv2.resize(letter, (28,28), interpolation = cv2.INTER_AREA)
activations = np.ravel(1-letter)
activations *= average_power/np.sum(activations)
plt.subplot(2,4,7)
sc = plt.scatter(xcoords, ycoords, s=activations*50+1, c=activations*1, alpha=0.8, vmin=0, vmax=vmax)
activations = np.dot(activations/(n-1),axons) #The electric potential must be divided up by how many nodes it is sent to
activations = np.square(activations)
activations[np.where(activations < actionPotential)] = 0
plt.subplot(2,4,8)
sc = plt.scatter(xcoords, ycoords, s=activations*50*howbig+1, c=activations*1, alpha=0.8, vmin=0, vmax=vmax)

plt.show()

#1268 is tough
