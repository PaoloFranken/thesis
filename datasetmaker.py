import numpy as np
import pandas as pd

# Open the Fashion MNIST data set which should be downloaded online before running this script
fmnist0 = np.genfromtxt('fashion-mnist.csv', delimiter=',')

# Select the relevant classes
sandal = fmnist0[fmnist0[:,0]==5]
sneaker = fmnist0[fmnist0[:,0]==7]
bag = fmnist0[fmnist0[:,0]==8]
ankleboot = fmnist0[fmnist0[:,0]==9]

# Select 400 random data points per class
sandal = sandal[np.random.choice(sandal.shape[0], 400, replace=False), :]
sandal[:,0] = 0
sneaker = sneaker[np.random.choice(sneaker.shape[0], 400, replace=False), :]
sneaker[:,0] = 1
bag = bag[np.random.choice(bag.shape[0], 400, replace=False), :]
bag[:,0] = 2
ankleboot = ankleboot[np.random.choice(ankleboot.shape[0], 400, replace=False), :]
ankleboot[:,0] = 3

# Merge all data points into one array 
fmnist = np.append(sandal,sneaker)
fmnist = np.append(fmnist, bag)
fmnist = np.append(fmnist, ankleboot)

fmnist = np.reshape(fmnist,(1600,785))

# Save new data set
np.savetxt("fmnist.csv", fmnist, delimiter=",")