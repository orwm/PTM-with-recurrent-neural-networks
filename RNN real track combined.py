import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import tensorflow as tf

#load data
fileloc = './Track_1.csv'
# filename = fileloc.split('/')[2]
d = pd.read_csv(fileloc,header=None)

#Set the framerate and number of steps
framerate = 50
M_steps = 15

x = np.array(d[0])*(2e-6)/30 #convert to m
y = np.array(d[1])*(2e-6)/30 #convert to m

t = np.linspace(0,len(x)-1,len(x))/framerate

#calculate the displacements wrt the position at t=0.
disps = np.sqrt( np.power(x-x[0],2) + np.power(y-y[0],2))

#Load model
model_path = os.path.abspath("RNN_1N_n15.h5")
low_model_path = os.path.abspath("RNN_21N_n15.h5")

model = tf.keras.models.load_model(model_path)
low_model = tf.keras.models.load_model(low_model_path)

#Create array that defines the value of H for each output neuron of the 21 N model
fixed_values = np.arange(0, 1.05, 0.05)  # Example values (0, 0.1, 0.2, ..., 1)

#loop through the data and estimate the hurst exponent in every window of 15 points
h = []
ht = []

a = int(np.ceil(M_steps/2))
b = M_steps - a + 1

for i in tqdm(range(a,len(disps)-b)):
    #segment the current window
    inx = disps[(i-a):(i+b)]
    #normalise displacements
    inx = (inx-np.amin(inx))/(np.amax(inx)-np.amin(inx))
    #apply differencing to get the step sizes
    inx = np.array([(inx[1:]-inx[0:-1])])
    #Predict H with 1N model
    test = model.predict(inx, verbose=0)
    test = test[0][0]
    
    #Decide which model to use based on both their outputs
    if test > 0.2:
        H = test
    else:
        test_ind = low_model.predict(inx, verbose = 0)
        test_l = fixed_values[np.argmax(test_ind, axis=1)]
        test_l = test_l[0]
        
        if test_l < 0.2:
            H = test_l
            
        else:
            H = test
    h.append(H)
    ht.append(t[i])

#plot displacements and H
plt.figure()
plt.subplot(211)
plt.plot(t,disps)
plt.ylabel('Disp.')
plt.xlim([t[0],t[-1]])
plt.subplot(212)
plt.plot(ht,h)
plt.plot([t[0],t[-1]],[0.5,0.5],'r-',lw=0.4)
plt.ylabel('H')
plt.xlabel('t')
plt.xlim(t[0],t[-1])
plt.ylim(0,1)
plt.tight_layout()