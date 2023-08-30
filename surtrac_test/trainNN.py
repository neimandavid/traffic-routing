#!/usr/bin/env python

import os
import sys
import optparse
import random
from numpy import inf
import numpy as np
#import time
#import matplotlib.pyplot as plt

import pickle #To save/load training data

import torch
from torch import nn

resetNN = False
nEpochs = 10000

agents = dict()
optimizers = dict()

actions = [0, 1]
learning_rate = 0.0005

nLossesBeforeReset = 100


loss_fn = torch.nn.CrossEntropyLoss(weight=torch.Tensor([1, 50]))

#Neural net things
LOG_FILE = 'imitate.log' # Logs will be appended every time the code is run.
MODEL_FILES = dict()
def log(*args, **kwargs):
    if LOG_FILE:
        with open(LOG_FILE, 'a+') as f:
            print(*args, file=f, **kwargs)
    print(*args, **kwargs)

class Net(torch.nn.Module):
    def __init__(self, in_size, out_size, hidden_size):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            #Input: Assume 4 roads, 3 lanes each, store #stopped and #total on each. Also store current phase and duration, and really hope the phases and roads are in the same order
            #So 26 inputs. Phase, duration, L1astopped, L1atotal, ..., L4dstopped, L4dtotal

            nn.Linear(in_size, hidden_size), #Blindly copying an architecture
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, out_size)
        )
        
    def forward(self, x):
        out = self.linear_relu_stack(x)
        return out

    def save(self, file_path):
        torch.save(self.state_dict(), file_path)

    def load(self, file_path):
        self.load_state_dict(torch.load(file_path))

def main(sumoconfig):
    print("Loading training data")
    with open("trainingdata/trainingdata_" + sys.argv[1] + ".pickle", 'rb') as handle:
        trainingdata = pickle.load(handle) #List of 2-elt tuples (in, out) = ([in1, in2, ...], out) indexed by light

    #Do NN setup
    for light in trainingdata:
        agents[light] = Net(26, 2, 512)
        optimizers[light] = torch.optim.Adam(agents[light].parameters(), lr=learning_rate)
        MODEL_FILES[light] = 'models/imitate_'+light+'.model' # Once your program successfully trains a network, this file will be written
        if not resetNN:
            try:
                agents[light].load(MODEL_FILES[light])
            except:
                print("Warning: Model not found, starting fresh")
                


    #Train things
    for epochnum in range(nEpochs):
        print("Going through training data, epoch " + str(epochnum))

        for light in trainingdata:
            print(light)
            avgloss = 0
            nlosses = 0
            random.shuffle(trainingdata[light])
            
            for data in trainingdata[light]:
                
                outputNN = agents[light](data[0]) # Output from NN
                # Find the best action to take from actions based on the output of the NN
                #actionNumber = outputNN.argmax(1).item() # Get best action number
                actionNumber = outputNN.argmax().item() # Get best action number
                actionNN = actions[actionNumber] # Get best action

                #target = torch.tensor([actionSurtrac]) # Target from expert
                target = torch.tensor([data[1]]) # Target from expert
                #target = torch.tensor([lightphases[light]%2]) # Target from expert
                loss = loss_fn(outputNN, target) # calculate loss between network action and expert action (Surtrac action)

                avgloss += loss.item() # record loss for printing
                nlosses += 1

                if nlosses % nLossesBeforeReset == 0:
                    avgloss /= nLossesBeforeReset
                    log('Iteration: {}   Loss: {}'.format(nlosses,avgloss))
                    avgloss = 0
                loss.backward() # perform backprop to collect gradients
                optimizers[light].step() # perform one step of update with optimizer
                optimizers[light].zero_grad() # reset accumulated gradients to 0
            
            agents[light].save(MODEL_FILES[light])
        

# this is the main entry point of this script
if __name__ == "__main__":
    main(sys.argv[1])
