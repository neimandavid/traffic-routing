#!/usr/bin/env python

import os
import sys
import optparse
import random
from numpy import inf
import numpy as np
#import time
import matplotlib.pyplot as plt
from datetime import datetime

import pickle #To save/load training data

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

import runnerQueueSplit18NN

#In case we want to pause a run and continue later, set these to false
reset = True
resetNN = reset
resetTrainingData = reset

nEpochs = 10
nDaggers = 100

agents = dict()
optimizers = dict()

actions = [0, 1]
learning_rate = 0.0005
batch_size = 1

nLossesBeforeReset = 100
losses = dict()
daggertimes = dict()

loss_fn = torch.nn.MSELoss() #torch.nn.CrossEntropyLoss(weight=torch.Tensor([1, 100]))


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

class TrafficLoader(Dataset):
    def __init__(self, datafile, light):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.datafile = datafile
        self.light = light
        with open(datafile, 'rb') as handle:
            temp = pickle.load(handle) #TODO: This is going to reload the pickle file for each light - this is slow
        self.dataset = temp[light]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        datapt = self.dataset[idx]
        print(datapt)
        sample = {'input': datapt[0], 'output': datapt[1]}

        return sample

# #NOT WORKING; not sure what's going wrong
# def main(sumoconfig):
#     trafficdataset = dict()
#     dataloader = dict()
#     with open("trainingdata/trainingdata_" + sys.argv[1] + ".pickle", 'rb') as handle:
#         trainingdata = pickle.load(handle)
#     for light in trainingdata:
#         trafficdataset[light] = TrafficLoader("trainingdata/trainingdata_" + sys.argv[1] + ".pickle", light)
#         dataloader[light] = DataLoader(trafficdataset, batch_size = batch_size, shuffle=True, num_workers=0)

#         agents[light] = Net(26, 2, 512) #Net(26, 2, 512)
#         optimizers[light] = torch.optim.Adam(agents[light].parameters(), lr=learning_rate)
#         MODEL_FILES[light] = 'models/imitate_'+light+'.model' # Once your program successfully trains a network, this file will be written
#         if not resetNN:
#             try:
#                 agents[light].load(MODEL_FILES[light])
#             except:
#                 print("Warning: Model not found, starting fresh")
#         losses[light] = []

#     for epochnum in range(nEpochs):
#         print("Going through training data, epoch " + str(epochnum))

#         for light in trainingdata:
#             print(light)
#             avgloss = 0
#             nlosses = 0
            
#             for databatch in dataloader[light]:
                
#                 outputNN = agents[light](databatch['input']) # Output from NN
#                 # Find the best action to take from actions based on the output of the NN
#                 #actionNumber = outputNN.argmax(1).item() # Get best action number
#                 actionNumber = outputNN.argmax().item() # Get best action number
#                 actionNN = actions[actionNumber] # Get best action

#                 target = torch.tensor(databatch['target']) # Target from expert
#                 loss = loss_fn(outputNN, target) # calculate loss between network action and expert action (Surtrac action)

#                 avgloss += float(loss.item()) # record loss for printing
#                 nlosses += 1

#                 if nlosses % nLossesBeforeReset == 0:
#                     avgloss /= nLossesBeforeReset
#                     log('Iteration: {}   Loss: {}'.format(nlosses,avgloss))
#                     avgloss = 0
#                 loss.backward() # perform backprop to collect gradients
#                 optimizers[light].step() # perform one step of update with optimizer
#                 optimizers[light].zero_grad() # reset accumulated gradients to 0
            
#             agents[light].save(MODEL_FILES[light])



def mainold(sumoconfig):

    if resetTrainingData:
        try:
            print("Archiving old training data. TODO archive models also.")
            os.rename("trainingdata/trainingdata_" + sys.argv[1] + ".pickle", "trainingdata/Archive/trainingdata_" + sys.argv[1] + str(datetime.now()) + ".pickle")
        except FileNotFoundError:
            print("Nothing to archive, this is fine")
            pass

    #DAgger loop
    for daggernum in range(nDaggers):

        #Get new training data
        #IMPORTANT: Make sure runnerQueueSplit18NN is set to testNN=True, testDumbtrac=True, resetTrainingData=False, appendTrainingData=True
        runnerQueueSplit18NN.main(sys.argv[1], 0, False)

        #Load current dataset
        print("Loading training data")
        with open("trainingdata/trainingdata_" + sys.argv[1] + ".pickle", 'rb') as handle:
            trainingdata = pickle.load(handle) #List of 2-elt tuples (in, out) = ([in1, in2, ...], out) indexed by light

        #Set up initial neural nets on first iteration
        if daggernum == 0:
            #Do NN setup
            for light in trainingdata:
                agents[light] = Net(26, 1, 512) #Net(26, 2, 512)
                optimizers[light] = torch.optim.Adam(agents[light].parameters(), lr=learning_rate)
                MODEL_FILES[light] = 'models/imitate_'+light+'.model' # Once your program successfully trains a network, this file will be written
                if not resetNN:
                    try:
                        agents[light].load(MODEL_FILES[light])
                    except:
                        print("Warning: Model not found, starting fresh")
                losses[light] = []
                daggertimes[light] = []


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

                    target = torch.tensor([float(data[1])]) # Target from expert
                    loss = loss_fn(outputNN, target) # calculate loss between network action and expert action (Surtrac action)

                    avgloss += float(loss.item()) # record loss for printing
                    nlosses += 1

                    if nlosses % nLossesBeforeReset == 0:
                        avgloss /= nLossesBeforeReset
                        losses[light].append(avgloss)
                        log('Iteration: {}   Loss: {}'.format(nlosses,avgloss))
                        avgloss = 0
                    loss.backward() # perform backprop to collect gradients
                    optimizers[light].step() # perform one step of update with optimizer
                    optimizers[light].zero_grad() # reset accumulated gradients to 0
                
                agents[light].save(MODEL_FILES[light])
                plt.figure()
                plt.plot(losses[light])
                for daggertime in daggertimes[light]:
                    plt.axvline(x=daggertime, color='k', linestyle='--')
                plt.xlabel("Sets of " + str(nLossesBeforeReset) + " Points")
                plt.ylabel("Average Loss")
                plt.title("Losses, light=" + str(light))
                #plt.show() #NOTE: Blocks code execution until you close the plot
                plt.savefig("Plots/Losses, light=" + str(light)+".png")
                plt.close()

        for light in trainingdata:
            daggertimes[light].append(len(losses[light]))
        

# this is the main entry point of this script
if __name__ == "__main__":
    mainold(sys.argv[1])
