#!/usr/bin/env python

import os
import sys
import optparse
import random
from numpy import inf
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

import sumolib #To query node/edge stuff about the network
import pickle #To save/load training data

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

#import runnerQueueSplit
import runnerQueueSplit27 as runnerQueueSplit #KEEP THIS UP TO DATE!!!
import intersectionGenerator
from importlib import reload
from Net import Net

import openpyxl #For writing training data to .xlsx files
import time

#From: https://www.geeksforgeeks.org/how-to-use-gpu-acceleration-in-pytorch/
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)} is available.")
else:
    print("No GPU available. Training will run on CPU.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

#In case we want to pause a run and continue later, set these to false
reset = False
resetNN = reset
resetTrainingData2 = reset
#UPDATE: Turns out appendTrainingData (there) gets updated automatically, as does noNNInMain
#Also, Surtrac network architecture works for FTPs as well
#So just make sure resetTrainingData=False, testDumbtrac and FTP are correct, and surtracFreq = 1ish (all in runnerQueueSplitWhatever)

crossEntropyLoss = True

#testSurtrac = True #Surtrac architecture works on FTP - just always setting this to true seems fine

nEpochs = 10
nDaggers = 100

agents = dict()
optimizers = dict()
dataloader = dict()

actions = [0, 1]
learning_rate = 0.00005
batch_size = 100

nLossesBeforeReset = 10000/batch_size
losses = dict()
epochlosses = dict()    
daggertimes = dict()

#Modified from: https://discuss.pytorch.org/t/hinge-loss-in-pytorch/86220
class HingeLoss(torch.nn.Module):

    def __init__(self):
        super(HingeLoss, self).__init__()
        self.relu = nn.ReLU()

    def forward(self, output, target):
        all_ones = torch.ones_like(target)
        #labels = 2 * target - all_ones #Turns 0 or 1 into -1 or 1
        losses = all_ones - torch.mul(output.squeeze(1), target)

        return torch.norm(self.relu(losses))

if crossEntropyLoss:
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.Tensor([100, 1])) #HingeLoss()#torch.nn.MSELoss()
else:
    loss_fn = torch.nn.MSELoss()


#Neural net things
LOG_FILE = 'imitate.log' # Logs will be appended every time the code is run.
MODEL_FILES = dict()
def log(*args, **kwargs):
    if LOG_FILE:
        with open(LOG_FILE, 'a+') as f:
            print(*args, file=f, **kwargs)
    print(*args, **kwargs)


#Based on https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
class TrafficDataset(Dataset):
    def __init__(self, datafile):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        with open(datafile, 'rb') as handle:
            temp = pickle.load(handle) #TODO: This is going to reload the pickle file for each light - this is slow
        self.dataset = temp["light"]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        datapt = self.dataset[idx]
        sample = {'input': datapt[0], 'target': datapt[1]}
        return sample

def readSumoCfg(sumocfg):
    netfile = ""
    roufile = ""
    with open(sumocfg, "r") as cfgfile:
        lines = cfgfile.readlines()
        for line in lines:
            if "net-file" in line:
                data = line.split('"')
                netfile = data[1]
            if "route-files" in line: #This is scary - probably breaks if there's many of them
                data = line.split('"')
                roufile = data[1]
    return (netfile, roufile)

def main(sumoconfigs):
    global daggertimes
    if resetNN:
        print("Archiving old models.")
        # lights = []
        # for sumoconfig in sumoconfigs:
        #     (netfile, routefile) = readSumoCfg(sumoconfig)
        #     for node in sumolib.xml.parse(netfile, ['junction']):
        #         if node.type == "traffic_light":
        #             lights.append(node.id)
        for light in ["light"]:# + lights:
            try:
                print("models/imitate_" + light + ".model")
                os.rename("models/imitate_" + light + ".model", "models/Archive/imitate_" + light + str(datetime.now()) + ".model")
            except FileNotFoundError:
                print("No model found for light " + light + ", this is fine")
        
    if resetTrainingData2:
        try:
            print("Archiving old training data.")
            os.rename("trainingdata/trainingdata_" + sys.argv[1] + ".pickle", "trainingdata/Archive/trainingdata_" + sys.argv[1] + str(datetime.now()) + ".pickle")
        except FileNotFoundError:
            print("Nothing to archive, this is fine")
            pass

    firstIter = True
    #DAgger loop
    while True: #for daggernum in range(nDaggers):

        #Get new training data
        if not(firstIter and not resetTrainingData2): #If first iteration and we already have training data, start by training on what we have
            print("Generating new training data")
            for sumoconfig in sumoconfigs:
                if sumoconfig == "IG":
                    #reload(intersectionGenerator)
                    intersectionGenerator.main()
                else:
                    reload(runnerQueueSplit)
                    runnerQueueSplit.main(sumoconfig, 0, False, False, True)
                    dumpTrainingData(trainingdata)

        #Load current dataset
        print("Loading training data")
        try:
            trafficdataset = TrafficDataset("trainingdata/trainingdata_" + sys.argv[1] + ".pickle")

        except FileNotFoundError as e:
            #No data, so generate some, then loop back
            print("Generating new training data")
            for sumoconfig in sumoconfigs:
                if sumoconfig == "IG":
                    #reload(intersectionGenerator)
                    intersectionGenerator.main()
                else:
                    reload(runnerQueueSplit)
                    runnerQueueSplit.main(sumoconfig, 0, False, False, True)
                    dumpTrainingData(trainingdata)
            continue
        
        
        #Set up and train initial neural nets on first iteration
        if firstIter:
            firstIter = False
            #Do NN setup
            for light in ["light"]:#trainingdata:
                maxnlanes = 3 #Going to assume we have at most 3 lanes per road, and that the biggest number lane is left-turn only
                maxnroads = 4 #And assume 4-way intersections for now
                maxnclusters = 5 #And assume at most 10 clusters per lane
                ndatapercluster = 3 #Arrival, departure, weight
                maxnphases = 12 #Should be enough to handle both leading and lagging lefts
                
                nextra = 2 #Proportion of phase length used, current time
                ninputs = maxnlanes*maxnroads*maxnclusters*ndatapercluster + maxnlanes*maxnroads*maxnphases + maxnphases + nextra

                if crossEntropyLoss:
                    agents[light] = Net(ninputs, 2, 1024)
                else:
                    agents[light] = Net(ninputs, 1, 128)
                
                # if testSurtrac:
                #     agents[light] = Net(182, 1, 64)
                # else:
                #     agents[light] = Net(26, 1, 32)
                optimizers[light] = torch.optim.Adam(agents[light].parameters(), lr=learning_rate)
                MODEL_FILES[light] = 'models/imitate_'+light+'.model' # Once your program successfully trains a network, this file will be written
                if not resetNN:
                    try:
                        agents[light].load(MODEL_FILES[light]).to('cuda')
                    except:
                        print("Warning: Model " + light + " not found, starting fresh")
                losses[light] = []
                epochlosses[light] = []
                daggertimes[light] = []
        
        #Train everything once - note that all losses are likely to spike after new training data comes in
        for light in ["light"]:#trainingdata:
            trainLight(light, trafficdataset)

        #New plan: Instead of training for a set number of epochs, we'll train the worst light until it stops improving, then run DAgger again
        while True:
            worstlight = None
            worstloss = -inf
            for light in ["light"]:#trainingdata:
                if worstlight == None or epochlosses[light][-1] > worstloss:
                    worstlight = light
                    worstloss = epochlosses[light][-1]
            trainLight(worstlight, trafficdataset)
            #If worst light didn't improve, break and get more data
            if epochlosses[worstlight][-1] >= epochlosses[worstlight][-2]:
                break #And get new data

        #Loop back to start the next DAgger loop, store when we did this for plotting purposes
        for light in ["light"]:#trainingdata:
            daggertimes[light].append(len(losses[light]))

#Train the given light for a single epoch
def trainLight(light, dataset):
    #global losses
    print(light)
    avgloss = 0
    nlosses = 0
    totalloss = 0

    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle=True, num_workers=0)
    for i, databatch in enumerate(dataloader):
        databatch.to('cuda')
        outputNN = agents[light](databatch['input']).flatten(1) # Output from NN
        target = databatch['target'].clone().detach().flatten() # Target from expert
        loss = loss_fn(outputNN, target) # calculate loss between network action and expert action (Surtrac action)

        avgloss += float(loss.item()) # record loss for printing. Pretty sure this is an average over the minibatch
        nlosses += 1

        if nlosses % nLossesBeforeReset <= 1:#(nlosses-batch_size) % nLossesBeforeReset:
            avgloss /= (nLossesBeforeReset)
            losses[light].append(avgloss)
            log('Iteration: {}   Loss: {}'.format(nlosses*batch_size,avgloss))
            avgloss = 0
        loss.backward() # perform backprop to collect gradients
        optimizers[light].step() # perform one step of update with optimizer
        optimizers[light].zero_grad() # reset accumulated gradients to 0
    
    epochlosses[light].append(totalloss/dataset.__len__())
    agents[light].save(MODEL_FILES[light])
    plt.figure()
    plt.plot(losses[light])
    for daggertime in daggertimes[light]:
        plt.axvline(x=daggertime, color='k', linestyle='--')
    plt.xlabel("Sets of " + str(nLossesBeforeReset*batch_size) + " Points")
    plt.ylabel("Average Loss")
    plt.title("Losses, light=" + str(light))
    #plt.show() #NOTE: Blocks code execution until you close the plot
    plt.savefig("Plots/Losses, light=" + str(light)+".png")
    plt.close()

#Dumps training data to an Excel file for human readability
def dumpTrainingData(trainingdata):
    timeout = 10#60
    starttime = time.time()
    print("Writing training data to spreadsheet")
    try:
        book = openpyxl.Workbook()
        sheets = dict()
        for light in trainingdata:
            sheets[light] = book.create_sheet(light, -1)
            sheets[light].cell(1, 1, "Input")
            row = 2
            for batch in trainingdata[light]:
                if time.time() - starttime > timeout:
                    print("Timed out while writing data to Excel")
                    assert(False) #This should trigger the try-catch and get us out of here
                for linenum in range(batch[0].size(0)):
                    col = 1
                    for entry in batch[0][linenum]:
                        sheets[light].cell(row, col, float(entry))
                        col += 1
                    if row == 2:
                        sheets[light].cell(1, col, "Expert Output")
                    sheets[light].cell(row, col, float(batch[1][linenum]))

                    col += 1
                    if row == 2:
                        sheets[light].cell(1, col, "NN Output")
                    if len(batch) > 2:
                        sheets[light].cell(row, col, float(batch[2][linenum]))
                    row += 1
    except Exception as e:
        print(e)
        print("Error dumping training data to Excel, ignoring and continuing")
    finally:
        book.save("trainingdata/trainingdata_"+sys.argv[1]+".xlsx")



# this is the main entry point of this script
if __name__ == "__main__":
    main(sys.argv[1:])