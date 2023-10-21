#!/usr/bin/env python
#Adapted from TraCI tutorial here: https://github.com/eclipse/sumo/blob/main/tests/complex/tutorial/traci_tls/runner.py

#QueueSplit5 added a first iteration of a Surtrac model to QueueSplit4
#New in QueueSplit6: When there's multiple light phases a lane can go in, don't double-create clusters
#New in QueueSplit7: Adding Surtrac to simulate-ahead
#New in QueueSplit8: Adding communication between intersections. Which then requires re-compacting clusters, etc. For now, don't take advantage of known routes
#New in QueueSplit9: Take advantage of known routes from vehicle routing (NOTE: route may change as vehicle approaches intersection...)
#New in QueueSplit10: Optimize for speed (calling Surtrac less often in routing, merging predicted clusters)
#New in QueueSplit11: Compute delay, not just average travel time, for cars. Also fixed a bug with Surtrac code DP (was removing sequences it shouldn't have)
#QueueSplit12: Multithread the Surtrac code (it's really slow otherwise). Also, use the full Surtrac schedule rather than assuming we'll update every timestep
#QueueSplit13: Surtrac now (correctly) no longer overwrites all the finish times of other lanes with the start time of the currently scheduled cluster (leads to problems when a long cluster, then compatible short cluster, get scheduled, as the next cluster can then start earlier than it should). VOI now gets split into all lanes on starting edge
#14: Anytime routing, better stats on timeouts and teleports, added mingap to all cluster durations, using a timestep that divides mingap and surtracFreq, opposing traffic blocks for mingap not just one timestep, combining clusters at lights into a single queue
#15: Surtrac now runs once, assuming all vehicles travel their same routes, and then gets looked up during routing simulations. Also adding a flag to disable predicting clusters with Surtrac. REAL-TIME PERFORMANCE!!!
#16: Be lazy in routing simulations - don't need to check a lane until the front vehicle gets close to the end. (Didn't seem to give much speedup though.) Reusing Surtrac between multiple timesteps (and backported this to 15). Various attempts at cleanup and speedup.
#17: Move computation of Surtrac's predicted outflows to the end, after we already know what the schedule should be, rather than building it into the algorithm and generating predicted outflows for schedules we won't end up using. Fixed a bug where initially splitting the VOI into all lanes added it at the modified start time, and possibly in the wrong place
#18: Add imitation learning version of Surtrac

from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import optparse
import random
from numpy import inf
import numpy as np
import time
import matplotlib.pyplot as plt
import math
from copy import deepcopy, copy
from collections import Counter
from heapq import * #priorityqueue
import threading
import xml.etree.ElementTree as ET

# we need to import python modules from the $SUMO_HOME/tools directory
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

from sumolib import checkBinary
import traci  #To interface with SUMO simulations
import sumolib #To query node/edge stuff about the network
import pickle #To save/load traffic light states

pSmart = 1.0 #Adoption probability
useLastRNGState = False #To rerun the last simulation without changing the seed on the random number generator

clusterthresh = 5 #Time between cars before we split to separate clusters
mingap = 2.5 #Minimum allowed space between cars
timestep = 0.5 #Amount of time between updates. In practice, mingap rounds up to the nearest multiple of this
detectordist = 50 #How far before the end of a road the detectors that trigger reroutes are

#Hyperparameters for multithreading
multithreadRouting = True #Do each routing simulation in a separate thread. Enable for speed, but can mess with profiling
multithreadSurtrac = True #Compute each light's Surtrac schedule in a separate thread. Enable for speed, but can mess with profiling
reuseSurtrac = False #Does Surtrac computations in a separate thread, shared between all vehicles doing routing. Keep this true unless we need everything single-threaded (ex: for debugging), or if running with fixed timing plans (routingSurtracFreq is huge) to avoid doing this computation
debugMode = True #Enables some sanity checks and assert statements that are somewhat slow but helpful for debugging
mainSurtracFreq = 1 #Recompute Surtrac schedules every this many seconds in the main simulation (technically a period not a frequency). Use something huge like 1e6 to disable Surtrac and default to fixed timing plans.
routingSurtracFreq = 2.5 #Recompute Surtrac schedules every this many seconds in the main simulation (technically a period not a frequency). Use something huge like 1e6 to disable Surtrac and default to fixed timing plans.
recomputeRoutingSurtracFreq = 1 #Maintain the previously-computed Surtrac schedules for all vehicles routing less than this many seconds in the main simulation. Set to 1 to only reuse results within the same timestep. Does nothing when reuseSurtrac is False.
disableSurtracPred = True #Speeds up code by having Surtrac no longer predict future clusters for neighboring intersections
predCutoffMain = 0 #Surtrac receives communications about clusters arriving this far into the future in the main simulation
predCutoffRouting = 0 #Surtrac receives communications about clusters arriving this far into the future in the routing simulations
predDiscount = 1 #Multiply predicted vehicle weights by this because we're not actually sure what they're doing. 0 to ignore predictions, 1 to treat them the same as normal cars.

#To test
testNNdefault = True #Uses NN over Dumbtrac for light control if both are true
testDumbtrac = True #If true, also stores Dumbtrac, not Surtrac, in training data (if appendTrainingData is also true)
resetTrainingData = False
appendTrainingData = False
learnYellow = False
learnMinMaxDurations = False
FTP = False

#Don't change parameters below here
#For testing durations to see if there's drift between fixed timing plans executed in main simulation and routing simulations.
simdurations = dict()
simdurationsUsed = False
realdurations = dict()

max_edge_speed = 0.0 #Overwritten when we read the route file

carsOnNetwork = []
oldids = dict()
isSmart = dict() #Store whether each vehicle does our routing or not
lightphasedata = dict()
lightlinks = dict()
prioritygreenlightlinks = dict()
lowprioritygreenlightlinks = dict()
prioritygreenlightlinksLE = dict()
lowprioritygreenlightlinksLE = dict()
lightlanes = dict()
lightoutlanes = dict()
notlightlanes = dict()
notlightoutlanes = dict()
lights = []
notLights = []
edges = []
lightlinkconflicts = dict()
lanenums = dict()
speeds = dict()
fftimes = dict() #Free flow times for each edge/lane (dict contains both) and from each light (min over outlanes)
links = dict()
lengths = dict()
turndata = []
normprobs = dict()
timedata = dict()
surtracdata = dict()
lanephases = dict()
mainlastswitchtimes = dict()
currentRoutes = dict()
routeStats = dict()
hmetadict = dict()
delay3adjdict = dict()
lightphases = dict()
laneDict = dict()
sumoPredClusters = [] #This'll update when we call doSurtrac from sumo things
rerouterLanes = dict()
rerouterEdges = dict()
vehiclesOnNetwork = []
dontReroute = []

#Predict traffic entering network
arrivals = dict()
maxarrivalwindow = -300 #Use negative number to not predict new incoming cars during routing
newcarcounter = 0

totalSurtracTime = 0
totalSurtracClusters = 0
totalSurtracRuns = 0

#Threading routing
toReroute = []
reroutedata = dict()
threads = dict()
killSurtracThread = True

nRoutingCalls = 0
routingTime = 0

#Neural net things
import torch
from torch import nn
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
            nn.Linear(hidden_size, hidden_size),
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

agents = dict()
optimizers = dict()
trainingdata = dict() #Going to be a list of 2-elt tuples (in, out) = ([in1, in2, ...], out)

actions = [0, 1]
learning_rate = 0.0005
avgloss = 0
nlosses = 0
nLossesBeforeReset = 1000

ndumbtrac = 0
ndumbtracerr = 0

loss_fn = torch.nn.CrossEntropyLoss(weight=torch.Tensor([1, 50]))

#Non-NN stuff
def mergePredictions(clusters, predClusters):
    mergedClusters = pickle.loads(pickle.dumps(clusters)) #Because pass-by-value stuff
    for lane in clusters:
        if lane in predClusters:
            mergedClusters[lane] += predClusters[lane] #Concatenate known clusters with predicted clusters
            consolidateClusters(mergedClusters[lane])
    return mergedClusters

def consolidateClusters(clusters):
    i = 0
    while i < len(clusters):
        j = i+1
        while j < len(clusters):
            #Check if clusters i and j should merge
            if clusters[i]["arrival"] <= clusters[j]["arrival"] and clusters[j]["arrival"] <= clusters[i]["departure"] + clusterthresh:
                #Merge j into i
                clusters[i]["departure"] = max(clusters[i]["departure"], clusters[j]["departure"])
                clusters[i]["weight"] += clusters[j]["weight"]
                clusters[i]["cars"] += clusters[j]["cars"] #Concatenate (I hope)
                clusters.pop(j)
            else:
                if clusters[j]["arrival"] <= clusters[i]["arrival"] and clusters[i]["arrival"] <= clusters[j]["departure"] + clusterthresh:
                    #Merge i into j
                    clusters[j]["departure"] = max(clusters[i]["departure"], clusters[j]["departure"])
                    clusters[j]["weight"] += clusters[i]["weight"]
                    clusters[j]["cars"] += clusters[i]["cars"] #Concatenate (I hope)
                    clusters[i] = clusters[j]
                    clusters.pop(j)
            j+=1
        i+=1

def dumbtrac(simtime, light, clusters, lightphases, lastswitchtimes):
    if FTP:
        return dumbtracFTP(simtime, light, clusters, lightphases, lastswitchtimes)
    else:
        return dumbtracActuated(simtime, light, clusters, lightphases, lastswitchtimes)

def dumbtracFTP(simtime, light, clusters, lightphases, lastswitchtimes):

    phase = lightphases[light]
    lastSwitch = lastswitchtimes[light]

    #For FTP
    if "Y" in lightphasedata[light][phase].state or "y" in lightphasedata[light][phase].state:
        return surtracdata[light][phase]["minDur"] - (simtime-lastSwitch)
    else:
        return 30 - (simtime-lastSwitch)

    

def dumbtracActuated(simtime, light, clusters, lightphases, lastswitchtimes):
    phase = lightphases[light]
    lastSwitch = lastswitchtimes[light]
    
    maxfreq = max(routingSurtracFreq, mainSurtracFreq, timestep, 1)

    if surtracdata[light][phase]["maxDur"]- maxfreq <= surtracdata[light][phase]["minDur"]:
        #Edge case where duration range is smaller than period between updates, in which case overruns are unavoidable
        if simtime - lastSwitch < surtracdata[light][phase]["minDur"]:
            phaselenprop = -1
        else:
            phaselenprop = 2
    else:
        phaselenprop = (simtime - lastSwitch - surtracdata[light][phase]["minDur"])/(surtracdata[light][phase]["maxDur"]- maxfreq - surtracdata[light][phase]["minDur"])

    #Satisfy phase length requirements
    if phaselenprop < 0:
        return 10 #If haven't reached min length, continue
    if phaselenprop >= 1:
        return -10 #If >= max length, switch

    out = -10
    for lane in surtracdata[light][phase]["lanes"]:
        if len(clusters[lane]) > 0 and clusters[lane][0]["arrival"] <= simtime+mingap:
            out = 10#max(out, 2.5*(clusters[lane][0]["weight"]+1)) #If anyone's waiting, continue
    return out #If no one's waiting, switch

def convertToNNInput(simtime, light, clusters, lightphases, lastswitchtimes):
    maxnlanes = 3 #Going to assume we have at most 3 lanes per road, and that the biggest number lane is left-turn only
    maxnroads = 4 #And assume 4-way intersections for now
    nqueued = np.zeros(maxnroads*maxnlanes)
    ntotal = np.zeros(maxnroads*maxnlanes)
    phase = lightphases[light]
    lastSwitch = lastswitchtimes[light]

    maxfreq = max(routingSurtracFreq, mainSurtracFreq, timestep, 1)

    if False: #surtracdata[light][phase]["maxDur"]- maxfreq <= surtracdata[light][phase]["minDur"]:
        #Edge case where duration range is smaller than period between updates, in which case overruns are unavoidable
        print("Warning: minDur and maxDur for light " + light + " are too close together, we might have rounding errors. Also training data is going to be hacky here.")
        if simtime - lastSwitch < surtracdata[light][phase]["minDur"]:
            phaselenprop = -1
        else:
            phaselenprop = 2
    else:
        phaselenprop = (simtime - lastSwitch - surtracdata[light][phase]["minDur"])/(surtracdata[light][phase]["maxDur"]- maxfreq - surtracdata[light][phase]["minDur"])

    #phaselenprop = (simtime - lastSwitch - surtracdata[light][phase]["minDur"])/surtracdata[light][phase]["maxDur"]
    #phaselenprop is negative if we're less than minDur, and greater than 1 if we're greater than maxDur

    prevRoad = None
    roadind = -1
    laneind = -1

    for lane in lightlanes[light]:
        temp = lane.split("_")
        road = temp[0] #Could have problems if road name has underscores, but ignoring for now...
        if lanenums[road] > maxnlanes:
            print("Warning: " + str(road) + " exceeds maxnlanes in convertToNNInput, ignoring some lanes")
        lanenum = int(temp[-1])
        if road != prevRoad or roadind < 0:
            roadind += 1
            laneind = -1
            prevRoad = road
        laneind += 1
        #Last lane on road assumed to be left-turn only and being inserted in last slot
        if laneind + 1 == lanenums[road] or laneind + 1 >= maxnlanes:
            laneind = maxnlanes - 1

        if len(clusters[lane]) > 0 and clusters[lane][0]["arrival"] <= simtime + mingap:
            nqueued[roadind*maxnlanes+laneind] = clusters[lane][0]["weight"]
        ntotaltemp = 0
        for clusterind in range(len(clusters[lane])):
            ntotaltemp += clusters[lane][clusterind]["weight"]
        ntotal[roadind*maxnlanes+laneind] = ntotaltemp

    #return torch.Tensor(np.array([np.concatenate(([phase], [phaselenprop]))]))
    return torch.Tensor(np.array([np.concatenate((nqueued, ntotal, [phase], [phaselenprop]))]))

def convertToNNInputSurtrac(simtime, light, clusters, lightphases, lastswitchtimes):
    maxnlanes = 3 #Going to assume we have at most 3 lanes per road, and that the biggest number lane is left-turn only
    maxnroads = 4 #And assume 4-way intersections for now
    maxnclusters = 10 #And assume at most 10 clusters per lane
    ndatapercluster = 3 #Arrival, departure, weight

    clusterdata = np.zeros(maxnroads*maxnlanes*maxnclusters*ndatapercluster)

    nqueued = np.zeros(maxnroads*maxnlanes)
    ntotal = np.zeros(maxnroads*maxnlanes)
    phase = lightphases[light]
    lastSwitch = lastswitchtimes[light]

    maxfreq = max(routingSurtracFreq, mainSurtracFreq, timestep, 1)

    if surtracdata[light][phase]["maxDur"]- maxfreq <= surtracdata[light][phase]["minDur"]:
        #Edge case where duration range is smaller than period between updates, in which case overruns are unavoidable
        if simtime - lastSwitch < surtracdata[light][phase]["minDur"]:
            phaselenprop = -1
        else:
            phaselenprop = 2
    else:
        phaselenprop = (simtime - lastSwitch - surtracdata[light][phase]["minDur"])/(surtracdata[light][phase]["maxDur"]- maxfreq - surtracdata[light][phase]["minDur"])

    #phaselenprop = (simtime - lastSwitch - surtracdata[light][phase]["minDur"])/surtracdata[light][phase]["maxDur"]
    #phaselenprop is negative if we're less than minDur, and greater than 1 if we're greater than maxDur

    prevRoad = None
    roadind = -1
    laneind = -1

    for lane in lightlanes[light]:
        temp = lane.split("_")
        road = temp[0] #Could have problems if road name has underscores, but ignoring for now...
        if lanenums[road] > maxnlanes:
            print("Warning: " + str(road) + " exceeds maxnlanes in convertToNNInput, ignoring some lanes")
        lanenum = int(temp[-1])
        if road != prevRoad or roadind < 0:
            roadind += 1
            laneind = -1
            prevRoad = road
        laneind += 1

        #Not sharing weights so I'll skip this
        #Last lane on road assumed to be left-turn only and being inserted in last slot
        # if laneind + 1 == lanenums[road] or laneind + 1 >= maxnlanes:
        #     laneind = maxnlanes - 1

        for clusterind in range(len(clusters[lane])):
            if clusterind > maxnclusters:
                print("Warning: Too many clusters on " + str(lane) + ", ignoring the last ones")
                break
            clusterdata[((roadind*maxnlanes+laneind)*maxnclusters+clusterind)*ndatapercluster : ((roadind*maxnlanes+laneind)*maxnclusters+clusterind+1)*ndatapercluster] = [clusters[lane][clusterind]["arrival"]-simtime, clusters[lane][clusterind]["departure"]-simtime, clusters[lane][clusterind]["weight"]]

    #return torch.Tensor(np.array([np.concatenate(([phase], [phaselenprop]))]))
    return torch.Tensor(np.array([np.concatenate((clusterdata, [phase], [phaselenprop]))]))

#@profile
def doSurtracThread(network, simtime, light, clusters, lightphases, lastswitchtimes, inRoutingSim, predictionCutoff, toSwitch, catpreds, bestschedules):
    
    if inRoutingSim:
        freq = max(routingSurtracFreq, timestep)
        ttimestep = timestep
    else:
        freq = max(mainSurtracFreq, 1)
        ttimestep = 1
        
    i = lightphases[light]
    if not learnYellow and ("Y" in lightphasedata[light][i].state or "y" in lightphasedata[light][i].state):
        #Force yellow phases to be min duration regardless of what anything else says, and don't store it as training data
        if simtime - lastswitchtimes[light] >= surtracdata[light][i]["minDur"]:
            dur = 0
        else:
            dur = (surtracdata[light][i]["minDur"] - (simtime - lastswitchtimes[light]))//ttimestep*ttimestep
        #Replace first element with remaining duration, rather than destroying the entire schedule, in case of Surtrac or similar
        if light in bestschedules:
            temp = pickle.loads(pickle.dumps(bestschedules[light][7]))
        else:
            temp = [0]
        temp[0] = dur
        bestschedules[light] = [None, None, None, None, None, None, None, temp]
        return

    if not learnMinMaxDurations:
        if inRoutingSim:
            freq = max(routingSurtracFreq, timestep)
            ttimestep = timestep
        else:
            freq = max(mainSurtracFreq, 1)
            ttimestep = 1
        #Force light to satisfy min/max duration requirements and don't store as training data
        if simtime - lastswitchtimes[light] < surtracdata[light][i]["minDur"]:
            dur = 1e6
            if light in bestschedules:
                temp = pickle.loads(pickle.dumps(bestschedules[light][7]))
            else:
                temp = [0]
            temp[0] = dur
            bestschedules[light] = [None, None, None, None, None, None, None, temp]
            return
        if simtime - lastswitchtimes[light] + freq > surtracdata[light][i]["maxDur"]:
            #TODO this is slightly sloppy if freq > ttimestep - if we're trying to change just before maxDur this'll assume we tried to change at it instead
            dur = (surtracdata[light][i]["maxDur"] - (simtime - lastswitchtimes[light]))//ttimestep*ttimestep
            if light in bestschedules:
                temp = pickle.loads(pickle.dumps(bestschedules[light][7]))
            else:
                temp = [0]
            temp[0] = dur
            bestschedules[light] = [None, None, None, None, None, None, None, temp]
            return

    if testNN or testDumbtrac:
        if testNN:
            if testDumbtrac:
                nnin = convertToNNInput(simtime, light, clusters, lightphases, lastswitchtimes)
            else:
                nnin = convertToNNInputSurtrac(simtime, light, clusters, lightphases, lastswitchtimes)
            outputNN = agents[light](nnin) # Output from NN

            if outputNN <= 0:
                actionNN = 1 #Switch
            else:
                actionNN = 0 #Stick

        if testDumbtrac and not testNN:
            outputDumbtrac = dumbtrac(simtime, light, clusters, lightphases, lastswitchtimes)
            if outputDumbtrac <= 0:
                actionDumbtrac = 1
            else:
                actionDumbtrac = 0
            actionNN = actionDumbtrac

        if actionNN == 0:
            dur = 1e6 #Something really big so we know the light won't change
        else:
            dur = 0
        testnnschedule = [None, None, None, None, None, None, None, [dur]] #Only thing the output needs is a schedule; returns either [0] for switch immediately or [1] for continue for at least another timestep
        assert(len(testnnschedule[7]) > 0)
        #return #Don't return early, might still need to append training data

    if (not testNN and not testDumbtrac) or (appendTrainingData and not testDumbtrac):
        #print("Running surtrac, double-check that this is intended.")
        #We're actually running Surtrac
        if debugMode:
            global totalSurtracRuns
            global totalSurtracClusters
            global totalSurtracTime

        surtracStartTime = time.time()
        totalSurtracRuns += 1

        sult = 3 #Startup loss time
        greedyDP = True

        #Figure out what an initial and complete schedule look like
        nPhases = len(surtracdata[light]) #Number of phases
        bestschedules[light] = [[]] #In case we terminate early or something??

        emptyStatus = dict()
        fullStatus = dict()
        nClusters = 0
        maxnClusters = 0

        #TODO: Does this vectorize somehow?
        for lane in lightlanes[light]:
            emptyStatus[lane] = 0
            fullStatus[lane] = len(clusters[lane])
            nClusters += fullStatus[lane]
            if maxnClusters < fullStatus[lane]:
                maxnClusters = fullStatus[lane]
        if debugMode:
            totalSurtracClusters += nClusters
        #If there's nothing to do, send back something we recognize as "no schedule"
        if nClusters == 0:
            bestschedules[light] = [[]]
            return

        #Stuff in the partial schedule tuple
        #0: list of indices of the clusters we've scheduled
        #1: schedule status (how many clusters from each lane we've scheduled)
        #2: current light phase
        #3: time when each direction will have finished its last scheduled cluster
        #4: time when all directions are finished with scheduled clusters ("total makespan" + starting time...)
        #5: total delay
        #6: last switch time
        #7: planned total durations of all phases
        #8: predicted outflows (as clusters - arrival, departure, list of cars, weights, etc.) Blank for now since I'll generate it at the end.
        #9: pre-predict data (cluster start times and compression factors) which I'll use to figure out predicted outflows once we've determined the best schedule

        emptyPreds = dict()
        for lane in lightoutlanes[light]:
            emptyPreds[lane] = []

        # emptyPrePreds = dict()
        # for lane in lightlanes[light]:
        #     emptyPrePreds[lane] = []
        lenlightlaneslight = len(lightlanes[light])
        assert(lenlightlaneslight > 0)
        emptyPrePreds = np.zeros((lenlightlaneslight, maxnClusters, 2))

        phase = lightphases[light]
        lastSwitch = lastswitchtimes[light]
        schedules = [([], emptyStatus, phase, [simtime]*len(surtracdata[light][phase]["lanes"]), simtime, 0, lastSwitch, [simtime - lastSwitch], [], emptyPrePreds)]

        for _ in range(nClusters): #Keep adding a cluster until #clusters added = #clusters to be added
            scheduleHashDict = dict()
            for schedule in schedules:
                for laneindex in range(lenlightlaneslight):
                    lane = lightlanes[light][laneindex]
    #            laneindex = -1
    #            for lane in lightlanes[light]:
    #                laneindex += 1
                    if schedule[1][lane] == fullStatus[lane]:
                        continue
                    #Consider adding next cluster from surtracdata[light][i]["lanes"][j] to schedule
                    newScheduleStatus = copy(schedule[1]) #Shallow copy okay? Dict points to int, which is stored by value
                    newScheduleStatus[lane] += 1
                    assert(newScheduleStatus[lane] <= maxnClusters)
                    phase = schedule[2]

                    #Now loop over all phases where we can clear this cluster
                    try:
                        assert(len(lanephases[lane]) > 0)
                    except:
                        print(lane)
                        print("ERROR: Can't clear this lane ever?")
                        
                    for i in lanephases[lane]:
                        #TODO: Stop scheduling stuff on yellow, it's probably bad (also slow?). Make sure this works though.
                        if not learnYellow and ("Y" in lightphasedata[light][i].state or "y" in lightphasedata[light][i].state):
                            continue
                        directionalMakespans = copy(schedule[3])

                        nLanes = len(surtracdata[light][i]["lanes"])
                        j = surtracdata[light][i]["lanes"].index(lane)

                        newDurations = copy(schedule[7]) #Shallow copy should be fine

                        clusterind = newScheduleStatus[lane]-1 #We're scheduling the Xth cluster; it has index X-1
                        ist = clusters[lane][clusterind]["arrival"] #Intended start time = cluster arrival time
                        dur = clusters[lane][clusterind]["departure"] - ist + mingap #+mingap because next cluster can't start until mingap after current cluster finishes
                        mindur = max((clusters[lane][clusterind]["weight"] )*mingap, 0) #No -1 because fencepost problem; next cluster still needs 2.5s of gap afterwards
                        delay = schedule[5]

                        if phase == i:
                            pst = schedule[3][j]
                            newLastSwitch = schedule[6] #Last switch time doesn't change
                            ast = max(ist, pst)
                            newdur = max(dur - (ast-ist), mindur)
                            currentDuration = max(ist, ast)+newdur-schedule[6] #Total duration of current light phase if we send this cluster without changing phase

                        if not phase == i or currentDuration > surtracdata[light][i]["maxDur"]: #We'll have to switch the light, possibly mid-cluster

                            if not phase == i:
                                #Have to switch light phases.
                                newFirstSwitch = max(schedule[6] + surtracdata[light][phase]["minDur"], schedule[4]-mingap) #Because I'm adding mingap after all clusters, but here the next cluster gets delayed
                            else:
                                #This cluster is too long to fit entirely in the current phase
                                newFirstSwitch = schedule[6] + surtracdata[light][phase]["maxDur"] #Set current phase to max duration
                                #Figure out how long the remaining part of the cluster is
                                tSent = surtracdata[light][i]["maxDur"] - (max(ist, ast)-schedule[6]) #Time we'll have run this cluster for before the light switches
                                if tSent < 0: #Cluster might arrive after the light would have switched due to max duration (ist is big), which would have made tSent go negative
                                    tSent = 0
                                    try:
                                        assert(mindur >= 0)
                                        assert(dur >= 0)
                                    except AssertionError as e:
                                        print(mindur)
                                        print(dur)
                                        raise(e)

                                if mindur > 0 and dur > 0: #Having issues with negative weights, possibly related to cars contributing less than 1 to weight having left the edge
                                    delay += tSent/dur*clusters[surtracdata[light][i]["lanes"][j]][clusterind]["weight"]*((ast-ist)-1/2*(dur-newdur) )
                                    mindur *= 1-tSent/dur #Assuming uniform density
                                else:
                                    print("Negative weight, what just happened?")
                                dur -= tSent

                            newLastSwitch = newFirstSwitch + surtracdata[light][(phase+1)%nPhases]["timeTo"][i] #Switch right after previous cluster finishes (why not when next cluster arrives minus sult? Maybe try both?)                        
                            pst = newLastSwitch + sult #Total makespan + switching time + startup loss time
                            #Technically this sult implementation isn't quite right, as a cluster might reach the light as the light turns green and not have to stop and restart
                            directionalMakespans = [pst]*nLanes #Other directions can't schedule a cluster before the light switches

                            newDurations[-1] = newFirstSwitch - schedule[6] #Previous phase lasted from when it started to when it switched
                            tempphase = (phase+1)%nPhases
                            while tempphase != i:
                                newDurations.append(surtracdata[light][i]["minDur"])
                                tempphase = (tempphase+1)%nPhases
                            newDurations.append(0) #Duration of new phase i. To be updated on future loops once we figure out when the cluster finishes
                            assert(newDurations != schedule[7]) #Confirm that shallow copy from before is fine

                        ast = max(ist, pst)
                        newdur = max(dur - (ast-ist), mindur) #Compress cluster once cars start stopping

                        newPrePredict = copy(schedule[9])#pickle.loads(pickle.dumps(schedule[9]))
                        # print(np.shape(newPrePredict))
                        # print(lightoutlanes[light])
                        # print(lane)
                        # print(laneindex)
                        # print(newScheduleStatus[lane]-1)
                        newPrePredict[laneindex][newScheduleStatus[lane]-1][0] = ast #-1 because zero-indexing; first cluster has newScheduleStatus[lane] = 1, but is stored at index 0
                        if dur <= mindur:
                            newPrePredict[laneindex][newScheduleStatus[lane]-1][1] = 0 #Squish factor = 0 (no squishing)
                        else:
                            newPrePredict[laneindex][newScheduleStatus[lane]-1][1] = (dur-newdur)/(dur-mindur) #Squish factor equals this thing
                            #If newdur = mindur, compression factor = 1, all gaps are 2.5 (=mindur)
                            #If newdur = dur, compression factor = 0, all gaps are original values
                            #Otherwise smoothly interpolate
                        
                        #Tell other clusters to also start no sooner than max(new ast, old directionalMakespan value) to preserve order
                        #That max is important, though; blind overwriting is wrong, as you could send a long cluster, then a short one, then change the light before the long one finishes
                        assert(len(directionalMakespans) == len(surtracdata[light][i]["lanes"]))
                        directionalMakespans[j] = ast+newdur+mingap

                        directionalMakespans = np.maximum(directionalMakespans, ast).tolist()
                        
                        delay += clusters[surtracdata[light][i]["lanes"][j]][clusterind]["weight"]*((ast-ist)-1/2*(dur-newdur) ) #Delay += #cars * (actual-desired). 1/2(dur-newdur) compensates for the cluster packing together as it waits (I assume uniform compression)
                        try:
                            assert(delay >= schedule[5] - 1e-10) #Make sure delay doesn't go negative somehow
                        except AssertionError as e:
                            print("Negative delay, printing lots of debug stuff")
                            #print(clusters)
                            print(light)
                            print(lane)
                            print(clusters[surtracdata[light][i]["lanes"][j]][clusterind])
                            print(ast)
                            print(ist)
                            print(dur)
                            print(newdur)
                            print((ast-ist)-1/2*(dur-newdur))
                            raise(e)

                        newMakespan = max(directionalMakespans)
                        currentDuration = newMakespan - newLastSwitch

                        newDurations[-1] = currentDuration 
                        #Stuff in the partial schedule tuple
                        #0: list of indices of the clusters we've scheduled
                        #1: schedule status (how many clusters from each lane we've scheduled)
                        #2: current light phase
                        #3: time when each direction will have finished its last scheduled cluster
                        #4: time when all directions are finished with scheduled clusters ("total makespan" + starting time...)
                        #5: total delay
                        #6: last switch time
                        #7: planned total durations of all phases
                        #8: predicted outflows (as clusters - arrival, departure, list of cars, weights, etc.) Blank for now since I'll generate it at the end.
                        #9: pre-predict data (cluster start times and compression factors) which I'll use to figure out predicted outflows once we've determined the best schedule

                        newschedule = (schedule[0]+[(i, j)], newScheduleStatus, i, directionalMakespans, newMakespan, delay, newLastSwitch, newDurations, [], newPrePredict)
                        
                        #DP on partial schedules
                        key = (tuple(newschedule[1].values()), newschedule[2]) #Key needs to be something immutable (like a tuple, not a list)

                        if not key in scheduleHashDict:
                            scheduleHashDict[key] = [newschedule]
                        else:
                            keep = True
                            testscheduleind = 0
                            while testscheduleind < len(scheduleHashDict[key]):
                                testschedule = scheduleHashDict[key][testscheduleind]

                                #These asserts should follow from how I set up scheduleHashDict
                                if debugMode:
                                    assert(newschedule[1] == testschedule[1])
                                    assert(newschedule[2] == testschedule[2])
                                
                                #NOTE: If we're going to go for truly optimal, we also need to check all makespans, plus the current phase duration
                                #OTOH, if people seem to think fast greedy approximations are good enough, I'm fine with that
                                if newschedule[5] >= testschedule[5] and (greedyDP or newschedule[4] >= testschedule[4]):
                                    #New schedule was dominated; remove it and don't continue comparing (old schedule beats anything new one would)
                                    keep = False
                                    break
                                if newschedule[5] <= testschedule[5] and (greedyDP or newschedule[4] <= testschedule[4]):
                                    #Old schedule was dominated; remove it
                                    scheduleHashDict[key].pop(testscheduleind)
                                    continue
                                #No dominance, keep going
                                testscheduleind += 1

                            if keep:
                                scheduleHashDict[key].append(newschedule)
                        if debugMode:
                            assert(len(scheduleHashDict[key]) > 0)

            schedules = sum(list(scheduleHashDict.values()), []) #Each key has a list of non-dominated partial schedules. list() turns the dict_values object into a list of those lists; sum() concatenates to one big list of partial schedules. (Each partial schedule is stored as a tuple)

        mindelay = np.inf
        bestschedule = [[]]
        for schedule in schedules:
            if schedule[5] < mindelay:
                mindelay = schedule[5]
                bestschedule = schedule

        if not bestschedule == [[]]:
            #We have our best schedule, now need to generate predicted outflows
            if disableSurtracPred:
                newPredClusters = emptyPreds
            else:
                newPredClusters = pickle.loads(pickle.dumps(emptyPreds)) #Deep copy needed if I'm going to merge clusters

                nextSendTimes = [] #Priority queue
                clusterNums = dict()
                carNums = dict()
                #for lane in lightlanes[light]:
                for laneind in range(lenlightlaneslight):
                    lane = lightlanes[light][laneind]
                    clusterNums[lane] = 0
                    carNums[lane] = 0
                    if len(clusters[lane]) > clusterNums[lane]: #In case there's no clusters on a given lane
                        #heappush(nextSendTimes, (bestschedule[9][lane][clusterNums[lane]][0], lane)) #Pre-predict for appropriate lane for first cluster, get departure time, stuff into a priority queue
                        heappush(nextSendTimes, (bestschedule[9][laneind][clusterNums[lane]][0], laneind)) #Pre-predict for appropriate lane for first cluster, get departure time, stuff into a priority queue

                while len(nextSendTimes) > 0:
                    (nextSendTime, laneind) = heappop(nextSendTimes)
                    lane = lightlanes[light][laneind]
                    if nextSendTime + fftimes[light] > simtime + predictionCutoff:
                        #fftimes[light] is the smallest fftime of any output lane
                        #So if we're here, there's no way we'll ever want to predict this or any later car
                        break

                    cartuple = clusters[lane][clusterNums[lane]]["cars"][carNums[lane]]
                    if not cartuple[0] in isSmart or isSmart[cartuple[0]]: #It's possible we call this from QueueSim, at which point we split the vehicle being routed and wouldn't recognize the new names. Anything else should get assigned to isSmart or not on creation
                        #Split on "|" and "_" to deal with splitty cars correctly
                        route = currentRoutes[cartuple[0].split("|")[0].split("_")[0]] #.split to deal with the possibility of splitty cars in QueueSim
                        edge = lane.split("_")[0]
                        if not edge in route:
                            #Not sure if or why this happens - maybe the route is changing and predictions aren't updating?
                            #Can definitely happen for a splitty car inside QueueSim
                            #Regardless, don't predict this car forward and hope for the best?
                            if not "|" in cartuple[0] and not "_" in cartuple[0]:
                                #Smart car is on an edge we didn't expect. Most likely it changed route between the previous and current Surtrac calls. Get rid of it now, TODO can we be cleverer?
                                # print(cartuple[0])
                                # print(route)
                                # print(edge)
                                # print("Warning, smart car on an edge that's not in its route. This shouldn't happen? Assuming a mispredict and removing")
                                continue
                            #TODO: else should predict it goes everywhere?
                            continue
                        edgeind = route.index(edge)
                        if edgeind+1 == len(route):
                            #At end of route, don't care
                            continue
                        nextedge = route[edgeind+1]
                        
                        if not nextedge in normprobs[lane]:
                            #Means normprobs[lane] would be 0; nobody turned from this lane to this edge in the initial data
                            #Might be happening if the car needs to make a last-minute lane change to stay on its route?
                            #TODO: Find a lane where it can continue with the route and go from there? Ignoring for now
                            #NEXT TODO: Apparently still a thing even with splitting the initial VOI to multiple lanes???
                            continue

                        for nextlaneind in range(lanenums[nextedge]):
                            nextlane = nextedge+"_"+str(nextlaneind)
                            arr = nextSendTime + fftimes[nextlane]
                            if arr > simtime + predictionCutoff:
                                #Don't add to prediction; it's too far in the future. And it'll be too far into the future for all other lanes on this edge too, so just stop
                                break

                            if not nextlane in turndata[lane] or turndata[lane][nextlane] == 0:
                                #Car has zero chance of going here, skip
                                continue

                            if len(newPredClusters[nextlane]) == 0 or arr > newPredClusters[nextlane][-1]["departure"] + clusterthresh:
                                #Add a new cluster
                                newPredCluster = dict()
                                newPredCluster["endpos"] = 0
                                newPredCluster["time"] = ast
                                newPredCluster["arrival"] = arr
                                newPredCluster["departure"] = arr
                                newPredCluster["cars"] = []
                                newPredCluster["weight"] = 0
                                newPredClusters[nextlane].append(newPredCluster)

                            modcartuple = (cartuple[0], arr, cartuple[2]*predDiscount*turndata[lane][nextlane] / normprobs[lane][nextedge], cartuple[3])
                            newPredClusters[nextlane][-1]["cars"].append(modcartuple)
                            newPredClusters[nextlane][-1]["weight"] += modcartuple[2]
                            newPredClusters[nextlane][-1]["departure"] = arr
                    else:
                        for nextlane in turndata[lane]:
                            #Copy-paste previous logic for creating a new cluster
                            arr = nextSendTime + fftimes[nextlane]
                            if arr > simtime + predictionCutoff:
                                #Don't add to prediction; it's too far in the future. Other lanes may differ though
                                continue

                            if not nextlane in turndata[lane] or turndata[lane][nextlane] == 0:
                                #Car has zero chance of going here, skip
                                continue

                            if len(newPredClusters[nextlane]) == 0 or arr > newPredClusters[nextlane][-1]["departure"] + clusterthresh:
                                #Add a new cluster
                                newPredCluster = dict()
                                newPredCluster["endpos"] = 0
                                newPredCluster["time"] = ast
                                newPredCluster["arrival"] = arr
                                newPredCluster["departure"] = arr
                                newPredCluster["cars"] = []
                                newPredCluster["weight"] = 0
                                newPredClusters[nextlane].append(newPredCluster)

                            modcartuple = (cartuple[0], arr, cartuple[2]*turndata[lane][nextlane], cartuple[3])
                            newPredClusters[nextlane][-1]["cars"].append(modcartuple)
                            newPredClusters[nextlane][-1]["weight"] += modcartuple[2]
                            newPredClusters[nextlane][-1]["departure"] = arr
                    
                    #Added car to predictions, now set up the next car
                    carNums[lane] += 1
                    while len(clusters[lane]) > clusterNums[lane] and len(clusters[lane][clusterNums[lane]]["cars"]) == carNums[lane]: #Should fire at most once, but use while just in case of empty clusters...
                        clusterNums[lane] += 1
                        carNums[lane] = 0
                    if len(clusters[lane]) == clusterNums[lane]:
                        #Nothing left on this lane, we're done here
                        #nextSendTimes.pop(lane)
                        continue
                    if carNums[lane] == 0:
                        heappush(nextSendTimes, (bestschedule[9][laneind][clusterNums[lane]][0], laneind)) #Time next cluster is scheduled to be sent
                    else:
                        #Account for cluster compression
                        prevSendTime = nextSendTime #When we sent the car above
                        rawSendTimeDelay = clusters[lane][clusterNums[lane]]["cars"][carNums[lane]][1] - clusters[lane][clusterNums[lane]]["cars"][carNums[lane]-1][1] #Time between next car and this car in the original cluster
                        compFac = bestschedule[9][laneind][clusterNums[lane]][1] #Compression factor in case cluster is waiting at a red light
                        sendTimeDelay = compFac*mingap + (1-compFac)*rawSendTimeDelay #Update time delay using compression factor
                        newSendTime = prevSendTime + sendTimeDelay #Compute time we'd send next car
                        heappush(nextSendTimes, (newSendTime, laneind))

            #Predicting should be done now
            #bestschedule[8] = newPredClusters #I'd store this, but tuples are immutable and we don't actually use it anywhere...
        
            catpreds.update(newPredClusters)
            if len(bestschedule[7]) > 0:
                bestschedule[7][0] -= (simtime - lastswitchtimes[light])
            bestschedules[light] = bestschedule
        else:
            print(light)
            print("No schedules anywhere? That shouldn't happen...")


            
        #nnin = convertToNNInput(simtime, light, clusters, lightphases, lastswitchtimes) #If stuff breaks, make sure none of this gets changed as we go. (Tested when I first wrote this.)
        # actionSurtrac = 0
        # if bestschedule[7][0] <= simtime - lastswitchtimes[light]:
        #     actionSurtrac = 1
        #NN should take in nnin, and try to return action, and do backprop accordingly
        #target = torch.tensor([bestschedule[7][0] - (simtime - lastswitchtimes[light])]) # Target from expert

        if debugMode:
            totalSurtracTime += time.time() - surtracStartTime
    
    if appendTrainingData:
        if testDumbtrac:
            outputDumbtrac = dumbtrac(simtime, light, clusters, lightphases, lastswitchtimes)
            target = torch.tensor([[outputDumbtrac-0.25]])#.unsqueeze(1) # Target from expert
            nnin = convertToNNInput(simtime, light, clusters, lightphases, lastswitchtimes)
        else:
            target = torch.tensor([[bestschedule[7][0]-0.25]])#.unsqueeze(1) # - (simtime - lastswitchtimes[light])]) # Target from expert
            nnin = convertToNNInputSurtrac(simtime, light, clusters, lightphases, lastswitchtimes)

        if testNN:
            trainingdata[light].append((nnin, target, torch.tensor([[outputNN]])))
        else:
            trainingdata[light].append((nnin, target))
        #NEXT TODO this seems to go through all lights three times before printing the doSurtrac output for all lights three times. Why?
        #No chance trainNN is creating multiple interacting instances of this, is there?
        #print("Light phase training data")
        #print(light)
        #print(nnin[:, -2:])
    
    if testNN or testDumbtrac:
        bestschedules[light] = testnnschedule
        #if len(bestschedules[light][7]) == 0:
        #    print('prepretest apparently empty duration list') #NEXT TODO this isn't triggering but the ones in doSurtrac are - what's happening??? Something to do with shallow copies, it looks like...


#@profile
def doSurtrac(network, simtime, realclusters=None, lightphases=None, lastswitchtimes=None, predClusters=None):
    global clustersCache

    toSwitch = []
    catpreds = dict()
    remainingDuration = dict()
    bestschedules = dict()

    surtracThreads = dict()

    inRoutingSim = True
    if realclusters == None and lightphases == None:
        inRoutingSim = False
        if clustersCache == None:
            clustersCache = loadClusters(network, simtime)
        (realclusters, lightphases) = pickle.loads(pickle.dumps(clustersCache))

    #predCutoff
    if inRoutingSim:
        predictionCutoff = predCutoffMain #Routing
    else:
        predictionCutoff = predCutoffRouting #Main simulation
    

    if not predClusters == None:
        clusters = mergePredictions(realclusters, predClusters)
    else:
        clusters = pickle.loads(pickle.dumps(realclusters))

    for light in lights:
        if multithreadSurtrac:
            surtracThreads[light] = threading.Thread(target=doSurtracThread, args=(network, simtime, light, clusters, lightphases, lastswitchtimes, inRoutingSim, predictionCutoff, toSwitch, catpreds, bestschedules))
            surtracThreads[light].start()
        else:
            doSurtracThread(network, simtime, light, clusters, lightphases, lastswitchtimes, inRoutingSim, predictionCutoff, toSwitch, catpreds, bestschedules)

    for light in lights:
        if multithreadSurtrac:
            surtracThreads[light].join()

        #bestschedules gets re-created each call to doSurtrac, so it's not some weird carryover thing
        #print(light)
        #print(bestschedules[light])
        bestschedule = bestschedules[light]
        if not bestschedule[0] == []: #Check for the case of Surtrac seeing no vehicles (which would default to default fixed timing plans)
            spentDuration = simtime - lastswitchtimes[light]
            remainingDuration[light] = pickle.loads(pickle.dumps(bestschedule[7])) #TODO test whether this fixes the empty Surtrac schedules I'm getting. Seriously, that worked? WHY?! Also some (all?) lights are now 0 length, oops

            #NOTE: This was old and hopefully no longer necessary TODO delete
            # if len(remainingDuration[light]) > 0 and not testNN and not testDumbtrac: #Because the NN doesn't account for spent duration when returning [1]
            #     remainingDuration[light][0] -= spentDuration #TODO I think this is right but learned Surtrac is being weird and I don't know why
            
            if len(remainingDuration[light]) == 0:
                print('pretest - empty remainingDuration')
            if len(remainingDuration[light]) > 0:
                if remainingDuration[light][0] >= 0 and not inRoutingSim:
                    #Update duration
                    traci.trafficlight.setPhaseDuration(light, remainingDuration[light][0]) #setPhaseDuration sets the remaining duration in the phase
                
                if remainingDuration[light][0] <= 0: #Light needs to change
                    #Light needs to change
                    toSwitch.append(light)

                    curphase = lightphases[light]
                    nPhases = len(surtracdata[light]) #Number of phases

                    #If Surtrac tells a light to change, the phase duration should be within the allowed bounds
                    #Surtrac in routing (which uses larger timesteps) might exceed maxDur, but by less than the timestep
                    #TODO: Actually, might exceed but by no more than routing's surtracFreq - pipe surtracFreq into this function eventually?
                    # if not (simtime - lastswitchtimes[light] >= surtracdata[light][curphase]["minDur"] and simtime - lastswitchtimes[light] <= surtracdata[light][curphase]["maxDur"]+timestep):
                    #     print("Duration violation on light " + light + "; actual duration " + str(simtime - lastswitchtimes[light]))

                    #NEXT TODO how is this returning 0 anyway? Shouldn't it be returning 1 or something??
                    if simtime - lastswitchtimes[light] < surtracdata[light][curphase]["minDur"]:
                        print("Duration violation on light " + light + "; actual duration " + str(simtime - lastswitchtimes[light]) + " but min duration " + str(surtracdata[light][curphase]["minDur"]))
                    if simtime - lastswitchtimes[light] > surtracdata[light][curphase]["maxDur"]+timestep:
                        print("Duration violation on light " + light + "; actual duration " + str(simtime - lastswitchtimes[light]) + " but max duration " + str(surtracdata[light][curphase]["maxDur"]))

                    lightphases[light] = (curphase+1)%nPhases #This would change the light if we're in routing sim
                    lastswitchtimes[light] = simtime

                    remainingDuration[light].pop(0)

                    if len(remainingDuration[light]) == 0:
                        remainingDuration[light] = [lightphasedata[light][(lightphases[light]+1)%len(lightphasedata[light])].duration]

                    if not inRoutingSim: #Actually change the light
                        traci.trafficlight.setPhase(light, (curphase+1)%nPhases) #Increment phase, duration defaults to default
                        if len(remainingDuration[light]) > 0:
                            #And set the new duration if possible
                            traci.trafficlight.setPhaseDuration(light, remainingDuration[light][0]) #Update duration if we know it
                            #pass
            else:
                #NEXT TODO this is going off with learned actuated control (and maybe others) the first time after doSurtrac returns 0 and the light switches. Unclear why.
                #print("test")
                #print(bestschedule)
                print("AAAAAAHHHHH! Surtrac's giving back an empty schedule!")


    #Predict-ahead for everything else; assume no delays.
    if not disableSurtracPred:
        for light in notLights:
            for lane in notlightlanes[light]:
                if not lane in turndata:
                    continue
                edge = lane.split("_")[0]

                #This might not merge cars coming from different lanes in chronological order. Is this a problem?
                
                for clusterind in range(len(clusters[lane])): #TODO: This is going to be really bad at chronological ordering. Should probably just recompute clusters at the end
                    predLanes = []
                    for outlane in turndata[lane]: #turndata has psuedocounts so everything should be fine here?
                    
                        ist = clusters[lane][clusterind]["arrival"] #Intended start time = cluster arrival time
                        dur = clusters[lane][clusterind]["departure"] - ist
                        
                        arr = ist + fftimes[outlane]
                        if arr > simtime + predictionCutoff:
                            #Cluster is farther in the future than we want to predict; skip it
                            continue
                        newPredCluster = dict()
                        newPredCluster["endpos"] = 0
                        newPredCluster["time"] = ist
                        newPredCluster["arrival"] = arr
                        newPredCluster["departure"] = newPredCluster["arrival"] + dur
                        newPredCluster["cars"] = []
                        newPredCluster["weight"] = 0
                        if outlane in catpreds:
                            catpreds[outlane].append(newPredCluster)
                        else:
                            catpreds[outlane] = [newPredCluster]
                        predLanes.append(outlane) #Track which lanes' clusters are within the prediction cutoff
                        assert(len(catpreds[outlane]) >= 1)
                    #Add cars to new clusters
                    for cartuple in clusters[lane][clusterind]["cars"]:
                        #cartuple[0] is name of car; cartuple[1] is departure time; cartuple[2] is debug info
                        #assert(cartuple[0] in isSmart)
                        if not cartuple[0] in isSmart or isSmart[cartuple[0]]: #It's possible we call this from QueueSim, at which point we split the vehicle being routed and wouldn't recognize the new names. Anything else should get assigned to isSmart or not on creation
                            route = currentRoutes[cartuple[0].split("|")[0].split("_")[0]] #.split to deal with the possibility of splitty cars in QueueSim
                            if not edge in route:
                                #Not sure if or why this happens - maybe the route is changing and predictions aren't updating?
                                #Can definitely happen for a splitty car inside QueueSim
                                #Regardless, don't predict this car forward and hope for the best?
                                if not "|" in cartuple[0]:
                                    pass
                                    #TODO: This still happens sometimes, not sure why
                                    #print("Warning, smart car on an edge that's not in its route. Assuming a mispredict and removing")
                                #TODO: Else should predict it goes everywhere? Does this happen??
                                continue
                            edgeind = route.index(edge)
                            if edgeind+1 == len(route):
                                #At end of route, don't care
                                continue
                            nextedge = route[edgeind+1]

                            if not nextedge in normprobs[lane]:
                                #Might be happening if the car needs to make a last-minute lane change to stay on its route?
                                #TODO: Find a lane where it can continue with the route and go from there? Ignoring for now
                                #NEXT TODO: We're splitting the initial vehicle onto all starting lanes; this shouldn't be happening at all anymore. Verify?
                                #This seems fine, but similar code in doSurtracThread is still triggering. This is the code for prediction through non-lights, though, which probably just doesn't come up often enough for me to notice a problem. Probably still bad.
                                ##print("Warning, no data, having Surtrac prediction ignore this car instead of making something up")
                                #print(lane)
                                #print("normprob == 0, something's probably wrong??")
                                continue

                            for nextlaneind in range(lanenums[nextedge]):
                                nextlane = nextedge+"_"+str(nextlaneind)
                                modcartuple = (cartuple[0], cartuple[1]+fftimes[nextlane], cartuple[2]*turndata[lane][nextlane] / normprobs[lane][nextedge], cartuple[3])
                                if nextlane in predLanes:
                                    #Make sure we're predicting this cluster
                                    catpreds[nextlane][-1]["cars"].append(modcartuple)
                                    catpreds[nextlane][-1]["weight"] += modcartuple[2]

                        else:
                            for nextlane in predLanes:
                                modcartuple = (cartuple[0], cartuple[1]+fftimes[nextlane], cartuple[2]*turndata[lane][nextlane], cartuple[3])
                                catpreds[nextlane][-1]["cars"].append(modcartuple)
                                catpreds[nextlane][-1]["weight"] += modcartuple[2]

                    for outlane in predLanes:
                        if catpreds[outlane][-1]["weight"] == 0:
                            #Remove predicted clusters that are empty
                            catpreds[outlane].pop(-1)
                            continue

                        if len(catpreds[outlane]) >=2 and catpreds[outlane][-1]["arrival"] - catpreds[outlane][-2]["departure"] < clusterthresh:
                            #Merge this cluster with the previous one
                            #Pos and time don't do anything here
                            #Arrival doesn't change - previous cluster arrived first
                            catpreds[outlane][-2]["departure"] = max(catpreds[outlane][-2]["departure"], catpreds[outlane][-1]["departure"])
                            catpreds[outlane][-2]["cars"] += catpreds[outlane][-1]["cars"] # += concatenates
                            catpreds[outlane][-2]["weight"] += catpreds[outlane][-1]["weight"]
                            catpreds[outlane].pop(-1)

        #Confirm that weight of cluster = sum of weights of cars
        for lane in catpreds:
            for preind in range(len(catpreds[lane])):
                weightsum = 0
                for ind in range(len(catpreds[lane][preind]["cars"])):
                    weightsum += catpreds[lane][preind]["cars"][ind][2]
                assert(abs(weightsum - catpreds[lane][preind]["weight"]) < 1e-10)

    return (toSwitch, catpreds, remainingDuration)

#For computing free-flow time, which is used for computing delay. Stolen from my old A* code, where it was a heuristic function.
def backwardDijkstra(network, goal):
    goalcost = lengths[goal+"_0"]/network.getEdge(goal).getSpeed()
    gvals = dict()
    gvals[goal] = goalcost
    pq = []
    heappush(pq, (goalcost, goal)) #Cost to traverse from start of goal to end of goal is goalcost, not 0

    prevgval = goalcost
    while len(pq) > 0: #When the queue is empty, we're done
        #print(pq)
        stateToExpand = heappop(pq)
        #Sanity check: Things we pop off the heap should keep getting bigger
        assert(prevgval <= stateToExpand[0])
        prevgval = stateToExpand[0]
        #fval = stateToExpand[0]
        edge = stateToExpand[1]
        gval = gvals[edge]

        #Get predecessor IDs
        succs = []
        for succ in list(network.getEdge(edge).getIncoming()):
            succs.append(succ.getID())
        
        for succ in succs:
            c = lengths[edge+"_0"]/network.getEdge(edge).getSpeed()

            h = 0 #Heuristic not needed here - search is fast
            if succ in gvals and gvals[succ] <= gval+c:
                #Already saw this state, don't requeue
                continue

            #NOTE: If we found a better way to reach succ, the old way is still in the queue
            #This is fine, though, since the previous check ensures we don't requeue bad stuff afterwards
            #TODO eventually: Uncomment this to avoid that? Shouldn't matter much though, this function is fast and doesn't run many times
            if succ in gvals and gvals[succ] > gval+c:
                pq.remove((gvals[succ], succ))
                heapify(pq)

            #Otherwise it's new or we're now doing better, so requeue it
            gvals[succ] = gval+c
            heappush(pq, (gval+c+h, succ))
    return gvals
                    
def run(network, rerouters, pSmart, verbose = True):
    global sumoPredClusters
    global currentRoutes
    global hmetadict
    global delay3adjdict
    global actualStartDict
    global locDict
    global laneDict
    global clustersCache
    #netfile is the filepath to the network file, so we can call sumolib to get successors
    #rerouters is the list of induction loops on edges with multiple successor edges
    
    startDict = dict()
    endDict = dict()
    delayDict = dict()
    delay2adjdict = dict()
    delay3adjdict = dict()
    locDict = dict()
    laneDict = dict()
    leftDict = dict()
    carsOnNetwork = []
    remainingDuration = dict()

    tstart = time.time()
    simtime = 0

    while traci.simulation.getMinExpectedNumber() > 0 and (not appendTrainingData or simtime < 5000):
        simtime += 1
        traci.simulationStep() #Tell the simulator to simulate the next time step
        clustersCache = None #Clear stored clusters list

        #Check for lights that switched phase (because previously-planned duration ran out, not because Surtrac etc. changed the plan); update custom data structures and current phase duration
        for light in lights:
            temp = traci.trafficlight.getPhase(light)
            if not(light in remainingDuration and len(remainingDuration[light]) > 0):
                #Only update remainingDuration if we have no schedule, in which case grab the actual remaining duration from SUMO
                remainingDuration[light] = [traci.trafficlight.getNextSwitch(light) - simtime]
            else:
                remainingDuration[light][0] -= 1
            if temp != lightphases[light]:
                mainlastswitchtimes[light] = simtime
                lightphases[light] = temp
                #Duration of previous phase was first element of remainingDuration, so pop that and read the next, assuming everything exists
                if light in remainingDuration and len(remainingDuration[light]) > 0:
                    #print(remainingDuration[light][0]) #Prints -1. Might be an off-by-one somewhere, but should be pretty close to accurate?
                    #NOTE: The light switches when remaining duration goes negative (in this case -1)
                    remainingDuration[light].pop(0)
                    if len(remainingDuration[light]) == 0:
                        remainingDuration[light] = [traci.trafficlight.getNextSwitch(light) - simtime]
                    else:
                        traci.trafficlight.setPhaseDuration(light, remainingDuration[light][0])
                else:
                    print("Unrecognized light " + light + ", this shouldn't happen")
        
        realdurations[simtime] = pickle.loads(pickle.dumps(remainingDuration))
        # if simtime in simdurations:
        #     print("DURRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRR")
        #     print(simdurations[simtime][lights[0]])
        #     print(realdurations[simtime][lights[0]])

        dontReroute = []

        #Decide whether new vehicles use our routing
        for vehicle in traci.simulation.getDepartedIDList():
            isSmart[vehicle] = random.random() < pSmart
            if isSmart[vehicle]:
                traci.vehicle.setColor(vehicle, [0, 255, 0, 255])
            else:
                traci.vehicle.setColor(vehicle, [255, 0, 0, 255])
            timedata[vehicle] = [simtime, -1, -1, 'unknown', 'unknown']
            currentRoutes[vehicle] = traci.vehicle.getRoute(vehicle)
            routeStats[vehicle] = dict()
            routeStats[vehicle]["nCalls"] = 0
            routeStats[vehicle]["nCallsFirst"] = 0
            routeStats[vehicle]["nCallsAfterFirst"] = 0
            routeStats[vehicle]["nSwaps"] = 0
            routeStats[vehicle]["nSwapsFirst"] = 0
            routeStats[vehicle]["nSwapsAfterFirst"] = 0
            routeStats[vehicle]["swapped"] = False
            routeStats[vehicle]["nTimeouts"] = 0
            routeStats[vehicle]["nTeleports"] = 0
            routeStats[vehicle]["distance"] = 0

            goaledge = currentRoutes[vehicle][-1]
            if not goaledge in hmetadict:
                hmetadict[goaledge] = backwardDijkstra(network, goaledge)
            delayDict[vehicle] = -hmetadict[goaledge][currentRoutes[vehicle][0]] #I'll add the actual travel time once the vehicle arrives
            laneDict[vehicle] = traci.vehicle.getLaneID(vehicle)
            currentRoutes[vehicle] = traci.vehicle.getRoute(vehicle)

            startDict[vehicle] = simtime
            locDict[vehicle] = traci.vehicle.getRoadID(vehicle)
            laneDict[vehicle] = traci.vehicle.getLaneID(vehicle)
            leftDict[vehicle] = 0

            lane = laneDict[vehicle]
            if not lane in arrivals:
                arrivals[lane] = []
            arrivals[lane].append(simtime) #Don't care who arrived, just when they arrived

        #Check predicted vs. actual travel times
        for vehicle in traci.simulation.getArrivedIDList():
            if isSmart[vehicle]:
                timedata[vehicle][1] = simtime
                # print("Actual minus expected: %f" % ((timedata[vehicle][1]-timedata[vehicle][0]) - timedata[vehicle][2]))
                # print("Actual : %f" % (timedata[vehicle][1]-timedata[vehicle][0]))
                # print("Expected : %f" % (timedata[vehicle][2]))
                # print("Percent error : %f" % ( ((timedata[vehicle][1]-timedata[vehicle][0]) - timedata[vehicle][2]) / (timedata[vehicle][1]-timedata[vehicle][0]) * 100))
                # print("Route from " + timedata[vehicle][3] + " to " + timedata[vehicle][4])
            endDict[vehicle] = simtime
            locDict.pop(vehicle)
            laneDict.pop(vehicle)
            dontReroute.append(vehicle) #Vehicle has left network and does not need to be rerouted

        surtracFreq = mainSurtracFreq #Period between updates in main SUMO sim
        if simtime%surtracFreq >= (simtime+1)%surtracFreq:
            temp = doSurtrac(network, simtime, None, None, mainlastswitchtimes, sumoPredClusters)
            #Don't bother storing toUpdate = temp[0], since doSurtrac has done that update already
            sumoPredClusters = temp[1]
            remainingDuration.update(temp[2])

        vehiclesOnNetwork = traci.vehicle.getIDList()
        carsOnNetwork.append(len(vehiclesOnNetwork)) #Store number of cars on network (for plotting)

        #Count left turns
        for id in laneDict:
            newlane = traci.vehicle.getLaneID(id)
            if len(newlane) == 0 or newlane[0] == ":":
                dontReroute.append(id) #Vehicle is mid-intersection or off network, don't try to reroute them
            if newlane != laneDict[id] and len(newlane) > 0 and  newlane[0] != ":":
                newloc = traci.vehicle.getRoadID(id)
                c0 = network.getEdge(locDict[id]).getFromNode().getCoord()
                c1 = network.getEdge(locDict[id]).getToNode().getCoord()
                theta0 = math.atan2(c1[1]-c0[1], c1[0]-c0[0])

                c2 = network.getEdge(newloc).getToNode().getCoord()
                theta1 = math.atan2(c2[1]-c1[1], c2[0]-c1[0])

                if (theta1-theta0+math.pi)%(2*math.pi)-math.pi > 0:
                    leftDict[id] += 1
                laneDict[id] = newlane
                locDict[id] = newloc

                #Remove vehicle from predictions, since the next intersection should actually see it now
                if not disableSurtracPred:
                    for predlane in sumoPredClusters:
                        for predcluster in sumoPredClusters[predlane]:

                            predcarind = 0
                            minarr = inf
                            maxarr = -inf
                            while predcarind < len(predcluster["cars"]):
                                predcartuple = predcluster["cars"][predcarind]
                                if predcartuple[0] == id:
                                    predcluster["cars"].pop(predcarind)
                                    predcluster["weight"] -= predcartuple[2]
                                else:
                                    predcarind += 1
                                    if predcartuple[1] < minarr:
                                        minarr = predcartuple[1]
                                    if predcartuple[1] > maxarr:
                                        maxarr = predcartuple[1]
                            if len(predcluster["cars"]) == 0:
                                sumoPredClusters[predlane].remove(predcluster)
                            else:
                                pass
                                #predcluster["arrival"] = minarr #predcluster["cars"][0][1]
                                #predcluster["departure"] = maxarr #predcluster["cars"][-1][1]

                            weightsum = 0
                            for predcarind in range(len(predcluster["cars"])):
                                weightsum += predcluster["cars"][predcarind][2]
                            assert(abs(weightsum - predcluster["weight"]) < 1e-10)

                #Store data to compute delay after first intersection
                if not id in delay2adjdict:
                    delay2adjdict[id] = simtime

                #Compute distance travelled if on last edge of route (since we can't do this once we leave the network)
                if newlane.split("_")[0] == currentRoutes[id][-1]:
                    routeStats[id]["distance"] = traci.vehicle.getDistance(id) + lengths[newlane]

        
        for car in traci.simulation.getStartingTeleportIDList():
            routeStats[car]["nTeleports"] += 1
            print("Warning: Car " + car + " teleported, time=" + str(simtime))

        #Moving this to the bottom so we've already updated the vehicle locations (when we checked left turns)
        reroute(rerouters, network, simtime, remainingDuration) #Reroute cars (including simulate-ahead cars)

        #Plot and print stats
        if simtime%100 == 0 or not traci.simulation.getMinExpectedNumber() > 0:
            #After we're done simulating... 
            plt.figure()
            plt.plot(carsOnNetwork)
            plt.xlabel("Time (s)")
            plt.ylabel("Cars on Network")
            plt.title("Congestion, Adoption Prob=" + str(pSmart))
            #plt.show() #NOTE: Blocks code execution until you close the plot
            plt.savefig("Plots/Congestion, AP=" + str(pSmart)+".png")
            plt.close()


            #Stats
            avgTime = 0
            avgLefts = 0
            bestTime = inf
            worstTime = 0

            avgTimeSmart = 0
            avgLeftsSmart = 0
            bestTimeSmart = inf
            worstTimeSmart = 0
            avgTimeNot = 0
            avgLeftsNot = 0
            bestTimeNot = inf
            worstTimeNot = 0

            totalcalls = 0
            totalcallsafterfirst = 0
            totalcallsfirst = 0
            totalswaps = 0
            totalswapsafterfirst = 0
            totalswapsfirst = 0
            nswapped = 0

            avgTime2 = 0
            avgTimeSmart2 = 0
            avgTimeNot2 = 0

            avgTime3 = 0
            avgTimeSmart3 = 0
            avgTimeNot3 = 0

            avgTime0 = 0
            avgTimeSmart0 = 0
            avgTimeNot0 = 0

            nCars = 0
            nSmart = 0
            ntimeouts = 0
            nsmartteleports = 0
            nnotsmartteleports = 0
            nteleports = 0

            avgerror = 0
            avgabserror = 0
            avgpcterror = 0
            avgabspcterror = 0

            totaldistance = 0
            totaldistanceSmart = 0
            totaldistanceNot = 0

            for id in endDict:
                if actualStartDict[id] >= 600 and actualStartDict[id] < 3000:
                    nCars += 1
                    if isSmart[id]:
                        nSmart += 1

            for id in endDict:
                #Only look at steady state - ignore first and last 10 minutes of cars
                if actualStartDict[id] < 600 or actualStartDict[id] >= 3000:
                    continue

                ntimeouts += routeStats[id]["nTimeouts"]
                nteleports += routeStats[id]["nTeleports"]
                if isSmart[id]:
                    nsmartteleports += routeStats[id]["nTeleports"]
                else:
                    nnotsmartteleports += routeStats[id]["nTeleports"]

                ttemp = (endDict[id] - startDict[id])+delayDict[id]
                avgTime += ttemp/nCars
                avgLefts += leftDict[id]/nCars
                if ttemp > worstTime:
                    worstTime = ttemp
                if ttemp < bestTime:
                    bestTime = ttemp

                if ttemp < 0:
                    print("Negative ttemp (=delay)?")
                    print(id)

                if isSmart[id]:
                    avgTimeSmart += ttemp/nSmart
                    avgLeftsSmart += leftDict[id]/nSmart
                    if ttemp > worstTimeSmart:
                        worstTimeSmart = ttemp
                    if ttemp < bestTimeSmart:
                        bestTimeSmart = ttemp
                else:
                    avgTimeNot += ttemp/(nCars-nSmart)
                    avgLeftsNot += leftDict[id]/(nCars-nSmart)
                    if ttemp > worstTimeNot:
                        worstTimeNot = ttemp
                    if ttemp < bestTimeNot:
                        bestTimeNot = ttemp

                #Delay2 computation (start clock after first intersection)
                if not id in delay2adjdict:
                    delay2adjdict[id] = startDict[id]
                ttemp2 = (endDict[id] - delay2adjdict[id])+delayDict[id]
                avgTime2 += ttemp2/nCars
                if isSmart[id]:
                    avgTimeSmart2 += ttemp2/nSmart
                else:
                    avgTimeNot2 += ttemp2/(nCars-nSmart)

                #Delay3 computation (start clock after first routing call)
                if not id in delay3adjdict:
                    delay3adjdict[id] = startDict[id]
                ttemp3 = (endDict[id] - delay3adjdict[id])+delayDict[id]
                avgTime3 += ttemp3/nCars
                if isSmart[id]:
                    avgTimeSmart3 += ttemp3/nSmart
                else:
                    avgTimeNot3 += ttemp3/(nCars-nSmart)

                #Delay0 computation (start clock at intended entrance time)
                ttemp0 = (endDict[id] - actualStartDict[id])+delayDict[id]
                avgTime0 += ttemp0/nCars
                if isSmart[id]:
                    avgTimeSmart0 += ttemp0/nSmart
                else:
                    avgTimeNot0 += ttemp0/(nCars-nSmart)

                totalcalls += routeStats[id]["nCalls"]
                totalcallsafterfirst += routeStats[id]["nCallsAfterFirst"]
                totalcallsfirst += routeStats[id]["nCallsFirst"]
                totalswaps += routeStats[id]["nSwaps"]
                totalswapsafterfirst += routeStats[id]["nSwapsAfterFirst"]
                totalswapsfirst += routeStats[id]["nSwapsFirst"]
                totaldistance += routeStats[id]["distance"]
                if isSmart[id]:
                    totaldistanceSmart += routeStats[id]["distance"]
                else:
                    totaldistanceNot += routeStats[id]["distance"]
                #Check which routes don't run into routing decisions at all
                # if isSmart[id] and routeStats[id]["nSwapsFirst"] == 0:
                #     print(currentRoutes[id])
                if routeStats[id]["swapped"] == True:
                    nswapped += 1

                if isSmart[id]:
                    avgerror += ((timedata[id][1]-timedata[id][0]) - timedata[id][2])/nSmart
                    avgabserror += abs((timedata[id][1]-timedata[id][0]) - timedata[id][2])/nSmart
                    avgpcterror += ((timedata[id][1]-timedata[id][0]) - timedata[id][2])/(timedata[id][1]-timedata[id][0])/nSmart*100
                    avgabspcterror += abs((timedata[id][1]-timedata[id][0]) - timedata[id][2])/(timedata[id][1]-timedata[id][0])/nSmart*100
        
            if verbose or not traci.simulation.getMinExpectedNumber() > 0 or (appendTrainingData and simtime == 5000):
                print(pSmart)
                print("\nCurrent simulation time: %f" % simtime)
                print("Total run time: %f" % (time.time() - tstart))
                print("Number of vehicles in network: %f" % traci.vehicle.getIDCount())
                print("Average delay: %f" % avgTime)
                print("Average delay0: %f" % avgTime0)
                print("Best delay: %f" % bestTime)
                print("Worst delay: %f" % worstTime)
                print("Average number of lefts: %f" % avgLefts)
                if nCars > 0:
                    print("Average number of calls to routing: %f" % (totalcalls/nCars))
                    if totalcalls > 0:
                        print("Proportion of timeouts in routing: %f" % (ntimeouts/totalcalls))
                    print("Average number of route changes: %f" % (totalswaps/nCars))
                    print("Average number of route changes after first routing decision: %f" % (totalswapsafterfirst/nCars))
                    print("Proportion of cars that changed route at least once: %f" % (nswapped/nCars))
                    print("Average number of teleports: %f" % (nteleports/nCars))
                    print("Average distance travelled: %f" % (totaldistance/nCars))
                print("Among adopters:")
                print("Average delay: %f" % avgTimeSmart)
                print("Best delay: %f" % bestTimeSmart)
                print("Worst delay: %f" % worstTimeSmart)
                print("Average number of lefts: %f" % avgLeftsSmart)
                if nSmart > 0:
                    print("Average error (actual minus expected) in predicted travel time: %f" % (avgerror))
                    print("Average absolute error in predicted travel time: %f" % (avgabserror))
                    print("Average percent error in predicted travel time: %f" % (avgpcterror))
                    print("Average absolute percent error in predicted travel time: %f" % (avgabspcterror))

                    print("Average number of calls to routing: %f" % (totalcalls/nSmart))
                    print("Average number of route changes: %f" % (totalswaps/nSmart))
                    print("Average number of route changes after first routing decision: %f" % (totalswapsafterfirst/nSmart))
                    if totalcalls > 0:
                        print("Proportion of timeouts in routing: %f" % (ntimeouts/totalcalls))
                        print("Proportion of routing decisions leading to a route change: %f" % (totalswaps/totalcalls))
                        if totalcallsfirst > 0:
                            print("Proportion of first routing decisions leading to a route change: %f" % (totalswapsfirst/totalcallsfirst))
                        else:
                            print("WARNING: Some routing calls, but no first routing calls; something's wrong with the stats!")
                        if totalcallsafterfirst > 0:
                            print("Proportion of routing decisions after first leading to a route change: %f" % (totalswapsafterfirst/totalcallsafterfirst))
                    print("Proportion of cars that changed route at least once: %f" % (nswapped/nSmart))
                    print("Average number of teleports: %f" % (nsmartteleports/nSmart))
                    print("Average distance travelled: %f" % (totaldistanceSmart/nSmart))
                print("Among non-adopters:")
                print("Average delay: %f" % avgTimeNot)
                print("Best delay: %f" % bestTimeNot)
                print("Worst delay: %f" % worstTimeNot)
                print("Average number of lefts: %f" % avgLeftsNot)
                if nCars - nSmart > 0:
                    print("Average number of teleports: %f" % (nnotsmartteleports/(nCars-nSmart)))
                    print("Average distance travelled: %f" % (totaldistanceNot/(nCars-nSmart)))
                #print(len(nRight)/1200)

                for lane in arrivals:
                    testlane = lane
                    break

                for lane in arrivals:
                    while len(arrivals[lane]) > 0 and arrivals[lane][0] < simtime - maxarrivalwindow:
                        arrivals[lane] = arrivals[lane][1:]

                # timeperarrival = min(simtime, maxarrivalwindow)/len(arrivals[testlane])
                # print(testlane)
                # print(timeperarrival)
    return [avgTime, avgTimeSmart, avgTimeNot, avgTime2, avgTimeSmart2, avgTimeNot2, avgTime3, avgTimeSmart3, avgTimeNot3, avgTime0, avgTimeSmart0, avgTimeNot0]

    

#Tell all the detectors to reroute the cars they've seen
#@profile
def reroute(rerouters, network, simtime, remainingDuration):
    global toReroute
    global threads
    global nToReroute
    global killSurtracThread

    toReroute = []
    reroutedata = dict()
    nToReroute = 0

    #Reuse Surtrac schedule between timesteps
    if recomputeRoutingSurtracFreq <= 1 or simtime%recomputeRoutingSurtracFreq >= (simtime+1)%recomputeRoutingSurtracFreq:
        killSurtracThread = True
        if "Surtrac" in threads:
            threads["Surtrac"].join()

    for r in rerouters:
        QueueReroute(r, network, reroutedata, simtime, remainingDuration)

    for vehicle in toReroute:
        if multithreadRouting:
            threads[vehicle].join() #Gets stuck when reusing Surtrac if the times we need Surtrac data for don't match what we've saved due to timestep being big. Hopefully fixed now.
            #threads.pop(vehicle) #Shouldn't be necessary, but would make sure threads only contains Surtrac and stuff in toReroute
        data = reroutedata[vehicle]
        
        newroute = data[0]

        routeStats[vehicle]["nCalls"] += 1
        if timedata[vehicle][2] == -1:
            routeStats[vehicle]["nCallsFirst"] += 1
        else:
            routeStats[vehicle]["nCallsAfterFirst"] += 1 #Not necessarily nCalls-1; want to account for vehicles that never got routed

        if not tuple(newroute) == currentRoutes[vehicle] and not newroute == currentRoutes[vehicle][-len(newroute):]:
            routeStats[vehicle]["nSwaps"] += 1
            routeStats[vehicle]["swapped"] = True
            if timedata[vehicle][2] == -1:
                routeStats[vehicle]["nSwapsFirst"] += 1
            else:
                routeStats[vehicle]["nSwapsAfterFirst"] += 1

        tcluster = data[1]
        if timedata[vehicle][2] == -1:
            timedata[vehicle][0] = simtime #Time prediction was made
            #timedata[vehicle][1] is going to be actual time at goal
            timedata[vehicle][2] = tcluster #Predicted time until goal
            timedata[vehicle][3] = currentRoutes[vehicle][0]
            timedata[vehicle][4] = currentRoutes[vehicle][-1]
        
        
        traci.vehicle.setRoute(vehicle, newroute)
        assert(newroute[-1] == currentRoutes[vehicle][-1])
        currentRoutes[vehicle] = newroute

#@profile
def QueueReroute(detector, network, reroutedata, simtime, remainingDuration):
    global toReroute
    global threads
    global delay3adjdict
    global nToReroute
    global killSurtracThread
    global clustersCache

    ids = traci.inductionloop.getLastStepVehicleIDs(detector) #All vehicles to be rerouted
    if len(ids) == 0:
        #No cars to route, we're done here
        return

    edge = rerouterEdges[detector]

    for vehicle in ids:
        if vehicle in dontReroute:
            #Mid-intersection or off network, don't route this
            continue
        if locDict[vehicle] != edge:
            print("Warning: Vehicle triggered detector on road " + edge + " but then left the road. Possible really short edge right before intersection??")
            continue

        if detector in oldids and vehicle in oldids[detector]:
            #Just routed this, don't try again
            continue

        if vehicle not in delay3adjdict:
            delay3adjdict[vehicle] = simtime

        if isSmart[vehicle]:
            #Convert current state
            toReroute.append(vehicle)
            reroutedata[vehicle] = [None]*2
            if clustersCache == None:
                clustersCache = loadClusters(network, simtime, vehicle)
            loaddata = pickle.loads(pickle.dumps(clustersCache))

            #Store routes once at the start to save time
            routes = pickle.loads(pickle.dumps(currentRoutes))

            for vehicletemp in vehiclesOnNetwork:
                if not isSmart[vehicletemp]:
                    #Sample random routes for non-adopters
                    routes[vehicletemp] = sampleRouteFromTurnData(vehicletemp, laneDict[vehicletemp], turndata)

            #Prepare to route
            nToReroute += 1
            if reuseSurtrac:
                if killSurtracThread:
                    killSurtracThread = False
                    #This is the first vehicle being routed; start a Surtrac thread using these routes
                    threads["Surtrac"] = threading.Thread(target=doClusterSimThreaded, args=(laneDict[vehicle], network, [], simtime, remainingDuration, reroutedata[vehicle], pickle.loads(pickle.dumps(loaddata)), routes))
                    threads["Surtrac"].start()

            if multithreadRouting:
                #print("Starting vehicle routing thread")
                threads[vehicle] = threading.Thread(target=doClusterSimThreaded, args=(laneDict[vehicle], network, vehicle, simtime, remainingDuration, reroutedata[vehicle], pickle.loads(pickle.dumps(loaddata)), routes))
                threads[vehicle].start()
            else:
                doClusterSimThreaded(laneDict[vehicle], network, vehicle, simtime, remainingDuration, reroutedata[vehicle], pickle.loads(pickle.dumps(loaddata)), routes) #If we want non-threaded

    oldids[detector] = ids

#@profile
def doClusterSimThreaded(prevlane, net, vehicle, simtime, remainingDuration, data, loaddata, routes):
    global nRoutingCalls
    global routingTime
    starttime = time.time()
    temp = runClusters(net, simtime, remainingDuration, vehicle, prevlane, loaddata, routes)
    nRoutingCalls += 1
    routingTime += time.time() - starttime
    for i in range(len(temp)):
        data[i] = temp[i]

#@profile
def loadClusters(net, simtime, VOI=None):
    #Load locations of cars and current traffic light states into custom data structures
    #If given, VOI is the vehicle triggering the routing call that triggered this, and needs to be unaffected when we add noise
    #TODO: We're caching the loaded clusters, which means we'll need to be better about not adding noise to any vehicles that could potentially be routed
    lightphases = dict()
    clusters = dict()

    #Cluster data structures
    for edge in edges:
        if edge[0] == ":":
            #Skip internal edges (=edges for the inside of each intersection)
            continue
        for lanenum in range(lanenums[edge]):
            lane = edge + "_" + str(lanenum)
            clusters[lane] = []
            for vehicle in reversed(traci.lane.getLastStepVehicleIDs(lane)): #Reversed so we go from end of edge to start of edge - first clusters to leave are listed first
                
                #Process vehicle into cluster somehow
                #If nearby cluster, add to cluster in sorted order (could probably process in sorted order)
                lanepos = traci.vehicle.getLanePosition(vehicle)
                if len(clusters[lane]) > 0 and abs(clusters[lane][-1]["time"] - simtime) < clusterthresh and abs(clusters[lane][-1]["endpos"] - lanepos)/speeds[edge] < clusterthresh:
                    #Add to cluster. pos and time track newest added vehicle to see if the next vehicle merges
                    #Departure time (=time to fully clear cluster) increases, arrival doesn't
                    clusters[lane][-1]["endpos"] = lanepos
                    clusters[lane][-1]["time"] = simtime
                    clusters[lane][-1]["departure"] = simtime + (lengths[lane]-clusters[lane][-1]["endpos"])/speeds[edge]
                    clusters[lane][-1]["cars"].append((vehicle, clusters[lane][-1]["departure"], 1, "Load append"))
                    clusters[lane][-1]["weight"] = len(clusters[lane][-1]["cars"])
                else:
                    #Else make a new cluster
                    newcluster = dict()
                    newcluster["startpos"] = lanepos
                    newcluster["endpos"] = lanepos
                    newcluster["time"] = simtime
                    newcluster["arrival"] = simtime + (lengths[edge+"_0"]-newcluster["endpos"])/speeds[edge]
                    newcluster["departure"] = newcluster["arrival"]
                    newcluster["cars"] = [(vehicle, newcluster["departure"], 1, "Load new")]
                    newcluster["weight"] = len(newcluster["cars"])
                    clusters[lane].append(newcluster)
                assert(clusters[lane][-1]["departure"] > simtime - 1e-10)
    
    #Traffic light info
    lightphases = dict()
    for light in lights:
        lightphases[light] = traci.trafficlight.getPhase(light)

    #Add noise
    #clusters = addNoise(clusters, VOI, 0.9, 2) #To simulate detector stuff
    return (clusters, lightphases)

def addNoise(clusters, VOI, detectprob, timeerr):
    #Randomly delete cars with probability noiseprob
    #Randomly clone non-deleted cars to make up for it

    if reuseSurtrac:
        print("Warning: We might be randomly deleting a vehicle to be routed later. TODO fix this...")

    #Cluster data structures
    for edge in edges:
        if edge[0] == ":":
            #Skip internal edges (=edges for the inside of each intersection)
            continue
        for lanenum in range(lanenums[edge]):
            lane = edge + "_" + str(lanenum)
            for clusternum in range(len(clusters[lane])):
                cluster = clusters[lane][clusternum]
                noisycluster = pickle.loads(pickle.dumps(cluster))
                noisycluster["cars"] = []
                for car in cluster["cars"]:
                    if car[0] == VOI:
                        #Don't perturb the VOI
                        noisycluster["cars"].append(car)
                        continue
                    noisecar = (car[0], car[1] + random.random()*timeerr*2-timeerr, car[2], "noisetest") #Because tuples are immutable...
                    if random.random() < 1-detectprob:
                        #Don't add car to noisycluster
                        noisycluster["weight"] -= noisecar[2]
                        continue
                    noisycluster["cars"].append(noisecar)
                    #Duplicate car, potentially multiple times, to estimate deleted traffic
                    while random.random() < 1-detectprob:
                        #Duplicate this car
                        noisycluster["cars"].append(noisecar)
                        noisycluster["weight"] += noisecar[2]
                clusters[lane][clusternum] = noisycluster
    return clusters


#NOTE: Multithreaded stuff doesn't get profiled...
#@profile
def runClusters(net, routesimtime, mainRemainingDuration, vehicleOfInterest, startlane, loaddata, routes):
    global surtracDict
    global nToReroute
    global killSurtracThread
    global newcarcounter

    #Fix routesimtime before we initialize the list of things to check, else we might be running Surtrac at slightly different times when we try to reuse schedules
    starttime = routesimtime
    if reuseSurtrac and recomputeRoutingSurtracFreq > 1:
        routesimtime = math.floor(routesimtime/timestep)*timestep

    computeSurtrac = len(vehicleOfInterest) == 0
    if computeSurtrac:
        surtracDict = dict()
    
    startedge = startlane.split("_")[0]

    if not computeSurtrac:
        goalEdge = routes[vehicleOfInterest][-1]
    splitinfo = dict()
    VOIs = [vehicleOfInterest]

    clusters = loaddata[0]
    lightphases = loaddata[1]
    
    finishedLanes = dict()

    lastDepartTime = dict()

    lanesToCheck = dict()
    lanesToCheck[routesimtime+timestep] = [] #+timestep because we increment time before doing our first checks

    edgelist = list(edges)
    edgeind = 0
    while edgeind < len(edgelist):
        if edgelist[edgeind][0] == ":":
            edgelist.pop(edgeind)
        else:
            edgeind += 1

    for edge in edgelist:
        for lanenum in range(lanenums[edge]):
            lane = edge + "_" + str(lanenum)
            lastDepartTime[lane] = -inf
            lanesToCheck[routesimtime+timestep].append(lane) #+timestep because we increment time before doing our first checks

    #Split initial VOI into all starting lanes
    if not computeSurtrac:
        startlanenum = int(startlane.split("_")[1])
        for lanenum in range(0, lanenums[startedge]):
            if lanenum == startlanenum:
                #Skip the original vehicle of interest
                continue
            #Clone original vehicle of interest into ALL the lanes
            lane = startedge+"_"+str(lanenum)
            startdist = lengths[lane]-detectordist
            #If no cluster covers len-50m, make a new one; else add to end of the one that does 
            VOIadded = False
            insertBeforeEnd = False
            vehicle = vehicleOfInterest+"_"+str(lanenum) #Name of copy of VOI to add
            VOIs.append(vehicle)
            for clusterind in range(len(clusters[lane])): #Loop over all existing clusters in the lane, looking for a reasonable one to append to
                cluster = clusters[lane][clusterind]
                assert(cluster["startpos"] >= cluster["endpos"]) #Startpos is position of first car to leave, which is closest to end of edge, but 0 is start of edge, so startpos should be larger
                if cluster["startpos"] >= startdist and cluster["endpos"] <= startdist:
                    #Found an appropriate cluster; append to it
                    assert(cluster["time"] == starttime)
                    ffdeparttime = starttime + (lengths[lane]-clusters[lane][-1]["endpos"])/speeds[startedge]
                    departtime = max(ffdeparttime, cluster["departure"])
                    cluster["cars"].append((vehicle, departtime, 1, "VOI append clone"))
                    cluster["weight"] += 1
                    VOIadded = True
                    finishedLanes[lane] = True
                    break
                if cluster["endpos"] > startdist:
                    #This is the first cluster after where we wanted to insert, so we'll want to create a new cluster before this
                    insertBeforeEnd = True
                    break
            if not VOIadded:
                #Else make a new cluster
                newcluster = dict()
                newcluster["startpos"] = startdist
                newcluster["endpos"] = newcluster["startpos"]
                newcluster["time"] = starttime
                newcluster["arrival"] = newcluster["time"] + (lengths[lane]-newcluster["endpos"])/speeds[startedge]
                newcluster["departure"] = newcluster["arrival"]
                newcluster["cars"] = [(vehicle, newcluster["departure"], 1, "VOI new clone")]
                newcluster["weight"] = len(newcluster["cars"])
                if insertBeforeEnd:
                    clusters[lane].insert(clusterind, newcluster) #Add the new cluster to right before the cluster after
                else:
                    clusters[lane].append(newcluster) #There is no cluster after, put the new one at the end
                finishedLanes[lane] = True


    queueSimPredClusters = pickle.loads(pickle.dumps(sumoPredClusters)) #Initial predicted clusters are whatever SUMO's Surtrac thinks it is
    queueSimLastSwitchTimes = pickle.loads(pickle.dumps(mainlastswitchtimes)) #Initial last switch times are whatever they were in the main simulation
    remainingDuration = pickle.loads(pickle.dumps(mainRemainingDuration)) #Copy any existing schedules from main sim
    surtracFreq = routingSurtracFreq #Time between Surtrac updates, in seconds, during routing. (Technically the period between updates)

    #Cutoff in case of infinite loop?
    routestartwctime = time.time()
    timeout = 60

    #Store durations to compare to real durations
    storeSimDurations = False
    newsim = True
    global simdurationsUsed
    global simdurations
    if simdurationsUsed == False:
        simdurations = dict()
        storeSimDurations = True
        simdurationsUsed = True

    #Loop through time and simulate things
    if not computeSurtrac:
        startedgeind = routes[vehicleOfInterest].index(startedge)
        bestroute = routes[vehicleOfInterest][startedgeind:]
        toupgrade = routes[vehicleOfInterest][startedgeind+1:]

    blockingLinks = dict()

    #Ignore old arrival data
    for lane in arrivals:
        while len(arrivals[lane]) > 0 and arrivals[lane][0] < starttime - maxarrivalwindow:
            arrivals[lane] = arrivals[lane][1:]
            
    while True:

        #Timeout if things have gone wrong somehow
        if not computeSurtrac and time.time()-routestartwctime > timeout:
            print("Routing timeout: Edge " + startedge + ", time: " + str(starttime))
            routeStats[vehicleOfInterest]["nTimeouts"] += 1
            
            nToReroute -= 1
            return (bestroute, -1)

        #If nothing needs Surtrac schedules, stop computing them
        if computeSurtrac and killSurtracThread:
            return [] #Zero element list so the data-copier doesn't get mad

        #Sanity check for debugging infinite loops where the vehicle of interest disappears
        if debugMode:
            if not computeSurtrac:
                notEmpty = False
                for thing in clusters:
                    for thingnum in range(len(clusters[thing])):
                        for testcartuple in clusters[thing][thingnum]["cars"]:
                            if testcartuple[0] in VOIs:
                                notEmpty = True
                                break
                if not notEmpty:
                    print(VOIs)
                    #print(clusters)
                    print(startlane)
                    raise Exception("Can't find vehicle of interest!")
                #End sanity check

        routesimtime += timestep
        #Reminder: We rounded the first routesimtime down to a multiple of the timestep so sims starting at different times ask for light info at fixed times.

        #Add new cars
        if routesimtime >= starttime: #Deal with initial routesimtime being rounded down - don't create new cars before starttime
            for nextlane in arrivals:
                nextedge = nextlane.split("_")[0]
                if len(arrivals[nextlane]) == 0:
                    #No recent arrivals, nothing to add
                    continue
                timeperarrival = min(starttime, maxarrivalwindow)/len(arrivals[nextlane])
                if timeperarrival <= timestep or routesimtime%timeperarrival >= (routesimtime+timestep)%timeperarrival:
                    #Add a car
                    #Check append to previous cluster vs. add new cluster
                    if len(clusters[nextlane]) > 0 and abs(clusters[nextlane][-1]["time"] - routesimtime) < clusterthresh and abs(clusters[nextlane][-1]["endpos"])/speeds[nextedge] < clusterthresh:
                        #Add to cluster. pos and time track newest added vehicle to see if the next vehicle merges
                        #Departure time (=time to fully clear cluster) increases, arrival doesn't
                        #TODO eventually: Be more precise with time and position over partial timesteps, allowing me to use larger timesteps?
                        clusters[nextlane][-1]["endpos"] = 0
                        clusters[nextlane][-1]["time"] = routesimtime
                        clusters[nextlane][-1]["departure"] = routesimtime + fftimes[nextedge]
                        newcarname = "ImANewCar"+str(newcarcounter)
                        clusters[nextlane][-1]["cars"].append((newcarname, clusters[nextlane][-1]["departure"], 1, "New car routing append"))
                        routes[newcarname] = sampleRouteFromTurnData(newcarname, nextlane, turndata)
                        newcarcounter += 1
                        clusters[nextlane][-1]["weight"] += 1
                    else:
                        #There is no cluster nearby
                        #So make a new cluster
                        newcluster = dict()
                        newcluster["endpos"] = 0
                        newcluster["time"] = routesimtime
                        newcluster["arrival"] = routesimtime + fftimes[nextedge]
                        newcluster["departure"] = newcluster["arrival"]
                        newcarname = "ImANewCar"+str(newcarcounter)
                        newcluster["cars"] = [(newcarname, newcluster["departure"], 1, "New car routing new cluster")]
                        routes[newcarname] = sampleRouteFromTurnData(newcarname, nextlane, turndata)
                        newcarcounter += 1
                        newcluster["weight"] = 1
                        clusters[nextlane].append(newcluster)
                    #Newly-created car has been added

        #Combine clusters waiting at lights
        clusters = recluster(clusters, routesimtime)

        #Update lights
        if surtracFreq <= timestep or routesimtime%surtracFreq >= (routesimtime+timestep)%surtracFreq:
            #Look up or store Surtrac
            if computeSurtrac:
                assert(not routesimtime in surtracDict) #Should only be one thread updating surtracDict, and it should be being cleared afterwards
                surtracDict[routesimtime] = doSurtrac(net, routesimtime, clusters, lightphases, queueSimLastSwitchTimes, queueSimPredClusters)

            #Wait for Surtrac-computing thread to compute a schedule
            if reuseSurtrac:
                while not routesimtime in surtracDict:
                    #print("Waiting on " + str(routesimtime))
                    time.sleep(0) #Apparently this yields to other threads

                #Load the schedule computed in the other thread
                #Be careful to deep copy so changes to these in one thread don't bleed over into other threads
                (toSwitch, queueSimPredClusters, newRemainingDuration) = pickle.loads(pickle.dumps(surtracDict[routesimtime]))
                if not computeSurtrac:
                    for light in toSwitch:
                        nPhases = len(surtracdata[light]) #Number of phases
                        lightphases[light] = (lightphases[light]+1)%nPhases #Change the light, since doSurtrac would've done it but wasn't called
                        queueSimLastSwitchTimes[light] = routesimtime
            else:
                #No Surtrac-computing thread, do it yourself
                (toSwitch, queueSimPredClusters, newRemainingDuration) = doSurtrac(net, routesimtime, clusters, lightphases, queueSimLastSwitchTimes, queueSimPredClusters)
            
            remainingDuration.update(pickle.loads(pickle.dumps(newRemainingDuration))) #Copy by value, not by location! Otherwise all the "decrement by timestep" stuff decrements everything in all threads!
        
        #Keep track of what lights change when, since we're not running Surtrac every timestep
        for light in lights:
            if light not in remainingDuration or len(remainingDuration[light]) == 0:
                print("Empty remainingDuration for light " + light + " in runClusters, which shouldn't happen; using the default value")
                remainingDuration[light] = [lightphasedata[light][lightphases[light]].duration]
            #All lights should have a non-zero length schedule in remainingDuration
            remainingDuration[light][0] -= timestep
            if remainingDuration[light][0] <= 0: #Note: Duration might not be divisible by timestep, so we might be getting off by a little over multiple phases??
                tosubtract = remainingDuration[light][0]
                remainingDuration[light].pop(0)
                lightphases[light] = (lightphases[light]+1)%len(lightphasedata[light])
                queueSimLastSwitchTimes[light] = routesimtime #TODO: Does this cause divisibility issues?
                if len(remainingDuration[light]) == 0:
                    remainingDuration[light] = [lightphasedata[light][lightphases[light]].duration + tosubtract] #tosubtract is negative
                #Main sim doesn't subtract for overruns if there's a Surtrac schedule, so we won't do that here either
        
        #Test code to make sure light schedules don't drift for large surtracFreq
        # if not routesimtime in simdurations:
        #     simdurations[routesimtime] = pickle.loads(pickle.dumps(remainingDuration))
        #     if newsim == True:
        #         simdurations[routesimtime][lights[0]] = str(simdurations[routesimtime][lights[0]]) + " NEW SIM"
        #     newsim = False

        if not routesimtime in lanesToCheck:
            #print("Nothing at time " + str(routesimtime))
            #Nothing to update, apparently; go to next timestep
            continue

        for lane in reversed(lanesToCheck[routesimtime]): #Reversing to handle zipper merges
            edge = lane.split("_")[0]
        #TODO: Make sure zipper merges still work...
        # reflist = pickle.loads(pickle.dumps(edgelist)) #Want to reorder edge list to handle priority stuff, but don't want to mess up the for loop indexing

            while len(clusters[lane]) > 0:
                cluster = clusters[lane][0]

                if cluster["arrival"] > routesimtime:
                    #This and future clusters don't arrive yet, done on this edge
                    break
                if len(cluster["cars"]) == 0:
                    #print("Warning: Empty cluster. This might be the result of addNoise removing all cars.")
                    clusters[lane].remove(cluster)
                    continue
                cartuple = cluster["cars"][0]
                
                #Check if lead car on lane can progress
                #This is a while loop, but should only execute once if I've done this right. (Not going to rewrite because break statements are convenient)
                while cartuple[1] < routesimtime and lastDepartTime[lane] <= routesimtime - mingap:
                    #Check if route is done; if so, stop
                    if cartuple[0] in VOIs and edge in toupgrade:
                        toupgrade = toupgrade[toupgrade.index(edge)+1:]
                        #Compute new best route
                        splitroute = cartuple[0].split("|")
                        splitroute.pop(0)
                        fullroute = [startedge]
                        for routepart in splitroute:
                            fullroute.append(routepart.split("_")[0]) #Only grab the edge, not the lane
                        bestroute = fullroute + list(toupgrade)
                    if cartuple[0] in VOIs and edge == goalEdge:
                        #We're done simulating; extract the route
                        splitroute = cartuple[0].split("|")
                        splitroute.pop(0)
                        fullroute = [startedge]
                        for routepart in splitroute:
                            fullroute.append(routepart.split("_")[0]) #Only grab the edge, not the lane
                        #print("End routing simulation. Time: " + str(starttime) + " , vehicle: " + vehicleOfInterest)
                        nToReroute -= 1
                        return (fullroute, routesimtime-starttime)
                    elif not cartuple[0] in VOIs and routes[cartuple[0]][-1] == edge:
                        cluster["cars"].pop(0) #Remove car from this edge
                        break

                    
                    #Add car to next edge. NOTE: Enforce merging collision etc. constraints here
                    node = net.getEdge(edge).getToNode().getID()
                    if not node in blockingLinks:
                        blockingLinks[node] = dict()
                    #print(node.getID()) #Matches the IDs on the traffic light list
                    #print(node.getType()) #zipper #traffic_light_right_on_red #dead_end
                    #print(lights)
                    #https://sumo.dlr.de/docs/TraCI/Traffic_Lights_Value_Retrieval.html
                    #If light, look up phase, decide who gets to go, merge foe streams somehow
                    #Or just separate left turn phases or something? Would mean no need to merge
                    #If no light, zipper somehow
                        
                    #Figure out where the car wants to go
                    if not (cartuple[0], edge) in splitinfo:
                        #Assume zipper
                        if cartuple[0] in VOIs:
                            nextedges = []
                            #Want all edges that current lane connects to
                            for nextlinktuple in links[lane]:
                                nextedge = nextlinktuple[0].split("_")[0]
                                if not nextedge in nextedges:
                                    nextedges.append(nextedge)
                        else:
                            route = routes[cartuple[0]]
                            routeind = route.index(edge)
                            nextedges = [route[routeind+1]]

                        #nextlanes is going to loop over everything in nextedges
                        #Splitty cars want to go to everything in nextlanes
                        #Non-splitty cars only want to go to one
                        nextlanes = []
                        for nextedge in nextedges:
                            for nextlanenum in range(lanenums[nextedge]):
                                nextlane = nextedge + "_" + str(nextlanenum)
                                
                                #If non-splitty car and this nextlane doesn't go to nextnextedge, disallow it
                                if not cartuple[0] in VOIs:
                                    if routeind + 2 < len(route): #Else there's no next next edge, don't be picky
                                        nextnextedge = route[routeind+2]
                                        usableLane = False
                                        for nextnextlinktuple in links[nextlane]:
                                            if nextnextlinktuple[0].split("_")[0] == nextnextedge: #linktuple[0].split("_")[0] gives edge the link goes to
                                                usableLane = True
                                                break
                                        if not usableLane: #This nextlane doesn't connect to nextnextedge
                                            continue #So try a different nextlane
                                nextlanes.append(nextedge + "_" + str(nextlanenum))
                        splitinfo[(cartuple[0], edge)] = nextlanes
                    #Else we've already figured out how we want to split this car

                    tempnextedges = pickle.loads(pickle.dumps(splitinfo[(cartuple[0], edge)]))

                    for nextlane in tempnextedges:
                        nextedge = nextlane.split("_")[0]

                        #Check light state
                        if node in lights:
                            isGreenLight = False

                            #If there's a relevant priority green link, use it
                            for linktuple in prioritygreenlightlinks[node][lightphases[node]]:
                                if linktuple[0] == lane and linktuple[1].split("_")[0] == nextedge: #If can go from this lane to next edge, it's relevant
                                    isGreenLight = True
                                    break


                            if not isGreenLight: #No priority streams work; need to check non-priority streams
                                if lane in lowprioritygreenlightlinksLE[node][lightphases[node]] and nextedge in lowprioritygreenlightlinksLE[node][lightphases[node]][lane]:
                                    for linktuple in lowprioritygreenlightlinksLE[node][lightphases[node]][lane][nextedge]:
                                        #Make sure we're not a g stream blocked by a G stream
                                        isBlocked = False

                                        #Check if anything we've already sent through will block this
                                        #Opposing traffic blocks links for some amount of time, not just one timestep
                                        for linktuple2 in prioritygreenlightlinks[node][lightphases[node]]+lowprioritygreenlightlinks[node][lightphases[node]]:
                                            conflicting = lightlinkconflicts[node][linktuple][linktuple2] #Precomputed to save time 

                                            if conflicting and (linktuple2 in blockingLinks[node] and blockingLinks[node][linktuple2] > routesimtime - 2*mingap): #That last n*mingap is the amount of blocking time
                                                isBlocked = True
                                                break
                                        if isBlocked:
                                            continue

                                        #Check for currently-unsent priority links (only needs to block for this timestep, after which we're in the above case)
                                        if not isBlocked:
                                            for linktuple2 in prioritygreenlightlinks[node][lightphases[node]]:
                                                conflicting = lightlinkconflicts[node][linktuple][linktuple2] #Precomputed to save time
                                                if not conflicting:
                                                    continue

                                                willBlock = False
                                                if len(clusters[linktuple2[0]]) > 0 and clusters[linktuple2[0]][0]["cars"][0][1] <= routesimtime: #clusters[linktuple2[0]][0]["arrival"] <= time:
                                                    blocker = clusters[linktuple2[0]][0]["cars"][0][0]
                                                    blockingEdge0 = linktuple2[0].split("_")[0]
                                                    blockingEdge1 = linktuple2[1].split("_")[0]
                                                    if blocker in VOIs:
                                                        if not blocker == cartuple[0]: #Don't block yourself
                                                            #You're behind a VOI, so you shouldn't matter
                                                            willBlock = True
                                                        
                                                    else:
                                                        blockerroute = routes[blocker]
                                                        blockerrouteind = blockerroute.index(blockingEdge0)
                                                        willBlock = (blockerrouteind+1<len(blockerroute) and blockerroute[blockerrouteind+1] == blockingEdge1)
                                                
                                                if willBlock:
                                                    isBlocked = True
                                                    break

                                        #If nothing already sent or to-be-sent with higher priority conflicts, this car can go
                                        if not isBlocked:
                                            isGreenLight = True
                                            break
                                        
                                    
                        else:
                            isGreenLight = True #Not a light, so assume a zipper and allow everything through

                        if not isGreenLight:
                            continue

                        #Check if next lane is completely full of cars; if so, can't fit more
                        totalcarnum = 0
                        carlength = 10 #meters #Traci claims default length is be 5m, but cars probably aren't literally bumper-to-bumper, and eyeballing this gets me closer to 10m
                        for nlc in clusters[nextlane]:
                            totalcarnum += nlc["weight"]
                        if totalcarnum >= (lengths[nextlane]-10) / carlength:
                            continue

                        #Check append to previous cluster vs. add new cluster
                        if len(clusters[nextlane]) > 0 and abs(clusters[nextlane][-1]["time"] - routesimtime) < clusterthresh and abs(clusters[nextlane][-1]["endpos"])/speeds[nextedge] < clusterthresh:
                            
                            #Make sure there's no car on the new road that's too close
                            if not abs(clusters[nextlane][-1]["time"] - routesimtime) < mingap:
                                #Do not add a VOI to a lane we've already added a VOI to (another copy won't help and will only confuse Surtrac)
                                if not cartuple[0] in VOIs or not nextlane in finishedLanes: #If so, don't need the extra copy of the VOI, but pretend we added it
                                    #Add to cluster. pos and time track newest added vehicle to see if the next vehicle merges
                                    #Departure time (=time to fully clear cluster) increases, arrival doesn't
                                    #TODO eventually: Be more precise with time and position over partial timesteps, allowing me to use larger timesteps?
                                    clusters[nextlane][-1]["endpos"] = 0
                                    clusters[nextlane][-1]["time"] = routesimtime
                                    clusters[nextlane][-1]["departure"] = routesimtime + fftimes[nextedge]
                                    if cartuple[0] in VOIs:
                                        clusters[nextlane][-1]["cars"].append((cartuple[0]+"|"+nextlane, clusters[nextlane][-1]["departure"], 1, "Zipper append"))
                                        VOIs.append(cartuple[0]+"|"+nextlane)
                                        finishedLanes[nextlane] = True
                                    else:
                                        clusters[nextlane][-1]["cars"].append((cartuple[0], clusters[nextlane][-1]["departure"], 1, "Zipper append"))
                                    clusters[nextlane][-1]["weight"] += 1
                            else:
                                #No space, try next lane
                                continue
                        else:
                            #Do not add a VOI to a lane we've already added a VOI to (another copy won't help and will only confuse Surtrac)
                            if not cartuple[0] in VOIs or not nextlane in finishedLanes: #If so, don't need the extra copy of the VOI, but pretend we added it
                                #There is no cluster nearby
                                #So make a new cluster
                                newcluster = dict()
                                newcluster["endpos"] = 0
                                newcluster["time"] = routesimtime
                                newcluster["arrival"] = routesimtime + fftimes[nextedge]
                                newcluster["departure"] = newcluster["arrival"]
                                if cartuple[0] in VOIs:
                                    newcluster["cars"] = [(cartuple[0]+"|"+nextlane, newcluster["departure"], 1, "Zipper new cluster")]
                                    VOIs.append(cartuple[0]+"|"+nextlane)
                                    finishedLanes[nextlane] = True
                                else:
                                    newcluster["cars"] = [(cartuple[0], newcluster["departure"], 1, "Zipper new cluster")]
                                newcluster["weight"] = 1
                                clusters[nextlane].append(newcluster)
                        #We've added a car to nextedge_nextlanenum
                        
                        #Remove vehicle from predictions, since the next intersection should actually see it now
                        if not disableSurtracPred:
                            for predlane in queueSimPredClusters:
                                for predcluster in queueSimPredClusters[predlane]:
                                    predcarind = 0
                                    minarr = inf
                                    maxarr = -inf
                                    while predcarind < len(predcluster["cars"]):
                                        predcartuple = predcluster["cars"][predcarind]
                                        if predcartuple[0] == id:
                                            predcluster["cars"].pop(predcarind)
                                            predcluster["weight"] -= predcartuple[2]
                                        else:
                                            predcarind += 1
                                            if predcartuple[1] < minarr:
                                                minarr = predcartuple[1]
                                            if predcartuple[1] > maxarr:
                                                maxarr = predcartuple[1]
                                    if len(predcluster["cars"]) == 0:
                                        sumoPredClusters[predlane].remove(predcluster)
                                    else:
                                        pass
                                        #Clusters apparently aren't sorted by arrival time
                                        # predcluster["arrival"] = minarr #predcluster["cars"][0][1]
                                        # predcluster["departure"] = maxarr #predcluster["cars"][-1][1]

                                    weightsum = 0
                                    for predcarind in range(len(predcluster["cars"])):
                                        weightsum += predcluster["cars"][predcarind][2]
                                    assert(abs(weightsum - predcluster["weight"]) < 1e-10)

                        try: #Can fail if linktuple isn't defined, which happens at non-traffic-lights
                            blockingLinks[node][linktuple] = routesimtime
                        except:
                            #print("Zipper test")
                            pass #It's a zipper?
                        splitinfo[(cartuple[0], edge)].remove(nextlane)
                        #Before, we'd break out of the lane loop because we'd only add to each edge once
                        #Now, we only get to break if it's a non-splitty car (splitty car goes to all nextlanes)
                        if not cartuple[0] in VOIs:
                            #Don't try to add non-splitty car anywhere else - it just needs one lane that works
                            splitinfo[(cartuple[0], edge)] = []
                            break

                    #This is the zipper merge logic in the cartuple < time loop, breaking it loops the cluster loop
                    #If zipper merge, need to alternate priority on things
                    if len(splitinfo[(cartuple[0], edge)]) == 0:
                        #We have successfully added the lead car to all next lanes it wanted to go to (all if splitty, target lane if not)
                        #Can now remove from current lane, etc.
                        lastDepartTime[lane] = routesimtime
                        #edgelist.append(edgelist.pop(edgelist.index(edge))) #Push edge to end of list to give it lower priority next time
                        cluster["cars"].pop(0) #Remove car from this edge
                        cluster["weight"] -= cartuple[2]
                        if len(cluster["cars"]) > 0:
                            cartuple = cluster["cars"][0]
                            #break #Only try to add one car
                            continue
                            #...and loop back try to add the next car
                        else:
                            #Cluster is empty, break out of cartuple loop to get the new cluster
                            break

                    #If we haven't added this car to all next edges, just stop?
                    else:
                        break
                        
                #Inside: while len(clusters[lane]) > 0
                if len(cluster["cars"]) == 0:
                    clusters[lane].pop(0) #Entire cluster is done, remove it
                    #break #Only try to add one car
                    continue
                else:
                    #Something's left in the first cluster, so everyone's blocked
                    break
            
            #Figure out the next time we need to check this lane
            #NOTE: I've somewhat David-proofed this, but double-check this if changing logic about checking when cars can go or where they get inserted on the next lane to avoid off-by-ones
            #TODO eventually: Drop timesteps completely and do this purely event-based? Still need to handle the traffic lights, though...
            if len(clusters[lane]) == 0:
                timetocheck = routesimtime + math.ceil(fftimes[lane]/timestep-1)*timestep #-1 in case I'm clever and teleport a vehicle partway onto the next road (pretty sure I don't do that now, but maybe I'll change it later)
            
            else:
                cluster = clusters[lane][0]

                cartuple = cluster["cars"][0]
                timetocheck = routesimtime + math.ceil((cartuple[1]-routesimtime)/timestep)*timestep #Since we'll never need to check before the vehicle reaches the intersection

            timetocheck = max(timetocheck, routesimtime+timestep, lastDepartTime[lane] + math.ceil(mingap/timestep)*timestep) #Since we disallow a new vehicle leaving less than mingap behind the most recent departure. Also make sure the next check happens sometime in the future...
            assert( (timetocheck - routesimtime) % timestep == 0) #If not, this lane gets frozen in time forever - oops.
            if not timetocheck in lanesToCheck:
                lanesToCheck[timetocheck] = [lane]
            else:
                lanesToCheck[timetocheck].append(lane)

def recluster(clusters, routesimtime):
    #Merge clusters that'll form initial queues to speed up Surtrac
    #TODO partial merging?
    for lane in clusters:
        while len(clusters[lane]) > 1:
            #Check if we arrive before previous stuff cleared and fill all the time until the originally scheduled departure
            if clusters[lane][1]["arrival"] < routesimtime + clusters[lane][0]["weight"]*mingap and clusters[lane][1]["departure"] < routesimtime + (clusters[lane][0]["weight"]+clusters[lane][1]["weight"])*mingap:
                #Merge cluster 0 into cluster 1

                #Pretty sure I don't care to recompute durations or anything - mindur math can take care of that
                clusters[lane][1]["arrival"] = clusters[lane][0]["arrival"]
                #Departure stays unchanged
                clusters[lane][1]["cars"] = clusters[lane][0]["cars"] + clusters[lane][1]["cars"] #Concatenate
                clusters[lane][1]["weight"] = clusters[lane][0]["weight"] + clusters[lane][1]["weight"] #Literally just addition
                #time and endpos shouldn't change either; time last car came into cluster 1 didn't change, and endpos does nothing outside of loadClusters
                
                #Delete old cluster 0
                clusters[lane] = clusters[lane][1:]
            else:
                #No more cluster merging on this lane
                break
    return clusters

def LAISB(a, b):
    #Line a intersects segment b
    #a is a tuple of points on the line; b is endpoints of segment
    #Negative if they intersect, positive if they don't, zero if an endpoint touches
    va = [ a[1][0] - a[0][0], a[1][1]-a[0][1] ]
    vb0 = [ b[0][0] - a[0][0], b[0][1]-a[0][1] ]
    vb1 = [ b[1][0] - a[0][0], b[1][1]-a[0][1] ]
    return np.cross(vb0, va) * np.cross(vb1, va)

def isIntersecting(a, b):
    #a and b are tuples of endpoints of line segments
    iab = LAISB(a, b)
    iba = LAISB(b, a)
    if iab > 0 or iba > 0:
        return False
    if iab == 0 and iba == 0:
        #Colinear, yuck
        return (min(a[0][0], a[1][0]) <= max(b[0][0], b[1][0]) and min(b[0][0], b[1][0]) <= max(a[0][0], a[1][0]) and
        min(a[0][1], a[1][1]) <= max(b[0][1], b[1][1]) and min(b[0][1], b[1][1]) <= max(a[0][1], a[1][1]))
    #Each segment either touches or crosses the other line, so we're good
    return True

def get_options():
    optParser = optparse.OptionParser()
    optParser.add_option("--nogui", action="store_true",
                         default=False, help="run the commandline version of sumo")
    options, args = optParser.parse_args()
    return options

#Generates induction loops on all the edges
def generate_additionalfile(sumoconfig, networkfile):
    #Create a third instance of a simulator so I can query the network
    try:
        traci.start([checkBinary('sumo'), "-c", sumoconfig,
                                "--start", "--no-step-log", "true",
                                "--xml-validation", "never", "--quit-on-end"], label="setup")
    except:
        #Worried about re-calling this without old setup instance being removed
        traci.switch("setup")

    net = sumolib.net.readNet(networkfile)
    rerouters = []
    global max_edge_speed

    with open("additional_autogen.xml", "w") as additional:
        print("""<additional>""", file=additional)
        for edge in traci.edge.getIDList():
            if edge[0] == ":":
                #Skip internal edges (=edges for the inside of each intersection)
                continue

            if (net.getEdge(edge).getSpeed() > max_edge_speed):
                max_edge_speed = net.getEdge(edge).getSpeed()

            for lanenum in range(traci.edge.getLaneNumber(edge)):
                lane = edge+"_"+str(lanenum)
                print('    <inductionLoop id="IL_%s" freq="1" file="outputAuto.xml" lane="%s" pos="-%i" friendlyPos="true" />' \
                      % (lane, lane, detectordist), file=additional)
                if len(net.getEdge(edge).getOutgoing()) > 1:
                    rerouters.append("IL_"+lane)
                    rerouterLanes["IL_"+lane] = lane
                    rerouterEdges["IL_"+lane] = edge
        print("</additional>", file=additional)
    
    return rerouters

def sampleRouteFromTurnData(vehicle, startlane, turndata):
    lane = startlane
    route = [lane.split("_")[0]]
    while lane in turndata:
        r = random.random()
        for nextlane in turndata[lane]:
            r -= turndata[lane][nextlane]
            if r <= 0:
                if nextlane.split("_")[0] == lane.split("_")[0]:
                    print("Warning: Sampling is infinite looping, stopping early")
                    return route
                lane = nextlane
                break
        route.append(lane.split("_")[0])
        #print(route)
    return route

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

def main(sumoconfig, pSmart, verbose = True, useLastRNGState = False, appendTrainingDataIn = False):
    global lowprioritygreenlightlinks
    global prioritygreenlightlinks
    global edges
    global turndata
    global actualStartDict
    global trainingdata
    global testNN
    global appendTrainingData
    options = get_options()

    if useLastRNGState:
        with open("lastRNGstate.pickle", 'rb') as handle:
            rngstate = pickle.load(handle)
    else:
        rngstate = random.getstate()
        with open("lastRNGstate.pickle", 'wb') as handle:
            pickle.dump(rngstate, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    appendTrainingData = appendTrainingDataIn

    # this script has been called from the command line. It will start sumo as a
    # server, then connect and run
    if options.nogui:
        sumoBinary = checkBinary('sumo')
    else:
        sumoBinary = checkBinary('sumo-gui')
    #NOTE: Script name is zeroth arg

    (netfile, routefile) = readSumoCfg(sumoconfig)

    network = sumolib.net.readNet(netfile)
    net = network
    rerouters = generate_additionalfile(sumoconfig, netfile)

    # this is the normal way of using traci. sumo is started as a
    # subprocess and then the python script connects and runs
    try:
        traci.start([sumoBinary, "-c", sumoconfig,
                                "--additional-files", "additional_autogen.xml",
                                #"--time-to-teleport", "-1",
                                "--log", "LOGFILE", "--xml-validation", "never", "--start", "--quit-on-end"], label="main")
    except:
        #Worried about re-calling this without old main instance being removed
        traci.switch("main")
        traci.load( "-c", sumoconfig,
                                "--additional-files", "additional_autogen.xml",
                                #"--time-to-teleport", "-1",
                                "--log", "LOGFILE", "--xml-validation", "never", "--start", "--quit-on-end")

    for node in sumolib.xml.parse(netfile, ['junction']):
        if node.type == "traffic_light":
            lights.append(node.id)
        else:
            notLights.append(node.id)
            notlightlanes[node.id] = []
            notlightoutlanes[node.id] = []

    for lane in sumolib.xml.parse(netfile, ['lane']):
        edge = lane.id.split("_")[0]
        if len(edge) == 0 or edge[0] == ":":
            continue
        toNode = network.getEdge(edge).getToNode()
        if toNode.getType() != "traffic_light":
            notlightlanes[toNode.getID()].append(lane.id)
        fromNode = network.getEdge(edge).getFromNode()
        if fromNode.getType() != "traffic_light":
            notlightoutlanes[fromNode.getID()].append(lane.id)

    #Grab stuff once at the start to avoid slow calls to traci in the routing
    edges = traci.edge.getIDList()
    lanes = traci.lane.getIDList()

    #Edges have speeds, but lanes have lengths, so it's a little annoying to get fftimes...
    for lane in lanes:
        if not lane[0] == ":":
            links[lane] = traci.lane.getLinks(lane)
            lengths[lane] = traci.lane.getLength(lane)

    for edge in edges:
        if not edge[0] == ":":
            lanenums[edge] = traci.edge.getLaneNumber(edge)
            speeds[edge] = network.getEdge(edge).getSpeed()
            fftimes[edge] = lengths[edge+"_0"]/speeds[edge]

    for lane in lanes:
        if not lane[0] == ":":
            fftimes[lane] = fftimes[lane.split("_")[0]]

    lowprioritygreenlightlinks = dict()
    prioritygreenlightlinks = dict()

    for light in lights:
        lightlinkconflicts[light] = dict()
        lightphasedata[light] = traci.trafficlight.getAllProgramLogics(light)[0].phases
        lightlinks[light] = traci.trafficlight.getControlledLinks(light)
        lightphases[light] = traci.trafficlight.getPhase(light)
        fftimes[light] = np.inf

        lightlanes[light] = []
        lightoutlanes[light] = []

        linklistlist = lightlinks[light]
        for linklist in linklistlist:

            for linktuple in linklist:
                inlane = linktuple[0]
                if not inlane in lightlanes[light]:
                    lightlanes[light].append(inlane)
                outlane = linktuple[1]
                if not outlane in lightoutlanes[light]:
                    lightoutlanes[light].append(outlane)
                    if fftimes[light] > fftimes[outlane]:
                        fftimes[light] = fftimes[outlane]

                lightlinkconflicts[light][linktuple] = dict()
                for linklist2 in linklistlist:
                    for linktuple2 in linklist2:
                        lightlinkconflicts[light][linktuple][linktuple2] = isIntersecting( (network.getLane(linktuple[0]).getShape()[1], (net.getLane(linktuple[1]).getShape()[0])), 
                        (net.getLane(linktuple2[0]).getShape()[1], (network.getLane(linktuple2[1]).getShape()[0])) )

    #Surtrac data
    for light in lights:
        surtracdata[light] = []
        mainlastswitchtimes[light] = 0
        lowprioritygreenlightlinks[light] = []
        prioritygreenlightlinks[light] = []
        lowprioritygreenlightlinksLE[light] = []
        prioritygreenlightlinksLE[light] = []

        n = len(lightphasedata[light])
        for i in range(n):
            surtracdata[light].append(dict())
            surtracdata[light][i]["minDur"] = 7#5#lightphasedata[light][i].minDur
            surtracdata[light][i]["maxDur"] = 120#lightphasedata[light][i].maxDur

            if "G" in lightphasedata[light][i].state or "g" in lightphasedata[light][i].state:
                surtracdata[light][i]["minDur"] = 5 #1#3.5#5#lightphasedata[light][i].minDur
            # if "Y" in lightphasedata[light][i].state or "y" in lightphasedata[light][i].state:
            #     surtracdata[light][i]["minDur"] = 2#5#lightphasedata[light][i].minDur #There is no all-red phase, keep this long
            
            #Force yellow to be the min length - not doing this since it messes with the training data (triggers the min-max <= maxfreq condition where we'll break stuff with discretization)
            # if "Y" in lightphasedata[light][i].state or "y" in lightphasedata[light][i].state:
            #     surtracdata[light][i]["maxDur"] = surtracdata[light][i]["minDur"] #Force this to be the correct length. Don't think this matters though...

            surtracdata[light][i]["lanes"] = []
            lightstate = lightphasedata[light][i].state
            lowprioritygreenlightlinks[light].append([])
            prioritygreenlightlinks[light].append([])
            lowprioritygreenlightlinksLE[light].append(dict())
            prioritygreenlightlinksLE[light].append(dict())
            
            linklistlist = lightlinks[light]
            for linklistind in range(len(linklistlist)):
                linkstate = lightstate[linklistind]
                for linktuple in linklistlist[linklistind]:
                    if not linktuple[0] in lanephases:
                        lanephases[linktuple[0]] = []

                    if linkstate == "G" and not linktuple[0] in surtracdata[light][i]["lanes"]:
                        surtracdata[light][i]["lanes"].append(linktuple[0]) #[0][x]; x=0 is from, x=1 is to, x=2 is via
                        
                        lanephases[linktuple[0]].append(i)

                linklist = linklistlist[linklistind] #TODO just unindented this, make sure nothing breaks
                for link in linklist:
                    if linkstate == "G":
                        prioritygreenlightlinks[light][i].append(link)
                        #Make sure lane->edge dictionary knows about this lane->edge pair
                        if not link[0] in prioritygreenlightlinksLE[light][i]:
                            prioritygreenlightlinksLE[light][i][link[0]] = dict()
                        if not link[1].split("_")[0] in prioritygreenlightlinksLE[light][i][link[0]]:
                            prioritygreenlightlinksLE[light][i][link[0]][link[1].split("_")[0]] = []
                        #Add to lane->edge dictionary
                        prioritygreenlightlinksLE[light][i][link[0]][link[1].split("_")[0]].append(link)
                    if linkstate == "g":
                        lowprioritygreenlightlinks[light][i].append(link)
                        #Make sure lane->edge dictionary knows about this lane->edge pair
                        if not link[0] in lowprioritygreenlightlinksLE[light][i]:
                            lowprioritygreenlightlinksLE[light][i][link[0]] = dict()
                        if not link[1].split("_")[0] in lowprioritygreenlightlinksLE[light][i][link[0]]:
                            lowprioritygreenlightlinksLE[light][i][link[0]][link[1].split("_")[0]] = []
                        #Add to lane->edge dictionary
                        lowprioritygreenlightlinksLE[light][i][link[0]][link[1].split("_")[0]].append(link)
                
            #Remove lanes if there's any direction that gets a non-green light ("g" is fine, single-lane left turns are just sad)
            for linklistind in range(len(linklistlist)):
                linkstate = lightstate[linklistind]
                for linktuple in linklistlist[linklistind]:

                    if not (linkstate == "G" or linkstate == "g") and linktuple[0] in surtracdata[light][i]["lanes"]: #NOTE: I'm being sloppy and assuming one-element lists of tuples, but I've yet to see a multi-element list here
                        surtracdata[light][i]["lanes"].remove(linktuple[0])
                        lanephases[linktuple[0]].remove(i)
                
        for i in range(n):
            #Compute min transition time between the start of any two phases
            surtracdata[light][i]["timeTo"] = [0]*n
            for joffset in range(1, n):
                j = (i + joffset) % n
                jprev = (j-1) % n
                surtracdata[light][i]["timeTo"][j] = surtracdata[light][i]["timeTo"][jprev] + surtracdata[light][jprev]["minDur"]

    if pSmart < 1 or True:
        with open("Lturndata_"+routefile.split(".")[0]+".pickle", 'rb') as handle:
            turndata = pickle.load(handle) #This is lane-to-lane normalized turndata in the format turndata[lane][nextlane]
            #When predicting adopters ahead, we know their next edge, but not the specific lane on that edge. Calculate probability of that edge so we can normalize
            for lane in turndata:
                normprobs[lane] = dict()
                for nextlane in turndata[lane]:
                    nextedge = nextlane.split("_")[0]
                    if not nextedge in normprobs[lane]:
                        normprobs[lane][nextedge] = 0
                    normprobs[lane][nextedge] += turndata[lane][nextlane]

    #Parse route file to get intended departure times (to account for delayed SUMO insertions due to lack of space)
    #Based on: https://www.geeksforgeeks.org/xml-parsing-python/
    # create element tree object
    tree = ET.parse(routefile)

    # get root element
    root = tree.getroot()

    actualStartDict = dict()
    # iterate news items
    for item in root.findall('./vehicle'):
        actualStartDict[item.attrib["id"]] = float(item.attrib["depart"])
    for item in root.findall('./trip'):
        actualStartDict[item.attrib["id"]] = float(item.attrib["depart"])

    #Do NN setup
    testNN = testNNdefault
    for light in lights:
        if resetTrainingData:
            trainingdata[light] = []

        if testNNdefault:
            if testDumbtrac:
                agents[light] = Net(26, 1, 32)
                #agents[light] = Net(2, 1, 32)
            else:
                agents[light] = Net(362, 1, 1024)
            optimizers[light] = torch.optim.Adam(agents[light].parameters(), lr=learning_rate)
            MODEL_FILES[light] = 'models/imitate_'+light+'.model' # Once your program successfully trains a network, this file will be written
            print("Checking if there's a learned model. Currently testNN="+str(testNN))
            try:
                agents[light].load(MODEL_FILES[light])
            except FileNotFoundError:
                print("Model doesn't exist - turning off testNN")
                testNN = False
    if not resetTrainingData and appendTrainingData:
        print("LOADING TRAINING DATA, this could take a while")
        try:
            with open("trainingdata/trainingdata_" + sys.argv[1] + ".pickle", 'rb') as handle:
                trainingdata = pickle.load(handle)
        except FileNotFoundError:
            print("Training data not found, starting fresh")
            for light in lights:
                trainingdata[light] = []

    outdata = run(network, rerouters, pSmart, verbose)
    
    if appendTrainingData:
        print("Saving training data")
        with open("trainingdata/trainingdata_" + sys.argv[1] + ".pickle", 'wb') as handle:
            pickle.dump(trainingdata, handle, protocol=pickle.HIGHEST_PROTOCOL)
    traci.close()

    print("Routing calls: " + str(nRoutingCalls))
    print("Total routing time: " + str(routingTime))
    if nRoutingCalls > 0:
        print("Average time per call: " + str(routingTime/nRoutingCalls))
    if debugMode:
        print("Surtrac calls (one for each light): " + str(totalSurtracRuns))
        if totalSurtracRuns > 0:
            print("Average time per Surtrac call: " + str(totalSurtracTime/totalSurtracRuns))
            print("Average number of clusters per Surtrac call: " + str(totalSurtracClusters/totalSurtracRuns))
    return [outdata, rngstate]


# this is the main entry point of this script
if __name__ == "__main__":
    if len(sys.argv) >= 3:
        pSmart = float(sys.argv[2])
    if len(sys.argv) >= 4:
        useLastRNGState = sys.argv[3]
    if len(sys.argv) >= 5:
        appendTrainingData = sys.argv[4]
    main(sys.argv[1], pSmart, True, useLastRNGState, appendTrainingData)
