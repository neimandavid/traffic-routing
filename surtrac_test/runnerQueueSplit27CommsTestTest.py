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
#19: Going back to A* in SUMO since ghost cars are causing problems. Telling all ghost cars to turn left, and spawning left-turning ghost cars when the turn completes so we account for oncoming traffic
#20: Storing Surtrac results for reuse
#21: Libsumo (optional hopefully), multithreaded routing, save files under different names so we can run two sets of code without making one error
#22: Did routing before Surtrac for more realism (in case we're not simultaneous so Surtrac's out of date). Not much effect there.
#23: 21, but now using a detector model to reconstruct the traffic state at the start of each routing simulation
#24: Detector model stops tracking specific names of non-adopters
#25: New plan for lane changes - blindly sample which lane stuff ends up in
#26: Detector model for Surtrac in routing as well (since the goal is to approximate what the main simulation would be doing)
#27: Support new SurtracNet (single network for all intersections, takes in intersection geometry and light phase info)

from __future__ import absolute_import
from __future__ import print_function

# import torch
# from torch import nn

import os
import sys
import optparse
import random
from numpy import inf
import numpy as np
import time
#import matplotlib.pyplot as plt
#import math
#from copy import deepcopy, copy
# from collections import Counter
from heapq import * #priorityqueue
# import threading
import xml.etree.ElementTree as ET

# we need to import python modules from the $SUMO_HOME/tools directory
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

from sumolib import checkBinary

useLibsumo = True
if useLibsumo:
    try:
        import libsumo as traci
    except:
        print("Error using libsumo. Dropping back to traci instead. Hopefully this is fine")
        useLibsumo = False
if not useLibsumo:
    import traci  #To interface with SUMO simulations

import sumolib #To query node/edge stuff about the network
import pickle #To save/load traffic light states

# from Net import Net
# import openpyxl #For writing training data to .xlsx files

# from multiprocessing import Process
# import multiprocessing

# try:
#     multiprocessing.set_start_method("fork")
# except:
#     pass

sumoconfig = None

pSmart = 1.0 #Adoption probability
useLastRNGState = False #To rerun the last simulation without changing the seed on the random number generator

appendTrainingData = False#True

nVehicles = []

# dumpIntersectionData = False
# intersectionData = dict()
# vehicleIntersectionData = dict()

max_edge_speed = 0.0 #Overwritten when we read the route file

lanes = []
edges = []
carsOnNetwork = []
oldids = dict()
isSmart = dict() #Store whether each vehicle does our routing or not

nLanes = dict()
speeds = dict()
fftimes = dict() #Free flow times for each edge/lane (dict contains both) and from each light (min over outlanes)
links = dict()
lengths = dict()
turndata = []
currentRoutes = dict()
hmetadict = dict()
laneDict = dict()

nRoutingCalls = 0
nSuccessfulRoutingCalls = 0
routingTime = 0
routeVerifyData = []

AStarCutoff = inf

oldids = dict()
timedata = dict()

savename = "MAIN_" + str(time.time())
netfile = "UNKNOWN_FILENAME_OOPS"

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

teleportdata = []

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

            #If we found a better way to reach succ, the old way is still in the queue; delete it
            if succ in gvals and gvals[succ] > gval+c:
                pq.remove((gvals[succ], succ))
                heapify(pq)

            #Otherwise it's new or we're now doing better, so requeue it
            gvals[succ] = gval+c
            heappush(pq, (gval+c+h, succ))
    return gvals

#@profile
def run(network, rerouters, pSmart, verbose = True):
    global sumoPredClusters
    global currentRoutes
    global hmetadict
    global delay3adjdict
    global actualStartDict
    global edgeDict
    global laneDict
    global clustersCache
    global teleportdata
    global adopterinfo
    #netfile is the filepath to the network file, so we can call sumolib to get successors
    #rerouters is the list of induction loops on edges with multiple successor edges
    
    startDict = dict()
    endDict = dict()
    delayDict = dict()
    delay2adjdict = dict()
    delay3adjdict = dict()
    edgeDict = dict()
    laneDict = dict()
    leftDict = dict()
    carsOnNetwork = []
    remainingDuration = dict()

    tstart = time.time()
    simtime = 0

    while traci.simulation.getMinExpectedNumber() > 0 and (not appendTrainingData or simtime < 5000):
        simtime += 1
        traci.simulationStep() #Tell the simulator to simulate the next time step

        #Decide whether new vehicles use our routing
        for vehicle in traci.simulation.getDepartedIDList():
            isSmart[vehicle] = random.random() < pSmart
            if isSmart[vehicle]:
                traci.vehicle.setColor(vehicle, [0, 255, 0, 255])
            else:
                traci.vehicle.setColor(vehicle, [255, 0, 0, 255])
            timedata[vehicle] = [simtime, -1, -1, 'unknown', 'unknown']
            currentRoutes[vehicle] = traci.vehicle.getRoute(vehicle)

            goaledge = currentRoutes[vehicle][-1]
            if not goaledge in hmetadict:
                hmetadict[goaledge] = backwardDijkstra(network, goaledge)
            delayDict[vehicle] = -hmetadict[goaledge][currentRoutes[vehicle][0]] #I'll add the actual travel time once the vehicle arrives
            laneDict[vehicle] = traci.vehicle.getLaneID(vehicle)
            edgeDict[vehicle] = traci.vehicle.getRoadID(vehicle)

            startDict[vehicle] = simtime
            leftDict[vehicle] = 0

        #Check predicted vs. actual travel times
        for vehicle in traci.simulation.getArrivedIDList():
            if isSmart[vehicle]:
                timedata[vehicle][1] = simtime
            endDict[vehicle] = simtime

        vehiclesOnNetwork = traci.vehicle.getIDList()
        carsOnNetwork.append(len(vehiclesOnNetwork)) #Store number of cars on network (for plotting)

        #Plot and print stats
        if simtime%100 == 0 or not traci.simulation.getMinExpectedNumber() > 0:
            
            #Stats
            avgTime = 0
            avgTime0 = 0
            nCars = 0
            nSmart = 0

            for id in endDict:
                if actualStartDict[id] >= 600 and actualStartDict[id] < 3000:
                    nCars += 1
                    if isSmart[id]:
                        nSmart += 1

            for id in endDict:
                #Only look at steady state - ignore first and last 10 minutes of cars
                if actualStartDict[id] < 600 or actualStartDict[id] >= 3000:
                    continue

                ttemp = (endDict[id] - startDict[id])+delayDict[id]
                avgTime += ttemp/nCars

                #Delay0 computation (start clock at intended entrance time)
                ttemp0 = (endDict[id] - actualStartDict[id])+delayDict[id]
                avgTime0 += ttemp0/nCars
                # if isSmart[id]:
                #     avgTimeSmart0 += ttemp0/nSmart
                # else:
                #     avgTimeNot0 += ttemp0/(nCars-nSmart)


            if verbose or not traci.simulation.getMinExpectedNumber() > 0 or (appendTrainingData and simtime == 5000):
                print(pSmart)
                print("\nCurrent simulation time: %f" % simtime)
                print("Total run time: %f" % (time.time() - tstart))
                print("Number of vehicles in network: %f" % traci.vehicle.getIDCount())
                print("Total cars that left the network: %f" % len(endDict))
                print("Average delay: %f" % avgTime)
                print("Average delay0: %f" % avgTime0)
                

    try:
        os.remove("savestates/teststate_"+savename+".xml") #Delete the savestates file so we don't have random garbage building up over time
    except FileNotFoundError:
        print("Warning: Trying to clean up savestates file, but no file found. This is weird - did you comment out routing or something? Ignoring for now.")
        pass

    return []#[avgTime, avgTimeSmart, avgTimeNot, avgTime2, avgTimeSmart2, avgTimeNot2, avgTime3, avgTimeSmart3, avgTimeNot3, avgTime0, avgTimeSmart0, avgTimeNot0, time.time()-tstart, nteleports, teleportdata]  

def get_options():
    optParser = optparse.OptionParser()
    optParser.add_option("--nogui", action="store_true",
                         default=False, help="run the commandline version of sumo")
    options, args = optParser.parse_args()
    return options

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

def main(sumoconfigin, pSmart, verbose = True, useLastRNGState = False, appendTrainingDataIn = False):
    global lowprioritygreenlightlinks
    global prioritygreenlightlinks
    global edges
    global lanes
    global turndata
    global actualStartDict
    global trainingdata
    global testNN
    global appendTrainingData
    global noNNinMain
    global sumoconfig
    global netfile

    sumoconfig = sumoconfigin
    #options = get_options()

    if useLastRNGState:
        with open("lastRNGstate.pickle", 'rb') as handle:
            rngstate = pickle.load(handle)
            random.setstate(rngstate)
    else:
        rngstate = random.getstate()
        with open("lastRNGstate.pickle", 'wb') as handle:
            pickle.dump(rngstate, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    appendTrainingData = appendTrainingDataIn
    if appendTrainingDataIn:
        #We're training, so overwrite whatever else we're doing
        noNNinMain = False
        debugNNslowness = False

    # this script has been called from the command line. It will start sumo as a
    # server, then connect and run

    #sumoBinary = checkBinary('sumo')
    sumoBinary = checkBinary('sumo-gui')
    #NOTE: Script name is zeroth arg

    (netfile, routefile) = readSumoCfg(sumoconfig)

    network = sumolib.net.readNet(netfile)
    net = network

    # this is the normal way of using traci. sumo is started as a
    # subprocess and then the python script connects and runs
    if useLibsumo:
        traci.load(["-c", sumoconfig,
                                #"--additional-files", "additional_autogen.xml",
                                "--no-step-log", "true",
                                "--log", "LOGFILE", "--xml-validation", "never", "--start", "--quit-on-end"])
    else:
        try:
            traci.start([sumoBinary, "-c", sumoconfig,
                                    #"--additional-files", "additional_autogen.xml",
                                    "--no-step-log", "true",
                                    "--log", "LOGFILE", "--xml-validation", "never", "--start", "--quit-on-end"], label="main")
            #Second simulator for running tests. No GUI
            #traci.start([sumoBinary, "-c", sumoconfig, #GUI in case we need to debug
            traci.start([checkBinary('sumo'), "-c", sumoconfig, #No GUI
                                    #"--additional-files", "additionalrouting_autogen.xml",
                                    "--start", "--no-step-log", "true",
                                    "--xml-validation", "never", "--quit-on-end",
                                    "--step-length", "1"], label="test")
            dontBreakEverything()
        except:
            #Worried about re-calling this without old main instance being removed
            if not useLibsumo:
                traci.switch("main")
            traci.load([ "-c", sumoconfig,
                                    "--additional-files", "additional_autogen.xml",
                                    "--no-step-log", "true",
                                    #"--time-to-teleport", "-1",
                                    "--log", "LOGFILE", "--xml-validation", "never", "--start", "--quit-on-end"])
            if not useLibsumo:
                traci.switch("test")
            traci.load([ "-c", sumoconfig,
                                    "--additional-files", "additionalrouting_autogen.xml",
                                    "--no-step-log", "true",
                                    #"--time-to-teleport", "-1",
                                    "--log", "LOGFILE", "--xml-validation", "never", "--start", "--quit-on-end"])
            dontBreakEverything()

    #Grab stuff once at the start to avoid slow calls to traci in the routing
    edges = traci.edge.getIDList()
    lanes = traci.lane.getIDList()

    #Edges have speeds, but lanes have lengths, so it's a little annoying to get fftimes...
    for lane in lanes:
        if not lane[0] == ":":
            links[lane] = traci.lane.getLinks(lane)
            lengths[lane] = traci.lane.getLength(lane)

    for edge in edges:
        nLanes[edge] = traci.edge.getLaneNumber(edge)
        if not edge[0] == ":":
            speeds[edge] = network.getEdge(edge).getSpeed()
            fftimes[edge] = lengths[edge+"_0"]/speeds[edge]

    for lane in lanes:
        if not lane[0] == ":":
            fftimes[lane] = fftimes[lane.split("_")[0]]

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

    outdata = run(network, [], pSmart, verbose)
    
    return [outdata, rngstate]

#Magically makes the vehicle lists stop deleting themselves somehow???
def dontBreakEverything():
    if not useLibsumo:
        traci.switch("test")
        traci.simulationStep()
        traci.switch("main")


# this is the main entry point of this script
if __name__ == "__main__":
    if len(sys.argv) >= 3:
        pSmart = float(sys.argv[2])
    if len(sys.argv) >= 4:
        useLastRNGState = sys.argv[3]
    if len(sys.argv) >= 5:
        appendTrainingData = sys.argv[4]
    main(sys.argv[1], pSmart, True, useLastRNGState, appendTrainingData)
