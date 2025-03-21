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

import torch
from torch import nn

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

from Net import Net
import openpyxl #For writing training data to .xlsx files

from multiprocessing import Process
import multiprocessing

try:
    multiprocessing.set_start_method("fork")
except:
    pass

manager = multiprocessing.Manager()
sumoconfig = None

pSmart = 1.0 #Adoption probability
useLastRNGState = False #To rerun the last simulation without changing the seed on the random number generator

clusterthresh = 5 #Time between cars before we split to separate clusters
mingap = 2.5 #Minimum allowed space between cars
timestep = 1 #Amount of time between updates. In practice, mingap rounds up to the nearest multiple of this #NOTE: I'm pretty sure this used to be the timestep length in routing simulations, but I've since just started using SUMO with the default timestep of 1. timestep clearly is still in the code, but I'm not sure what it does anymore
detectordist = 50 #How far before the end of a road the detectors that trigger reroutes are
simdetectordist = 0 #How far after the start of a road the detectors for reconstructing initial routing sim traffic state are. TODO I'm not actually using this when making detectors, I just assume they're at start of lane. But then they miss all the cars, so I'm just faking those detectors anyway

#Hyperparameters for multithreading
multithreadRouting = True #Do each routing simulation in a separate thread. Enable for speed, but can mess with profiling
if not useLibsumo:
    multithreadRouting = False
multithreadSurtrac = False #Compute each light's Surtrac schedule in a separate thread. Enable for speed, but can mess with profiling
reuseSurtrac = False #Does Surtrac computations in a separate thread, shared between all vehicles doing routing. Keep this true unless we need everything single-threaded (ex: for debugging), or if running with fixed timing plans (routingSurtracFreq is huge) to avoid doing this computation
debugMode = True #Enables some sanity checks and assert statements that are somewhat slow but helpful for debugging
simToSimStats = False
routingSimUsesSUMO = True #Only switch this if we go back to custom routing simulator or something
mainSurtracFreq = 1 #Recompute Surtrac schedules every this many seconds in the main simulation (technically a period not a frequency). Use something huge like 1e6 to disable Surtrac and default to fixed timing plans.
routingSurtracFreq = 1 #Recompute Surtrac schedules every this many seconds in the main simulation (technically a period not a frequency). Use something huge like 1e6 to disable Surtrac and default to fixed timing plans.
recomputeRoutingSurtracFreq = 1 #Maintain the previously-computed Surtrac schedules for all vehicles routing less than this many seconds in the main simulation. Set to 1 to only reuse results within the same timestep. Does nothing when reuseSurtrac is False.
disableSurtracComms = True #Speeds up code by having Surtrac no longer predict future clusters for neighboring intersections
predCutoffMain = 20 #Surtrac receives communications about clusters arriving this far into the future in the main simulation
predCutoffRouting = 20 #Surtrac receives communications about clusters arriving this far into the future in the routing simulations
predDiscount = 1 #Multiply predicted vehicle weights by this because we're not actually sure what they're doing. 0 to ignore predictions, 1 to treat them the same as normal cars.

testNNdefault = False #Uses NN over Dumbtrac for light control if both are true
noNNinMain = False
debugNNslowness = False #Prints context information whenever loadClusters is slow, and runs the NN 1% of the time ignoring other NN settings
testDumbtrac = False #If true, overrides Surtrac with Dumbtrac (FTP or actuated control) in simulations and training data (if appendTrainingData is also true)
FTP = True #If false, and testDumbtrac = True, runs actuated control instead of fixed timing plans. If true, runs fixed timing plans (should now be same as SUMO defaults)
resetTrainingData = False#True
appendTrainingData = False#True
crossEntropyLoss = True

detectorModel = False #REMINDER: As currently implemented, turning this on makes even 0% and 100% routing non-deterministic, as we're guessing lanes for vehicles before running Surtrac
detectorSurtrac = detectorModel
detectorRouting = detectorModel
detectorRoutingSurtrac = detectorModel #If false, uses omniscient Surtrac in routing regardless of detectorSurtrac. If true, defers to detectorSurtrac
adopterComms = True
adopterCommsSurtrac = adopterComms
adopterCommsRouting = adopterComms

clusterStats = False #ONLY WORKS WITH REAL SURTRAC! If we want to record cluster stats when starting Surtrac calls for external use (ex: training NNs)
clusterNumsStats = []
clusterWeights = []
clusterLens = []
clusterGaps = []
firstClusterGaps = []

testNNrolls = []
nVehicles = []

learnYellow = False #False to strictly enforce that yellow lights are always their minimum length (no scheduling clusters during yellow+turn arrow, and ML solution isn't used there)
learnMinMaxDurations = False #False to strictly enforce min/max duration limits (in particular, don't call ML, just do the right thing)

#Don't change parameters below here
#For testing durations to see if there's drift between fixed timing plans executed in main simulation and routing simulations.
simdurations = dict()
simdurationsUsed = False
realdurations = dict()

dumpIntersectionData = False
intersectionData = dict()
vehicleIntersectionData = dict()

max_edge_speed = 0.0 #Overwritten when we read the route file

lanes = []
edges = []
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
lightlinkconflicts = dict()
nLanes = dict()
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
nonExitEdgeDetections = dict() #nonExitEdgeDetections[road] = array of sections of road, each with a detector in each lane at the start. Store (madeupname, lane, time) for all cars. If new vehicle in first road segment, look up where it came from and steal the oldest car from that lane segment, else the oldest car we can find. If non-first road segment, steal oldest from previous segment disregarding lane
nonExitLaneDetectors = dict() #nonExitLaneDetections[lane] = [(detectorname1, detectorpos1), ...], should be same length as nonExitEdgeDetections[road]
wasFull = dict() #We're now using this to store stats for lane transition probabilities (specifically, the times we saw vehicles)
wasFullWindow = 300
vehiclesOnNetwork = []
dontReroute = []
surtracDict = dict()
adopterinfo = dict()

#Predict traffic entering network
arrivals = dict()
maxarrivalwindow = 300 #Use negative number to not predict new incoming cars during routing
arrivals2 = dict()
maxarrivalwindow2 = 60 #Same as maxarrivalwindow if you just want the baseline arrival rate. Go smaller (~0.5-1 light cycles) if you want to predict initial off-network queues based on recent arrival rates being low
newcarcounter = 0

totalSurtracTime = 0
totalSurtracClusters = 0
totalSurtracRuns = 0

totalLoadTime = 0
totalLoadCars = 0
totalLoadRuns = 0

#Threading routing
toReroute = []
reroutedata = dict()
threads = dict()
killSurtracThread = True

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

        if debugMode:
            assert(simtime == traci.simulation.getTime())
        clustersCache = None #Clear stored clusters list

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
            edgeDict[vehicle] = traci.vehicle.getRoadID(vehicle)

            startDict[vehicle] = simtime
            leftDict[vehicle] = 0

            lane = laneDict[vehicle]
            if not lane in arrivals:
                arrivals[lane] = []
                arrivals2[lane] = []
            arrivals[lane].append(simtime) #Don't care who arrived, just when they arrived - this is for estimating future inflows if we turn that on in routing
            arrivals2[lane].append(simtime)

            #Manually pretend to be a detector at the start of the input lanes, since newly added vehicles have issues triggering those
            temp = lane.split("_")
            edge = temp[0]
            lanenum = int(temp[-1])
            if edge in nonExitEdgeDetections: #If it's an exit lane we hopefully don't care
                if isSmart[vehicle]:
                    nonExitEdgeDetections[edge][0].append((vehicle, lane, simtime))
                else:
                    nonExitEdgeDetections[edge][0].append((lane+"."+str(simtime), lane, simtime))

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

            #In case the target was a non-exit road, delete from old road detections
            # if edgeDict[vehicle] in nonExitEdgeDetections:
            #     oldEdgeStuff = nonExitEdgeDetections[edgeDict[vehicle]][0] #Since we're only storing stuff in index 0 anyway
            #     if len(oldEdgeStuff) > 0:
            #         oldEdgeStuff.pop(0) #Pop oldest from old road, don't care from which lane. Might not actually be the adopter in question
            #     else:
            #         print("Warning: Ran out of cars to remove on edge " + edgeDict[vehicle] + "!!!!!!!!!!!!!!!!!")

            #     #Make sure we don't have a duplicate of this adopter on the last edge. If we do, make it a random car instead
            #     if isSmart[vehicle]:
            #         for vehicletupleind in range(len(nonExitEdgeDetections[edgeDict[vehicle]][0])):
            #             vehicletuple = nonExitEdgeDetections[edgeDict[vehicle]][0][vehicletupleind]
            #             if vehicletuple[0] == vehicle:
            #                 nonExitEdgeDetections[edgeDict[vehicle]][0][vehicletupleind] = (edgeDict[vehicle]+".0oldsmartcar."+str(simtime), vehicletuple[1], vehicletuple[2]) #This seems to alias as intended

            edgeDict.pop(vehicle)
            laneDict.pop(vehicle)
            dontReroute.append(vehicle) #Vehicle has left network and does not need to be rerouted

        vehiclesOnNetwork = traci.vehicle.getIDList()
        carsOnNetwork.append(len(vehiclesOnNetwork)) #Store number of cars on network (for plotting)

        #Count left turns
        for id in laneDict:
            newlane = traci.vehicle.getLaneID(id)
            if len(newlane) == 0 or newlane[0] == ":":
                dontReroute.append(id) #Vehicle is mid-intersection or off network, don't try to reroute them
            if newlane != laneDict[id] and len(newlane) > 0 and  newlane[0] != ":":
                newloc = traci.vehicle.getRoadID(id)

                # #Pretend to be detectors at the start of each road (need to know where we came from so we can steal from the correct previous lane)
                # if newloc != edgeDict[id]: #Moved to a new road
                #     #Delete from old road detections
                #     if edgeDict[id] in nonExitEdgeDetections:
                #         oldEdgeStuff = nonExitEdgeDetections[edgeDict[id]][0] #Since we're only storing stuff in index 0 anyway
                #         if len(oldEdgeStuff) > 0:
                #             oldEdgeStuff.pop(0) #Pop oldest from old road, don't care from which lane. Might not actually be the adopter in question
                #         else:
                #             print("Warning: Ran out of cars to remove on edge " + edgeDict[id] + "!!!!!!!!!!!!!!!!!")

                #         #Make sure we don't have a duplicate of this adopter on the last edge. If we do, make it a random car instead
                #         if isSmart[id]:
                #             for vehicletupleind in range(len(nonExitEdgeDetections[edgeDict[id]][0])):
                #                 vehicletuple = nonExitEdgeDetections[edgeDict[id]][0][vehicletupleind]
                #                 if vehicletuple[0] == id:
                #                     nonExitEdgeDetections[edgeDict[id]][0][vehicletupleind] = (edgeDict[id]+".0oldsmartcar."+str(simtime), vehicletuple[1], vehicletuple[2]) #This seems to alias as intended

                #     #Add to new road detections
                #     if newloc in nonExitEdgeDetections:
                #         assert(newlane.split("_")[0] == newloc)
                #         if isSmart[id]:
                #             nonExitEdgeDetections[newloc][0].append((id, newlane, simtime))
                #         else:
                #             nonExitEdgeDetections[newloc][0].append((newlane+".0maindetect."+str(simtime), newlane, simtime))

                # #Remove vehicle from predictions, since the next intersection should actually see it now
                # if not disableSurtracComms:
                #     removeVehicleFromPredictions(sumoPredClusters, edgeDict[id])

                c0 = network.getEdge(edgeDict[id]).getFromNode().getCoord()
                c1 = network.getEdge(edgeDict[id]).getToNode().getCoord()
                theta0 = math.atan2(c1[1]-c0[1], c1[0]-c0[0])

                c2 = network.getEdge(newloc).getToNode().getCoord()
                theta1 = math.atan2(c2[1]-c1[1], c2[0]-c1[0])

                if (theta1-theta0+math.pi)%(2*math.pi)-math.pi > 0:
                    leftDict[id] += 1
                laneDict[id] = newlane
                edgeDict[id] = newloc
                #assert(laneDict[id] == traci.vehicle.getLaneID(id))
                try:
                    assert(laneDict[id].split("_")[0] in currentRoutes[id])
                except Exception as e:
                    print("Vehicle got off route somehow?")
                    #print(traci.getLabel()) #Should be main
                    print(id)
                    print(laneDict[id])
                    print(traci.vehicle.getLaneID(id))
                    print(currentRoutes[id])
                    print(traci.vehicle.getRoute(id))
                    #assert(traci.getLabel() == "main")
                    #raise(e)
                    laneDict[id] = traci.vehicle.getLaneID(id)
                    currentRoutes[id] = traci.vehicle.getRoute(id)
                    assert(laneDict[id].split("_")[0] in currentRoutes[id])

                #Store data to compute delay after first intersection
                if not id in delay2adjdict:
                    delay2adjdict[id] = simtime

                #Compute distance travelled if on last edge of route (since we can't do this once we leave the network)
                if newlane.split("_")[0] == currentRoutes[id][-1]:
                    routeStats[id]["distance"] = traci.vehicle.getDistance(id) + lengths[newlane]

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

                try:
                    if isSmart[id]:
                        avgerror += ((timedata[id][1]-timedata[id][0]) - timedata[id][2])/nSmart
                        avgabserror += abs((timedata[id][1]-timedata[id][0]) - timedata[id][2])/nSmart
                        avgpcterror += ((timedata[id][1]-timedata[id][0]) - timedata[id][2])/(timedata[id][1]-timedata[id][0])/nSmart*100
                        avgabspcterror += abs((timedata[id][1]-timedata[id][0]) - timedata[id][2])/(timedata[id][1]-timedata[id][0])/nSmart*100
                except:
                    print("Missing timedata for vehicle " + id)
                    print(isSmart[id])
                    print(timedata[id])

            if verbose or not traci.simulation.getMinExpectedNumber() > 0 or (appendTrainingData and simtime == 5000):
                print(pSmart)
                print("\nCurrent simulation time: %f" % simtime)
                print("Total run time: %f" % (time.time() - tstart))
                print("Number of vehicles in network: %f" % traci.vehicle.getIDCount())
                print("Total cars that left the network: %f" % len(endDict))
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
                if clusterStats:
                    if len(clusterNumsStats) > 1:
                        print("Cluster numbers: %f +/- %f", np.mean(clusterNumsStats), np.std(clusterNumsStats))
                    if len(clusterLens) > 1:
                        print("Cluster lengths: %f +/- %f", np.mean(clusterLens), np.std(clusterLens))
                    if len(clusterGaps) > 1:
                        print("Cluster gaps: %f +/- %f", np.mean(clusterGaps), np.std(clusterGaps))
                    if len(firstClusterGaps) > 1:
                        print("First cluster gaps: %f +/- %f", np.mean(firstClusterGaps), np.std(firstClusterGaps))
                    if len(clusterWeights) > 1:
                        print("Cluster weights: %f +/- %f", np.mean(clusterWeights), np.std(clusterWeights))

                if nSmart > 0:
                    print("Average error (actual minus expected) in predicted travel time: %f" % (avgerror))
                    print("Average absolute error in predicted travel time: %f" % (avgabserror))
                    print("Average percent error in predicted travel time: %f" % (avgpcterror))
                    print("Average absolute percent error in predicted travel time: %f" % (avgabspcterror))

                    if len(routeVerifyData) > 0:
                        avgVerifyError = 0
                        avgAbsVerifyError = 0
                        avgPctVerifyError = 0
                        avgPctAbsVerifyError = 0
                        nVerifyPts = 0
                        for verifytuple in routeVerifyData:
                            if verifytuple[0] < 0 or verifytuple[1] < 0: #If routing times out, these return -1
                                continue
                            avgVerifyError += (verifytuple[0]-verifytuple[1])
                            avgAbsVerifyError += abs((verifytuple[0]-verifytuple[1]))
                            avgPctVerifyError += (verifytuple[0]-verifytuple[1])/verifytuple[0]*100
                            avgPctAbsVerifyError += abs((verifytuple[0]-verifytuple[1]))/verifytuple[0]*100
                            nVerifyPts += 1
                        avgVerifyError /= nVerifyPts
                        avgAbsVerifyError /= nVerifyPts
                        avgPctVerifyError /= nVerifyPts
                        avgPctAbsVerifyError /= nVerifyPts
                        print("Average sim-to-sim error (actual minus expected) in predicted travel time: %f" % (avgVerifyError))
                        print("Average absolute sim-to-sim error (actual minus expected) in predicted travel time: %f" % (avgAbsVerifyError))
                        print("Average percent sim-to-sim error (actual minus expected) in predicted travel time: %f" % (avgPctVerifyError))
                        print("Average absolute percent sim-to-sim error (actual minus expected) in predicted travel time: %f" % (avgPctAbsVerifyError))
                            
                    print("Proportion of cars that changed route at least once: %f" % (nswapped/nSmart))
                    print("Average number of teleports: %f" % (nsmartteleports/nSmart))
                    print("Average distance travelled: %f" % (totaldistanceSmart/nSmart))
                    print("Average number of calls to routing: %f" % (totalcalls/nSmart)) #NOTE: This only counts calls done by cars that exited
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

                print("Among non-adopters:")
                print("Average delay: %f" % avgTimeNot)
                print("Best delay: %f" % bestTimeNot)
                print("Worst delay: %f" % worstTimeNot)
                print("Average number of lefts: %f" % avgLeftsNot)
                if nCars - nSmart > 0:
                    print("Average number of teleports: %f" % (nnotsmartteleports/(nCars-nSmart)))
                    print("Average distance travelled: %f" % (totaldistanceNot/(nCars-nSmart)))

                print("Routing calls (including cars not yet exited): " + str(nRoutingCalls))
                print("Total routing time: " + str(routingTime))
                if nRoutingCalls > 0:
                    print("Average time per call: " + str(routingTime/nRoutingCalls))
                    print("Proportion of successful routing calls: " + str(nSuccessfulRoutingCalls/nRoutingCalls))
                if debugMode:
                    print("Surtrac calls (one for each light): " + str(totalSurtracRuns))
                    if totalSurtracRuns > 0:
                        print("Average time per Surtrac call: " + str(totalSurtracTime/totalSurtracRuns))
                        print("Average number of clusters per Surtrac call: " + str(totalSurtracClusters/totalSurtracRuns))
                    print("loadClusters calls: " + str(totalLoadRuns))
                    if totalLoadRuns > 0:
                        print("Average time per loadClusters call: " + str(totalLoadTime/totalLoadRuns))
                        print("Average cars per loadClusters call: " + str(totalLoadCars/totalLoadRuns))
                print("\n")

                for lane in arrivals:
                    while len(arrivals[lane]) > 0 and arrivals[lane][0] < simtime - maxarrivalwindow:
                        arrivals[lane] = arrivals[lane][1:]

                for lane in arrivals2:
                    while len(arrivals2[lane]) > 0 and arrivals2[lane][0] < simtime - maxarrivalwindow2:
                        arrivals2[lane] = arrivals2[lane][1:]

    #Dump intersection data to Excel
    if dumpIntersectionData:
        dumpIntersectionDataFun(intersectionData, network)

    try:
        os.remove("savestates/teststate_"+savename+".xml") #Delete the savestates file so we don't have random garbage building up over time
    except FileNotFoundError:
        print("Warning: Trying to clean up savestates file, but no file found. This is weird - did you comment out routing or something? Ignoring for now.")
        pass

    return [avgTime, avgTimeSmart, avgTimeNot, avgTime2, avgTimeSmart2, avgTimeNot2, avgTime3, avgTimeSmart3, avgTimeNot3, avgTime0, avgTimeSmart0, avgTimeNot0, time.time()-tstart, nteleports, teleportdata]  


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

    if useLibsumo:
        traci.start([checkBinary('sumo'), "-c", sumoconfig,
                                    "--start", "--no-step-log", "true",
                                    "--xml-validation", "never", "--quit-on-end"])
    else:
        try:
            traci.start([checkBinary('sumo'), "-c", sumoconfig,
                                    "--start", "--no-step-log", "true",
                                    "--xml-validation", "never", "--quit-on-end"], label="setup")
        except:
            #Worried about re-calling this without old setup instance being removed

            #Need to reload in case we're training over multiple networks
            traci.switch("setup")
            traci.load(["-c", sumoconfig,
                                    "--start", "--no-step-log", "true",
                                    "--xml-validation", "never", "--quit-on-end"])
            pass

    net = sumolib.net.readNet(networkfile)
    rerouters = dict()
    global max_edge_speed

    #Copying this from run() so I can use these in here too. Annoyingly, lengths needs a SUMO simulation to compute, and the SUMO sim needs to know about the additional file, so ordering these is annoying
    #Edges have speeds, but lanes have lengths, so it's a little annoying to get fftimes...
    edges = traci.edge.getIDList()
    lanes = traci.lane.getIDList()

    for lane in lanes:
        if not lane[0] == ":":
            links[lane] = traci.lane.getLinks(lane)
            lengths[lane] = traci.lane.getLength(lane)

    for edge in edges:
        nLanes[edge] = traci.edge.getLaneNumber(edge)
        if not edge[0] == ":":
            speeds[edge] = net.getEdge(edge).getSpeed()
            fftimes[edge] = lengths[edge+"_0"]/speeds[edge]

    with open("additional_autogen.xml", "w") as additional:
        print("""<additional>""", file=additional)
        if not useLibsumo:
            print('    <edgeData id="%s" file="%s" period="%i"/>' % (savename, "edgedata/"+savename+".xml", 1e6), file=additional)
            print('    <laneData id="%s" file="%s" period="%i"/>' % (savename, "lanedata/"+savename+".xml", 1e6), file=additional)
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
                    rerouters["IL_"+lane] = lane
                    rerouterLanes["IL_"+lane] = lane
                    rerouterEdges["IL_"+lane] = edge

                if len(net.getEdge(edge).getOutgoing()) > 0:
                    nonExitEdgeDetections[edge] = []
                    nonExitLaneDetectors[lane] = []
                    for dist in [0, lengths[lane]-2]: #Add to this if we need more detectors, remember to update it both here and below in additionalrouting_autogen
                        name = "ILd_" + lane + "_" + str(dist)
                        print('    <inductionLoop id="%s" freq="1" file="outputAuto.xml" lane="%s" pos="%i" friendlyPos="true" />' \
                        % (name, lane, dist), file=additional)
                        nonExitEdgeDetections[edge].append([]) #This is inefficient since we keep clearing and rebuilding nonExitEdgeDetections[edge] once per lane, but shouldn't be that terrible
                        nonExitLaneDetectors[lane].append((name, dist))
                        wasFull[name] = []
        print("</additional>", file=additional)

    with open("additionalrouting_autogen.xml", "w") as additional:
        print("""<additional>""", file=additional)
        for edge in traci.edge.getIDList():
            if edge[0] == ":":
                #Skip internal edges (=edges for the inside of each intersection)
                continue

            for lanenum in range(traci.edge.getLaneNumber(edge)):
                lane = edge+"_"+str(lanenum)
                print('    <inductionLoop id="IL_%s" freq="1" file="outputAuto.xml" lane="%s" pos="-%i" friendlyPos="true" />' \
                      % (lane, lane, detectordist), file=additional)
                if len(net.getEdge(edge).getOutgoing()) > 0:
                    for dist in [0, lengths[lane]-2]: #Add to this if we need more detectors, remember to update it both here and above in additional_autogen
                        name = "ILd_" + lane + "_" + str(dist)
                        print('    <inductionLoop id="%s" freq="1" file="outputAuto.xml" lane="%s" pos="%i" friendlyPos="true" />' \
                        % (name, lane, dist), file=additional)
        print("</additional>", file=additional)
    
    return rerouters

def sampleRouteFromTurnData(startlane, turndata):
    lane = startlane
    route = [lane.split("_")[0]]
    while lane in turndata:
        r = random.random()

        oops = True
        for nextlane in turndata[lane]:
            r -= turndata[lane][nextlane]
            if r <= 0:
                if nextlane.split("_")[0] == lane.split("_")[0]:
                    print("Warning: Sampling is infinite looping, stopping early")
                    return route

                #Check if lane connects to nextlane
                for nextlinktuple in links[lane]:
                    tempnextedge = nextlinktuple[0].split("_")[0]
                    if nextlane.split("_")[0] == tempnextedge:
                        oops = False
                        break
                if oops:
                    print("sampleRouteFromTurnData found an invalid connection??? Trying again...")
                    print(lane + " -> " + nextlane)
                    break
                # else:
                #     print("Yay, did a thing!!!")
                #     print(lane + " -> " + nextlane)
                lane = nextlane
                break
        if not oops:
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
    rerouters = generate_additionalfile(sumoconfig, netfile)


    # this is the normal way of using traci. sumo is started as a
    # subprocess and then the python script connects and runs
    if useLibsumo:
        traci.load(["-c", sumoconfig,
                                "--additional-files", "additional_autogen.xml",
                                "--no-step-log", "true",
                                "--log", "LOGFILE", "--xml-validation", "never", "--start", "--quit-on-end"])
    else:
        try:
            traci.start([sumoBinary, "-c", sumoconfig,
                                    "--additional-files", "additional_autogen.xml",
                                    "--no-step-log", "true",
                                    "--log", "LOGFILE", "--xml-validation", "never", "--start", "--quit-on-end"], label="main")
            #Second simulator for running tests. No GUI
            #traci.start([sumoBinary, "-c", sumoconfig, #GUI in case we need to debug
            traci.start([checkBinary('sumo'), "-c", sumoconfig, #No GUI
                                    "--additional-files", "additionalrouting_autogen.xml",
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
        nLanes[edge] = traci.edge.getLaneNumber(edge)
        if not edge[0] == ":":
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
            surtracdata[light][i]["minDur"] = 7#5#lightphasedata[light][i].minDur #Min duration of yellow
            surtracdata[light][i]["maxDur"] = 120#lightphasedata[light][i].maxDur #Max duration of everything

            if "G" in lightphasedata[light][i].state or "g" in lightphasedata[light][i].state:
                surtracdata[light][i]["minDur"] = 5 #1#3.5#5#lightphasedata[light][i].minDur #Min duration of green
            # if "y" in lightphasedata[light][i].state:
            #     surtracdata[light][i]["minDur"] = 7 #1#3.5#5#lightphasedata[light][i].minDur #Min duration of yellow
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

                linklist = linklistlist[linklistind]
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
    print("testNN="+str(testNN))
    for light in ["light"]:#lights:
        if resetTrainingData:
            trainingdata[light] = []

        if testNNdefault:
            #NOTE: These are also hardcoded in the convertToNNInputSurtrac function
            maxnlanes = 3 #Going to assume we have at most 3 lanes per road, and that the biggest number lane is left-turn only
            maxnroads = 4 #And assume 4-way intersections for now
            maxnclusters = 5 #And assume at most 10 clusters per lane
            ndatapercluster = 3 #Arrival, departure, weight
            maxnphases = 12 #Should be enough to handle both leading and lagging lefts
            
            nextra = 1 #Proportion of phase length used
            ninputs = maxnlanes*maxnroads*maxnclusters*ndatapercluster + maxnlanes*maxnroads*maxnphases + maxnphases + nextra

            if crossEntropyLoss:
                agents[light] = Net(ninputs, 2, 128)
            else:
                agents[light] = Net(ninputs, 1, 128)
            # if testDumbtrac:
            #     # agents[light] = Net(26, 1, 32)
            #     # #agents[light] = Net(2, 1, 32)
            #     # if FTP:
            #     agents[light] = Net(182, 1, 64)
            # else:
            #     agents[light] = Net(182, 1, 64)
            optimizers[light] = torch.optim.Adam(agents[light].parameters(), lr=learning_rate)
            MODEL_FILES[light] = 'models/imitate_'+light+'.model' # Once your program successfully trains a network, this file will be written
            print("Checking if there's a learned model. Currently testNN="+str(testNN))
            try:
                checkpoint = torch.load(MODEL_FILES[light], weights_only=True)
                agents[light].load_state_dict(checkpoint['model_state_dict'])
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
            for light in ["light"]:#lights:
                trainingdata[light] = []
    
    outdata = run(network, rerouters, pSmart, verbose)
    
    if appendTrainingData:
        print("Saving training data")
        with open("trainingdata/trainingdata_" + sys.argv[1] + ".pickle", 'wb') as handle:
            pickle.dump(trainingdata, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #traci.close()

    p = pSmart
    newdata = outdata
    try:
        with open("delaydata/delaydata_" + sys.argv[1] + ".pickle", 'rb') as handle:
                data = pickle.load(handle)
    except:
        #If no data found, start fresh
        data = dict()
    if not p in data:
        data[p] = dict()
    for l in ["All", "Adopters", "Non-Adopters", "All2", "Adopters2", "Non-Adopters2", "All3", "Adopters3", "Non-Adopters3", "All0", "Adopters0", "Non-Adopters0", "Runtime", "NTeleports", "TeleportData", "RNGStates"]:
        if not l in data[p]:
            data[p][l] = []

    data[p]["All"].append(newdata[0])
    data[p]["Adopters"].append(newdata[1])
    data[p]["Non-Adopters"].append(newdata[2])
    data[p]["All2"].append(newdata[3])
    data[p]["Adopters2"].append(newdata[4])
    data[p]["Non-Adopters2"].append(newdata[5])
    data[p]["All3"].append(newdata[6])
    data[p]["Adopters3"].append(newdata[7])
    data[p]["Non-Adopters3"].append(newdata[8])
    data[p]["All0"].append(newdata[9])
    data[p]["Adopters0"].append(newdata[10])
    data[p]["Non-Adopters0"].append(newdata[11])
    data[p]["Runtime"].append(newdata[12])
    data[p]["NTeleports"].append(newdata[13])
    data[p]["TeleportData"].append(newdata[14])
    
    data[p]["RNGStates"].append(rngstate)
    with open("delaydata/delaydata_" + sys.argv[1] + ".pickle", 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return [outdata, rngstate]

def getDTheta(startedge, nextedge, network):
    #Preprocess in case we accidentally pass in lanes instead of edges
    startedge = startedge.split("_")[0]
    nextedge = nextedge.split("_")[0]

    c0 = network.getEdge(startedge).getFromNode().getCoord()
    c1 = network.getEdge(startedge).getToNode().getCoord()
    theta0 = math.atan2(c1[1]-c0[1], c1[0]-c0[0])

    c2 = network.getEdge(nextedge).getToNode().getCoord()
    theta1 = math.atan2(c2[1]-c1[1], c2[0]-c1[0])

    dtheta = (theta1-theta0+math.pi)%(2*math.pi)-math.pi
    return dtheta

def getLeftEdge(startlane, network):
    #Returns the leftmost edge with a connection from startlane
    startedge = startlane.split("_")[0]
    leftedge = None
    maxleft = -np.inf
    for nextlinktuple in links[startlane]:
        #Add to all lanes on that edge
        nextedge = nextlinktuple[0].split("_")[0]

        dtheta = getDTheta(startedge, nextedge, network)
        if dtheta > maxleft:
            leftedge = nextedge
            maxleft = dtheta
    return leftedge

#@profile
#NOTE: Profiling says this function isn't terrible, probably don't need to speed it up right now
def removeVehicleFromPredictions(sumoPredClusters, lastedge):
    for predlane in sumoPredClusters.keys():
        #Don't bother checking lanes on edges that weren't the last edge
        if not predlane.split("_")[0] == lastedge:
            continue

        predclusterind = 0
        while predclusterind < len(sumoPredClusters[predlane]):
            predcluster = sumoPredClusters[predlane][predclusterind]

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

            #Sanity check
            if debugMode:
                weightsum = 0
                for predcarind in range(len(predcluster["cars"])):
                    weightsum += predcluster["cars"][predcarind][2]
                assert(abs(weightsum - predcluster["weight"]) < 1e-10)

            #Remove empty clusters
            if len(predcluster["cars"]) == 0:
#                                sumoPredClusters[predlane].remove(predcluster)
                #sumoPredClusters[predlane].pop(predclusterind)
                sumoPredClusters[predlane] = sumoPredClusters[predlane][0:predclusterind]+sumoPredClusters[predlane][predclusterind+1:] #Apparently this is necessary when working with manager dict stuff? Maybe it's immutable??
            else:
                predcluster["arrival"] = minarr #predcluster["cars"][0][1]
                predcluster["departure"] = maxarr #predcluster["cars"][-1][1]
                predclusterind += 1


# Gets successor edges of a given edge in a given network
# Parameters:
#   edge: an edge ID string
#   network: the network object from sumolib.net.readNet(netfile)
# Returns:
#   successors: a list of edge IDs for the successor edges (outgoing edges from the next intersection)
def getSuccessors(edge, network):
    ids = []
    for succ in list(network.getEdge(edge).getOutgoing()):
        ids.append(succ.getID())
    return ids

def saveStateInfo(edge, remainingDuration, lastSwitchTimes, sumoPredClusters, lightphases):
    #Copy state from main sim to test sim
    traci.simulation.saveState("savestates/teststate_"+edge+".xml")
    #saveState apparently doesn't save traffic light states despite what the docs say
    #So save all the traffic light states and copy them over
    lightStates = dict()
    for light in traci.trafficlight.getIDList():
        lightStates[light] = [traci.trafficlight.getPhase(light), traci.trafficlight.getPhaseDuration(light)]
        #Why do the built-in functions have such terrible names?!
        lightStates[light][1] = traci.trafficlight.getNextSwitch(light) - traci.simulation.getTime()
    #Save lightStates to a file
    with open("savestates/lightstate_"+edge+".pickle", 'wb') as handle:
        pickle.dump(lightStates, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open("savestates/lightstate_"+edge+"_remainingDuration.pickle", 'wb') as handle:
        pickle.dump(remainingDuration, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open("savestates/lightstate_"+edge+"_lastSwitchTimes.pickle", 'wb') as handle:
        pickle.dump(lastSwitchTimes, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open("savestates/lightstate_"+edge+"_sumoPredClusters.pickle", 'wb') as handle:
        pickle.dump(sumoPredClusters, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open("savestates/lightstate_"+edge+"_lightphases.pickle", 'wb') as handle:
        pickle.dump(lightphases, handle, protocol=pickle.HIGHEST_PROTOCOL)

#prevedge is just used as part of the filename - can pass in a constant string so we overwrite, or something like a timestamp to support multiple instances of the code running at once
def loadStateInfo(prevedge, simtime, network): #simtime is just so I can pass it into loadStateInfoDetectors...
    if detectorRouting:
        return loadStateInfoDetectors(prevedge, simtime, network)

    #Load traffic state
    traci.simulation.loadState("savestates/teststate_"+prevedge+".xml")
    #Load light state
    with open("savestates/lightstate_"+prevedge+".pickle", 'rb') as handle:
        lightStates = pickle.load(handle)

    #Randomize non-adopter routes
    for lane in lanes:
        if len(lane) == 0 or lane[0] == ":":
            continue
        for vehicle in traci.lane.getLastStepVehicleIDs(lane):
            if not vehicle in isSmart or not isSmart[vehicle]:
                traci.vehicle.setRoute(vehicle, sampleRouteFromTurnData(lane, turndata))

    #Copy traffic light timings
    for light in traci.trafficlight.getIDList():
        traci.trafficlight.setPhase(light, lightStates[light][0])
        traci.trafficlight.setPhaseDuration(light, lightStates[light][1])
        #print(lightStates[light][1])
    with open("savestates/lightstate_"+prevedge+"_remainingDuration.pickle", 'rb') as handle:
        remainingDuration = pickle.load(handle)
    with open("savestates/lightstate_"+prevedge+"_lastSwitchTimes.pickle", 'rb') as handle:
        lastSwitchTimes = pickle.load(handle)
    with open("savestates/lightstate_"+prevedge+"_sumoPredClusters.pickle", 'rb') as handle:
        sumoPredClusters2 = pickle.load(handle)
    with open("savestates/lightstate_"+prevedge+"_lightphases.pickle", 'rb') as handle:
        lightphases = pickle.load(handle)
    return (remainingDuration, lastSwitchTimes, sumoPredClusters2, lightphases, deepcopy(edgeDict), deepcopy(laneDict))

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
