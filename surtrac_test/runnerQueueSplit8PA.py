#!/usr/bin/env python
# Eclipse SUMO, Simulation of Urban MObility; see https://eclipse.org/sumo
# Copyright (C) 2009-2021 German Aerospace Center (DLR) and others.
# This program and the accompanying materials are made available under the
# terms of the Eclipse Public License 2.0 which is available at
# https://www.eclipse.org/legal/epl-2.0/
# This Source Code may also be made available under the following Secondary
# Licenses when the conditions for such availability set forth in the Eclipse
# Public License 2.0 are satisfied: GNU General Public License, version 2
# or later which is available at
# https://www.gnu.org/licenses/old-licenses/gpl-2.0-standalone.html
# SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later

# @file    runner.py
# @author  Lena Kalleske
# @author  Daniel Krajzewicz
# @author  Michael Behrisch
# @author  Jakob Erdmann
# @date    2009-03-26


#QueueSplit5 added a first iteration of a Surtrac model to QueueSplit4
#New in QueueSplit6: When there's multiple light phases a lane can go in, don't double-create clusters
#New in QueueSplit7: Adding Surtrac to simulate-ahead
#New in QueueSplit8: Adding communication between intersections. Which then requires re-compacting clusters, etc. For now, don't take advantage of known routes


#Code is slow
#Slow part is QueueSim; slowest part of that is Surtrac (somehow). Even if predictionCutoff is 0, which should reduce to previous code. No single obvious slow line

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
from copy import deepcopy
from collections import Counter

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

isSmart = dict() #Store whether each vehicle does our routing or not
pSmart = 1.0 #Adoption probability

carsOnNetwork = []
max_edge_speed = 0.0

oldids = dict()

clusterthresh = 6 #Time between cars before we split to separate clusters
mingap = 2.5 #Minimum allowed space between cars
timestep = mingap #Amount of time between updates. In practice, mingap rounds up to the nearest multiple of this



lightphasedata = dict()
lightlinks = dict()
lightlanes = dict()
lightoutlanes = dict()
lights = []
edges = []
lightlinkconflicts = dict()
lanenums = dict()
speeds = dict()
links = dict()
lengths = dict()
turndata = []
timedata = dict()
surtracdata = dict()
lastswitchtimes = dict()

sumoPredClusters = None #This'll update when we call doSurtrac from sumo things

def mergePredictions(clusters, predClusters):
    mergedClusters = pickle.loads(pickle.dumps(clusters)) #Because pass-by-value stuff
    for lane in clusters:
        if lane in predClusters:
            #TODO later: Allen thesis says we might need to interpolate route-based clusters (any reason these wouldn't just be single vehicles? Maybe if an entire cluster was known to be going the same way)
            #That's using routes as super-long-term lookahead, rather than just not sampling next-road-forward predictions... not going to do that now
            mergedClusters[lane] += predClusters[lane] #Concatenate known clusters with predicted clusters
    return mergedClusters

#@profile
def doSurtrac(network, simtime, realclusters=None, lightinfo=None, predClusters=None):
    sult = 3 #Startup loss time
    predictionCutoff = 60 #Drop predicted clusters that'll happen more than this far in the future

    toSwitch = []
    catpreds = dict()
    inQueueSim = True
    if realclusters == None and lightinfo == None:
        inQueueSim = False
        (realclusters, lightinfo) = loadClusters(network)
    

    if not predClusters == None:
        clusters = mergePredictions(realclusters, predClusters)
    else:
        clusters = pickle.loads(pickle.dumps(realclusters))

    for light in lights:
        #Figure out what an initial and complete schedule look like
        nPhases = len(surtracdata[light]) #Number of phases

        emptyStatus = dict()
        fullStatus = dict()
        nClusters = 0
        for lane in lightlanes[light]:
            emptyStatus[lane] = 0
            fullStatus[lane] = len(clusters[lane])
            nClusters += fullStatus[lane]

        #Stuff in the partial schedule tuple
        #0: list of indices of the clusters we've scheduled
        #1: schedule status (how many clusters from each lane we've scheduled)
        #2: current light phase
        #3: time when each direction will have finished its last scheduled cluster
        #4: time when all directions are finished with scheduled clusters ("total makespan", but also add starting time...)
        #5: total delay
        #6: last switch time
        #7: planned total durations of all phases
        #8: predicted outflows (as clusters - arrival, departure, list of cars, weights, etc.)

        emptyPreds = dict()
        for lane in lightoutlanes[light]:
            emptyPreds[lane] = []

        phase = lightinfo[light]["index"]
        lastSwitch = lastswitchtimes[light]
        schedules = [([], emptyStatus, phase, [simtime]*len(surtracdata[light][phase]["lanes"]), simtime, 0, lastSwitch, [simtime - lastSwitch], emptyPreds)]
        
        
        for iternum in range(nClusters): #Keep adding a cluster until #clusters added = #clusters to be added
            scheduleHashDict = dict()
            for schedule in schedules:
                for lane in lightlanes[light]:
                    if schedule[1][lane] == fullStatus[lane]:
                        continue
                    #Consider adding next cluster from surtracdata[light][i]["lanes"][j] to schedule
                    newScheduleStatus = pickle.loads(pickle.dumps(schedule[1])) #deepcopy(schedule[1])
                    newScheduleStatus[lane] += 1
                    phase = schedule[2]

                    #Now loop over all phases where we can clear this cluster
                    for i in range(nPhases):
                        if not lane in surtracdata[light][i]["lanes"]:
                            #Don't consider phases where this cluster can't go
                            continue

                        nLanes = len(surtracdata[light][i]["lanes"])
                        j = surtracdata[light][i]["lanes"].index(lane)

                        newDurations = pickle.loads(pickle.dumps(schedule[7])) #deepcopy(schedule[7])

                        clusterind = newScheduleStatus[lane]-1 #We're scheduling the Xth cluster; it has index X-1
                        ist = clusters[lane][clusterind]["arrival"] #Intended start time = cluster arrival time
                        dur = clusters[lane][clusterind]["departure"] - ist
                        mindur = max((clusters[lane][clusterind]["weight"] - 1)*mingap, 0) #-1 because fencepost problem


                        if phase == i:
                            pst = schedule[3][j]
                            newLastSwitch = schedule[6] #Last switch time doesn't change
                            ast = max(ist, pst)
                            newdur = max(dur - (ast-ist), mindur)
                            currentDuration = max(ist, ast)+newdur-schedule[6] #Total duration of current light phase if we send this cluster without changing phase
                            
                        if not phase == i or currentDuration > surtracdata[light][i]["maxDur"]:
                            #Have to switch light phases. TODO: In the event that we exceeded maxDur, I'm assuming we switch immediately, though it would be better to split the cluster at maxDur
                            newFirstSwitch = max(schedule[6] + surtracdata[light][phase]["minDur"], schedule[4])
                            pst = newFirstSwitch + surtracdata[light][(phase+1)%nPhases]["timeTo"][i] + sult #Total makespan + switching time + startup loss time
                            #TODO: Technically this sult implementation isn't quite right, as a cluster might reach the light as the light turns green and not have to stop and restart
                            directionalMakespans = [pst]*nLanes #Other directions can't schedule a cluster before the light switches
                            newLastSwitch = newFirstSwitch + surtracdata[light][(phase+1)%nPhases]["timeTo"][i] #Switch right after previous cluster finishes (why not when next cluster arrives minus sult? Maybe try both?)
                            newDurations[-1] = newFirstSwitch - schedule[6] #Previous phase lasted from when it started to when it switched
                            tempphase = (phase+1)%nPhases
                            while tempphase != i:
                                newDurations.append(surtracdata[light][i]["minDur"])
                                tempphase = (tempphase+1)%nPhases
                            newDurations.append(0) #Duration of new phase i. To be updated on future loops once we figure out when the cluster finishes

                        #TODO Compress cluster as it goes through intersection (affects delay computation)
                        ast = max(ist, pst)
                        newdur = max(dur - (ast-ist), mindur)
                        directionalMakespans = [pst]*nLanes #Other directions can't schedule a cluster before this one
                        directionalMakespans[j] = ast+newdur
                        delay = schedule[5] + clusters[surtracdata[light][i]["lanes"][j]][clusterind]["weight"]*((ast-ist)-1/2*(dur-newdur)) #Delay += #cars * (actual-desired). 1/2(dur-newdur) compensates for the cluster packing together as it waits (I assume uniform compression)
                        newMakespan = max(directionalMakespans)
                        currentDuration = newMakespan - newLastSwitch

                        newPredClusters = pickle.loads(pickle.dumps(schedule[8]))
                        predLanes = []
                        for outlane in lightoutlanes[light]: #turndata[lane]:
                            newPredCluster = dict()
                            newPredCluster["pos"] = 0
                            newPredCluster["time"] = ast
                            newPredCluster["arrival"] = ast + lengths[outlane]/speeds[outlane.split("_")[0]] #.split("_")[0] to extract the edge name
                            if newPredCluster["arrival"] > simtime + predictionCutoff:
                                break #Skip predicted clusters that are super far in the future
                            #TODO recompress if ast > ist
                            newPredCluster["departure"] = newPredCluster["arrival"] + newdur
                            #TODO deal with known-route cars better
                            newPredCluster["cars"] = clusters[lane][clusterind]["cars"]
                            #TODO deal with known-route cars better
                            newPredCluster["weight"] = len(newPredCluster["cars"]) * turndata[lane][outlane]
                            newPredClusters[outlane].append(newPredCluster)
                            predLanes.append(outlane) #Track which lanes' clusters are within the prediction cutoff

                            

                        newDurations[-1] = currentDuration 

                        newschedule = (schedule[0]+[(i, j)], newScheduleStatus, i, directionalMakespans, newMakespan, delay, newLastSwitch, newDurations, newPredClusters)

                        #Turn schedule status from list-of-lists to tuple-of-tuples, then stick that in a tuple with the phase and use that as the key
                        key = (tuple(tuple(sub) for sub in schedule[1]), newschedule[2]) #From: https://www.geeksforgeeks.org/python-convert-list-of-lists-to-tuple-of-tuples/
                        if not key in scheduleHashDict:
                            scheduleHashDict[key] = [newschedule]
                        else:
                            keep = True
                            testscheduleind = 0
                            while testscheduleind < len(scheduleHashDict[key]):
                                testschedule = scheduleHashDict[key][testscheduleind]
                                
                                greedy = True
                                #NOTE: If we're going to go for truly optimal, I think we also need to check all makespans, plus the current phase duration
                                #OTOH, if people seem to think fast greedy approximations are good enough, I'm fine with that
                                if newschedule[5] >= testschedule[5] and (greedy or newschedule[4] >= testschedule[4]):
                                    #New schedule was dominated; remove it and don't continue comparing (old schedule beats anything new one would)
                                    keep = False
                                    break
                                if newschedule[5] <= testschedule[5] and (greedy or newschedule[4] <= testschedule[4]):
                                    #Old schedule was dominated; remove it
                                    scheduleHashDict[key].pop(testscheduleind)
                                    continue
                                #No dominance, keep going
                                testscheduleind += 1

                            if keep:
                                scheduleHashDict[key].append(newschedule)

            schedules = sum(list(scheduleHashDict.values()), []) #Each key has a list of non-dominated partial schedules. list() turns the dict_values object into a list of those lists; sum() concatenates to one big list of partial schedules. (Each partial schedule is stored as a tuple)

        mindelay = np.inf
        bestschedule = None
        for schedule in schedules:
            if schedule[5] < mindelay:
                mindelay = schedule[5]
                bestschedule = schedule

        if not bestschedule[0] == []:
            #remainingDuration = traci.trafficlight.getNextSwitch(light) - simtime
            spentDuration = simtime - lastswitchtimes[light]

            if bestschedule[7][0] - spentDuration > 0 and not inQueueSim:
                #Update duration
                traci.trafficlight.setPhaseDuration(light, bestschedule[7][0] - spentDuration) #setPhaseDuration sets the remaining duration in the phase
            else:
                #Light needs to change
                toSwitch.append(light)
                if not inQueueSim:
                    traci.trafficlight.setPhase(light, (traci.trafficlight.getPhase(light)+1)%nPhases) #Increment phase, duration defaults to default
                    lastswitchtimes[light] = simtime
                    if len(bestschedule[7]) > 1:
                        #And set the new duration if possible
                        traci.trafficlight.setPhaseDuration(light, bestschedule[7][1]) #Update duration if we know it
                        #pass
        catpreds.update(bestschedule[8])

    return (toSwitch, catpreds)
                    

def run(network, rerouters, pSmart, verbose = True):
    global sumoPredClusters
    #netfile is the filepath to the network file, so we can call sumolib to get successors
    #rerouters is the list of induction loops on edges with multiple successor edges
    #We want to reroute all the cars that passed some induction loop in rerouters using A*
    
    """execute the TraCI control loop"""
    
    startDict = dict()
    endDict = dict()
    locDict = dict()
    leftDict = dict()

    tstart = time.time()
    simtime = 0
    while traci.simulation.getMinExpectedNumber() > 0:
        simtime += 1

        traci.simulationStep() #Tell the simulator to simulate the next time step
        temp = doSurtrac(network, simtime, None, None, sumoPredClusters)
        sumoPredClusters = temp[1]

        #Decide whether new vehicles use our routing
        for vehicle in traci.simulation.getDepartedIDList():
            isSmart[vehicle] = random.random() < pSmart
            timedata[vehicle] = [simtime, -1, -1]
        for vehicle in traci.simulation.getArrivedIDList():
            timedata[vehicle][1] = simtime
            #print("Actual minus expected:")
            #print( (timedata[vehicle][1]-timedata[vehicle][0]) - timedata[vehicle][2])

        reroute(rerouters, network, simtime, True) #Reroute cars (including simulate-ahead cars)
        carsOnNetwork.append(len(traci.vehicle.getIDList())) #Track number of cars on network (for plotting)
        
        for id in traci.simulation.getDepartedIDList():
            startDict[id] = simtime
            locDict[id] = traci.vehicle.getRoadID(id)
            leftDict[id] = 0
        for id in traci.simulation.getArrivedIDList():
            endDict[id] = simtime
            locDict.pop(id)

        #Count left turns
        for id in locDict:
            if traci.vehicle.getRoadID(id) != locDict[id] and traci.vehicle.getRoadID(id)[0] != ":":
                c0 = network.getEdge(locDict[id]).getFromNode().getCoord()
                c1 = network.getEdge(locDict[id]).getToNode().getCoord()
                theta0 = math.atan2(c1[1]-c0[1], c1[0]-c0[0])

                #assert(c1 == network.getEdge(traci.vehicle.getRoadID(id)).getFromNode().getCoord())
                c2 = network.getEdge(traci.vehicle.getRoadID(id)).getToNode().getCoord()
                theta1 = math.atan2(c2[1]-c1[1], c2[0]-c1[0])

                if (theta1-theta0+math.pi)%(2*math.pi)-math.pi > 0:
                    leftDict[id] += 1
                
                locDict[id] = traci.vehicle.getRoadID(id)

        #Plot and print things
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

            nSmart = 0
            for id in endDict:
                if isSmart[id]:
                    nSmart += 1
            avgTimeSmart = 0
            avgLeftsSmart = 0
            bestTimeSmart = inf
            worstTimeSmart = 0
            avgTimeNot = 0
            avgLeftsNot = 0
            bestTimeNot = inf
            worstTimeNot = 0

            for id in endDict:
                ttemp = endDict[id] - startDict[id]
                avgTime += ttemp/len(endDict)
                avgLefts += leftDict[id]/len(endDict)
                if ttemp > worstTime:
                    worstTime = ttemp
                if ttemp < bestTime:
                    bestTime = ttemp

                if isSmart[id]:
                    avgTimeSmart += ttemp/nSmart
                    avgLeftsSmart += leftDict[id]/nSmart
                    if ttemp > worstTimeSmart:
                        worstTimeSmart = ttemp
                    if ttemp < bestTimeSmart:
                        bestTimeSmart = ttemp
                else:
                    avgTimeNot += ttemp/(len(endDict)-nSmart)
                    avgLeftsNot += leftDict[id]/(len(endDict)-nSmart)
                    if ttemp > worstTimeNot:
                        worstTimeNot = ttemp
                    if ttemp < bestTimeNot:
                        bestTimeNot = ttemp

        
            if verbose:
                print("\nCurrent simulation time: %f" % simtime)
                print("Total run time: %f" % (time.time() - tstart))
                print("Average time in network: %f" % avgTime)
                print("Best time: %f" % bestTime)
                print("Worst time: %f" % worstTime)
                print("Average number of lefts: %f" % avgLefts)
                print("Among adopters:")
                print("Average time in network: %f" % avgTimeSmart)
                print("Best time: %f" % bestTimeSmart)
                print("Worst time: %f" % worstTimeSmart)
                print("Average number of lefts: %f" % avgLeftsSmart)
                print("Among non-adopters:")
                print("Average time in network: %f" % avgTimeNot)
                print("Best time: %f" % bestTimeNot)
                print("Worst time: %f" % worstTimeNot)
                print("Average number of lefts: %f" % avgLeftsNot)
    return [avgTime, avgTimeSmart, avgTimeNot]

    

#Tell all the detectors to reroute the cars they've seen
def reroute(rerouters, network, simtime, rerouteAuto=True):
    doAstar = True #Set to false to stick with SUMO default routing

    if doAstar:
        for r in rerouters:
            QueueReroute(r, network, simtime, rerouteAuto)

def QueueReroute(detector, network, simtime, rerouteAuto=True):

    ids = traci.inductionloop.getLastStepVehicleIDs(detector) #All vehicles to be rerouted
    if len(ids) == 0:
        #No cars to route, we're done here
        return

    # getRoadID: Returns the edge id the vehicle was last on
    edge = traci.vehicle.getRoadID(ids[0])
    
    for vehicle in ids:
        
        if rerouteAuto and detector in oldids and vehicle in oldids[detector]:
            #print("Duplicate car " + vehicle + " at detector " + detector)
            continue

        if rerouteAuto and isSmart[vehicle]:
            #Convert current state

            #tstart = time.time()

            data = doClusterSim(edge, network, vehicle, simtime)
            newroute = data[0]
            tcluster = data[1]
            #print(routes[vehicle])
            #print(edge)
            #print(tcluster)
            if timedata[vehicle][2] == -1:
                timedata[vehicle][2] = tcluster
                
            #traci.switch("main")
            traci.vehicle.setRoute(vehicle, newroute)
        #if rerouteAuto and not isSmart[vehicle]:
            #Should be doing nothing here, just testing for now
            #traci.vehicle.setRoute(vehicle, sampleRouteFromTurnData(vehicle, traci.vehicle.getRoadID(vehicle), turndata))
    if rerouteAuto:
        oldids[detector] = ids

def doClusterSim(prevedge, net, vehicle, simtime):
    loaddata = loadClusters(net)
    simtime = traci.simulation.getTime()

    return runClusters(net, simtime, vehicle, prevedge, loaddata)

def loadClusters(net):
    #Load locations of cars and current traffic light states into custom data structures
    lightinfo = dict()
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
                if len(clusters[lane]) > 0 and abs(clusters[lane][-1]["time"] - traci.simulation.getTime()) < clusterthresh and abs(clusters[lane][-1]["pos"] - traci.vehicle.getLanePosition(vehicle))/speeds[edge] < clusterthresh:
                    #Add to cluster. pos and time track newest added vehicle to see if the next vehicle merges
                    #Departure time (=time to fully clear cluster) increases, arrival doesn't
                    clusters[lane][-1]["pos"] = traci.vehicle.getLanePosition(vehicle)
                    clusters[lane][-1]["time"] = traci.simulation.getTime()
                    clusters[lane][-1]["departure"] = traci.simulation.getTime() + (lengths[lane]-clusters[lane][-1]["pos"])/speeds[edge]
                    clusters[lane][-1]["cars"].append((vehicle, clusters[lane][-1]["departure"], "Load append"))
                    clusters[lane][-1]["weight"] = len(clusters[lane][-1]["cars"])
                else:
                    #Else make a new cluster
                    newcluster = dict()
                    newcluster["pos"] = traci.vehicle.getLanePosition(vehicle)
                    newcluster["time"] = traci.simulation.getTime()
                    newcluster["arrival"] = traci.simulation.getTime() + (lengths[edge+"_0"]-newcluster["pos"])/speeds[edge]
                    newcluster["departure"] = newcluster["arrival"]
                    newcluster["cars"] = [(vehicle, newcluster["departure"], "Load new")]
                    newcluster["weight"] = len(newcluster["cars"])
                    clusters[lane].append(newcluster)
    
    #Traffic light info
    lightinfo = dict()
    for light in lights:
        lightinfo[light] = dict()
        lightinfo[light]["state"] = traci.trafficlight.getRedYellowGreenState(light)
        lightinfo[light]["switchtime"] = traci.trafficlight.getNextSwitch(light)
        lightinfo[light]["index"] = traci.trafficlight.getPhase(light)
    return (clusters, lightinfo)

#@profile
def runClusters(net, time, vehicleOfInterest, startedge, loaddata):
    #Store routes once at the start to save time
    routes = dict()
    vehicles = traci.vehicle.getIDList()

    for vehicle in vehicles:
        if isSmart[vehicle]:
            routes[vehicle] = traci.vehicle.getRoute(vehicle)
        else:
            routes[vehicle] = sampleRouteFromTurnData(vehicle, traci.vehicle.getLaneID(vehicle), turndata)

    goalEdge = routes[vehicleOfInterest][-1]
    splitinfo = dict()
    VOIs = [vehicleOfInterest]

    clusters = loaddata[0]
    lightinfo = loaddata[1]

    starttime = time
    edgelist = list(edges)
    edgeind = 0
    while edgeind < len(edgelist):
        if edgelist[edgeind][0] == ":":
            edgelist.pop(edgeind)
        else:
            edgeind += 1

    queueSimPredClusters = pickle.loads(pickle.dumps(sumoPredClusters)) #Initial predicted clusters are whatever SUMO's Surtrac thinks it is
    #Loop through time and simulate things
    while True:
        time += timestep

        #Update lights
        (toUpdate, queueSimPredClusters) = doSurtrac(net, time, clusters, lightinfo, queueSimPredClusters)

        #I'm now assuming we won't skip a phase between updates. Hopefully fine.
        for light in toUpdate:
            phases = lightphasedata[light]
            lightinfo[light]["index"] += 1
            if lightinfo[light]["index"] == len(phases):
                #After last phase, loop back to 0
                lightinfo[light]["index"] = 0
            phaseind = lightinfo[light]["index"]
            lightinfo[light]["switchtime"] += phases[phaseind].duration
            lightinfo[light]["state"] = phases[phaseind].state

        #Sanity check for debugging infinite loops where the vehicle of interest disappears
        notEmpty = False
        for thing in clusters:
            for thingnum in range(len(clusters[thing])):
                for testcartuple in clusters[thing][thingnum]["cars"]:
                    if testcartuple[0] in VOIs:
                        notEmpty = True
                        break
        if not notEmpty:
            print(VOIs)
            print(clusters)
            raise Exception("Can't find vehicle of interest!")
        #End sanity check

        blockingLinks = dict()
        reflist = pickle.loads(pickle.dumps(edgelist)) #deepcopy(edgelist) #Want to reorder edge list to handle priority stuff, but don't want to mess up the for loop indexing
        for edge in reflist:

            for lanenum in range(lanenums[edge]):
                lane = edge + "_" + str(lanenum)

                while len(clusters[lane]) > 0:
                    cluster = clusters[lane][0]

                    if cluster["arrival"] > time:
                        #This and future clusters don't arrive yet, done on this edge
                        break
                    if len(cluster["cars"]) == 0:
                        print("Warning: Empty cluster. This shouldn't happen")
                        clusters[lane].remove(cluster)
                        continue
                    cartuple = cluster["cars"][0]
                    
                    while cartuple[1] < time:
                        #Check if route is done; if so, stop
                        if cartuple[0] in VOIs and edge == goalEdge:
                            #Check if we're done simulating
                            splitroute = cartuple[0].split("|")
                            splitroute.pop(0)
                            fullroute = [startedge]
                            for routepart in splitroute:
                                fullroute.append(routepart.split("_")[0]) #Only grab the edge, not the lane
                            return (fullroute, time-starttime)
                        elif not cartuple[0] in VOIs and routes[cartuple[0]][-1] == edge:
                            cluster["cars"].pop(0) #Remove car from this edge
                            break

                        
                        #Add car to next edge. NOTE: Enforce merging collision etc. constraints here
                        node = net.getEdge(edge).getToNode().getID()
                        if not node in blockingLinks:
                            blockingLinks[node] = []
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

                        tempnextedges = pickle.loads(pickle.dumps(splitinfo[(cartuple[0], edge)])) #deepcopy(splitinfo[(cartuple[0], edge)])
                        for nextlane in tempnextedges:
                            nextedge = nextlane.split("_")[0]

                            #Check light state
                            if node in lights:
                                isGreenLight = False
                                linklistlist = lightlinks[node]
                                for linklistind in range(len(linklistlist)):
                                    linkstate = lightinfo[node]["state"][linklistind]

                                    if linkstate == "G" or linkstate == "g":
                                        linklist = linklistlist[linklistind]
                                        for linktuple in linklist:
                                            if linktuple[0] == lane and linktuple[1].split("_")[0] == nextedge: #If can go from this lane to next edge, it's relevant
                                                #Make sure we're not a g stream blocked by a G stream
                                                isBlocked = False
                                                if linkstate == "g":
                                                    for linklistind2 in range(len(linklistlist)):
                                                        linkstate2 = lightinfo[node]["state"][linklistind2]
                                                        if not linkstate2 == "G":
                                                            continue
                                                        
                                                        for linktuple2 in linklistlist[linklistind2]:
                                                            conflicting = lightlinkconflicts[node][linktuple][linktuple2] #Precomputed to save time 

                                                            if not conflicting:
                                                                continue
                                                            blocking = (linklistind2 in blockingLinks[node]) #NOTE: Not clear what it'd mean to have multiple tuples in a single link (docs say something about signal groups), this might need to change if that happens

                                                            willBlock = False
                                                            if len(clusters[linktuple2[0]]) > 0 and clusters[linktuple2[0]][0]["cars"][0][1] <= time:
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
                                                                    willBlock = blockerroute[blockerrouteind+1] == blockingEdge1
                                                            
                                                            if blocking or willBlock:
                                                                isBlocked = True
                                                                break
                                                        if isBlocked:
                                                            break
                                                if not isBlocked:
                                                    isGreenLight = True
                                                    break
                                        #Can shortcut the outer loop if we find a link that works
                                        if isGreenLight:
                                            break
                            else:
                                isGreenLight = True #Not a light, so assume a zipper and allow everything through

                            if not isGreenLight:
                                continue

                            #Check append to previous cluster vs. add new cluster
                            if len(clusters[nextlane]) > 0 and abs(clusters[nextlane][-1]["time"] - time) < clusterthresh and abs(clusters[nextlane][-1]["pos"])/speeds[nextedge] < clusterthresh:
                                
                                #Make sure there's no car on the new road that's too close
                                if not abs(clusters[nextlane][-1]["time"] - time) < mingap:
                                    #Add to cluster. pos and time track newest added vehicle to see if the next vehicle merges
                                    #Departure time (=time to fully clear cluster) increases, arrival doesn't
                                    #TODO eventually: Be more precise with time and position over partial timesteps, allowing me to use larger timesteps?
                                    clusters[nextlane][-1]["pos"] = 0
                                    clusters[nextlane][-1]["time"] = time
                                    clusters[nextlane][-1]["departure"] = time + lengths[nextlane]/speeds[nextedge]
                                    if cartuple[0] in VOIs:
                                        clusters[nextlane][-1]["cars"].append((cartuple[0]+"|"+nextlane, clusters[nextlane][-1]["departure"], "Zipper append"))
                                        VOIs.append(cartuple[0]+"|"+nextlane)
                                    else:
                                        clusters[nextlane][-1]["cars"].append((cartuple[0], clusters[nextlane][-1]["departure"], "Zipper append"))
                                    clusters[nextlane][-1]["weight"] += 1
                                else:
                                    #No space, try next lane
                                    continue
                            else:
                                #There is no cluster nearby
                                #So make a new cluster
                                newcluster = dict()
                                newcluster["pos"] = 0
                                newcluster["time"] = time
                                newcluster["arrival"] = time + lengths[nextlane]/speeds[nextedge]
                                newcluster["departure"] = newcluster["arrival"]
                                if cartuple[0] in VOIs:
                                    newcluster["cars"] = [(cartuple[0]+"|"+nextlane, newcluster["departure"], "Zipper new cluster")]
                                    VOIs.append(cartuple[0]+"|"+nextlane)
                                else:
                                    newcluster["cars"] = [(cartuple[0], newcluster["departure"], "Zipper new cluster")]
                                newcluster["weight"] = 1
                                clusters[nextlane].append(newcluster)

                            #We've added a car to nextedge_nextlanenum
                            blockingLinks[node].append(linklistind)
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
                            edgelist.append(edgelist.pop(edgelist.index(edge))) #Push edge to end of list to give it lower priority next time
                            cluster["cars"].pop(0) #Remove car from this edge
                            cluster["weight"] -= 1
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

def dontBreakEverything():
    #No-op function here to avoid blind copy-paste throwing errors
    print("Can comment the call to dontBreakEverything here1 - there's only one simulator")

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
                print('    <inductionLoop id="IL_%s" freq="1" file="outputAuto.xml" lane="%s" pos="-50" friendlyPos="true" />' \
                      % (lane, lane), file=additional)
                if len(net.getEdge(edge).getOutgoing()) > 1:
                    rerouters.append("IL_"+lane)
        print("</additional>", file=additional)
    
    return rerouters

#@profile
def sampleRouteFromTurnData(vehicle, startlane, turndata):
    lane = startlane
    route = [lane.split("_")[0]]
    while lane in turndata:
        r = random.random()
        for nextlane in turndata[lane]:
            r -= turndata[lane][nextlane]
            if r <= 0:
                lane = nextlane
                break
        route.append(lane.split("_")[0])
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


def main(sumoconfig, pSmart, verbose = True):
    #I don't know why these global calls are important, but at least some apparently are. Commenting the block triggers the "vehicle of interest not found" error
    global lightphasedata
    global lightlinks
    global lightlanes
    global lights
    global edges
    global lightlinkconflicts
    global lanenums
    global speeds
    global links
    global lengths
    global turndata
    global timedata
    options = get_options()

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
                                "--log", "LOGFILE", "--xml-validation", "never", "--start", "--quit-on-end"], label="main")
    except:
        #Worried about re-calling this without old main instance being removed
        traci.switch("main")
        traci.load( "-c", sumoconfig,
                                "--additional-files", "additional_autogen.xml",
                                "--log", "LOGFILE", "--xml-validation", "never", "--start", "--quit-on-end")

    lights = traci.trafficlight.getIDList()
    edges = traci.edge.getIDList()
    #Grab stuff once at the start to avoid slow calls to traci in the routing

    for light in lights:
        lightlinkconflicts[light] = dict()
        lightphasedata[light] = traci.trafficlight.getCompleteRedYellowGreenDefinition(light)[0].phases
        lightlinks[light] = traci.trafficlight.getControlledLinks(light)
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

                lightlinkconflicts[light][linktuple] = dict()
                for linklist2 in linklistlist:
                    for linktuple2 in linklist2:
                        lightlinkconflicts[light][linktuple][linktuple2] = isIntersecting( (network.getLane(linktuple[0]).getShape()[1], (net.getLane(linktuple[1]).getShape()[0])), 
                        (net.getLane(linktuple2[0]).getShape()[1], (network.getLane(linktuple2[1]).getShape()[0])) )
    
    #Surtrac data
    for light in lights:
        surtracdata[light] = []
        lastswitchtimes[light] = 0
        #Has lagging lefts

        n = len(lightphasedata[light])
        for i in range(n):
            surtracdata[light].append(dict())
            surtracdata[light][i]["minDur"] = lightphasedata[light][i].minDur
            surtracdata[light][i]["maxDur"] = lightphasedata[light][i].maxDur
            surtracdata[light][i]["lanes"] = []
            lightstate = lightphasedata[light][i].state
            
            linklistlist = lightlinks[light]
            for linklistind in range(len(linklistlist)):
                linkstate = lightstate[linklistind]

                if linkstate == "G" and not linklistlist[linklistind][0][0] in surtracdata[light][i]["lanes"]: #NOTE: I'm being sloppy and assuming one-element lists of tuples, but I've yet to see a multi-element list here
                    surtracdata[light][i]["lanes"].append(linklistlist[linklistind][0][0]) #[0][x]; x=0 is from, x=1 is to, x=2 is via
                
            #Remove lanes if there's any direction that gets a non-green light ("g" is fine, single-lane left turns are just sad)
            for linklistind in range(len(linklistlist)):
                linkstate = lightstate[linklistind]

                if not (linkstate == "G" or linkstate == "g") and linklistlist[linklistind][0][0] in surtracdata[light][i]["lanes"]: #NOTE: I'm being sloppy and assuming one-element lists of tuples, but I've yet to see a multi-element list here
                    surtracdata[light][i]["lanes"].remove(linklistlist[linklistind][0][0])
                

            #Compute min transition time between the start of any two phases
            surtracdata[light][i]["timeTo"] = [0]*n
            for joffset in range(1, n):
                j = (i + joffset) % n
                jprev = (j-1) % n
                surtracdata[light][i]["timeTo"][j] = surtracdata[light][i]["timeTo"][jprev] + lightphasedata[light][jprev].minDur

    if pSmart < 1:
        with open("Lturndata_"+routefile.split(".")[0]+".pickle", 'rb') as handle:
                turndata = pickle.load(handle)

    for edge in edges:
        if not edge[0] == ":":
            lanenums[edge] = traci.edge.getLaneNumber(edge)
            speeds[edge] = network.getEdge(edge).getSpeed()

    for lane in traci.lane.getIDList():
        links[lane] = traci.lane.getLinks(lane)
        lengths[lane] = traci.lane.getLength(lane)

    [avgTime, avgTimeSmart, avgTimeNot] = run(network, rerouters, pSmart, verbose)
    traci.close()
    return [avgTime, avgTimeSmart, avgTimeNot]


# this is the main entry point of this script
if __name__ == "__main__":
    if len(sys.argv) >= 3:
        pSmart = float(sys.argv[2])
    main(sys.argv[1], pSmart)
