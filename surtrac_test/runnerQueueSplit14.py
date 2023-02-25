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
#New in QueueSplit9: Take advantage of known routes from vehicle routing (NOTE: route may change as vehicle approaches intersection...)
#New in QueueSplit10: Optimize for speed (calling Surtrac less often in routing, merging predicted clusters)
#New in QueueSplit11: Compute delay, not just average travel time, for cars. Also fixed a bug with Surtrac code DP (was removing sequences it shouldn't have)
#QueueSplit12: Multithread the Surtrac code (it's really slow otherwise). Also, use the full Surtrac schedule rather than assuming we'll update every timestep
#QueueSplit13: Surtrac now (correctly) no longer overwrites all the finish times of other lanes with the start time of the currently scheduled cluster (leads to problems when a long cluster, then compatible short cluster, get scheduled, as the next cluster can then start earlier than it should). VOI now gets split into all lanes on starting edge
#14: Anytime routing, better stats on timeouts and teleports, added mingap to all cluster durations, using a timestep that divides mingap and surtracFreq, opposing traffic blocks for mingap not just one timestep

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

clusterthresh = 3 #Time between cars before we split to separate clusters
mingap = 2.5 #Minimum allowed space between cars
timestep = 0.5#mingap#1#mingap #Amount of time between updates. In practice, mingap rounds up to the nearest multiple of this
detectordist = 50
ntimeouts = 0

#Test durations to see if there's drift
simdurations = dict()
simdurationsUsed = False
realdurations = dict()

#Toggles for multithreading
multithreadRouting = False
multithreadSurtrac = True

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
fftimes = dict() #Free flow times for each edge/lane (dict contains both)
links = dict()
lengths = dict()
turndata = []
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

#Threading routing
toReroute = []
reroutedata = dict()
threads = dict()

nRoutingCalls = 0
routingTime = 0

#Quick test for UESO, TODO delete
nRight = []

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

#@profile
def doSurtracThread(network, simtime, light, clusters, lightphases, lastswitchtimes, inQueueSim, predictionCutoff, toSwitch, catpreds, remainingDuration, bestschedules):
    sult = 3 #Startup loss time

    #Figure out what an initial and complete schedule look like
    nPhases = len(surtracdata[light]) #Number of phases
    bestschedules[light] = [[]] #In case we terminate early or something??

    emptyStatus = dict()
    fullStatus = dict()
    nClusters = 0
    for lane in lightlanes[light]:
        emptyStatus[lane] = 0
        fullStatus[lane] = len(clusters[lane])
        nClusters += fullStatus[lane]

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
    #8: predicted outflows (as clusters - arrival, departure, list of cars, weights, etc.)

    emptyPreds = dict()
    for lane in lightoutlanes[light]:
        emptyPreds[lane] = []

    phase = lightphases[light]
    lastSwitch = lastswitchtimes[light]
    schedules = [([], emptyStatus, phase, [simtime]*len(surtracdata[light][phase]["lanes"]), simtime, 0, lastSwitch, [simtime - lastSwitch], emptyPreds)]


    for _ in range(nClusters): #Keep adding a cluster until #clusters added = #clusters to be added
        scheduleHashDict = dict()
        for schedule in schedules:
            assert(len(lightlanes[light]) > 0)
            for lane in lightlanes[light]:
                if schedule[1][lane] == fullStatus[lane]:
                    continue
                #Consider adding next cluster from surtracdata[light][i]["lanes"][j] to schedule
                newScheduleStatus = copy(schedule[1]) #Shallow copy okay? Dict points to int, which is stored by value #pickle.loads(pickle.dumps(schedule[1])) #deepcopy(schedule[1])
                newScheduleStatus[lane] += 1
                phase = schedule[2]

                #Now loop over all phases where we can clear this cluster
                try:
                    assert(len(lanephases[lane]) > 0)
                except:
                    print(lane)
                    print("ERROR: Can't clear this lane ever?")
                    
                for i in lanephases[lane]:
                    directionalMakespans = copy(schedule[3])

                    nLanes = len(surtracdata[light][i]["lanes"])
                    j = surtracdata[light][i]["lanes"].index(lane)

                    newDurations = copy(schedule[7]) #Should be fine #(pickle.dumps(schedule[7])) #deepcopy(schedule[7])

                    clusterind = newScheduleStatus[lane]-1 #We're scheduling the Xth cluster; it has index X-1
                    ist = clusters[lane][clusterind]["arrival"] #Intended start time = cluster arrival time
                    dur = clusters[lane][clusterind]["departure"] - ist + mingap #+mingap because next cluster can't start until mingap after current cluster finishes
                    #mindur = dur #TODO: Actually compute it
                    mindur = max((clusters[lane][clusterind]["weight"] - 1)*mingap, 0) #-1 because fencepost problem
                    mindur = max((clusters[lane][clusterind]["weight"] )*mingap, 0) #No -1 because fencepost problem; next cluster still needs 2.5s of gap afterwards
                    #Slight improvement when I use correct mindur for computing delay. Big unimprovement when I use it for directionalMakespans. Going to leave newdur=dur for now
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
                            assert(mindur >= 0)
                            assert(dur >= 0)
                            if mindur > 0 and dur > 0: #Having issues with negative weights, possibly related to cars contributing less than 1 to weight having left the edge
                                delay += tSent/dur*clusters[surtracdata[light][i]["lanes"][j]][clusterind]["weight"]*((ast-ist)-1/2*(dur-newdur) )
                                mindur *= 1-tSent/dur #Assuming uniform density
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
                    #Might want to tell other clusters to also start no sooner than max(new ast, old directionalMakespan value), but the DP should clear redundancies due to functionally equivalent orderings anyway.
                    #That max would be important, though; blind overwriting is wrong, as you could send a long cluster, then a short one, then change the light before the long one finishes
                    assert(len(directionalMakespans) == len(surtracdata[light][i]["lanes"]))
                    directionalMakespans[j] = ast+newdur+mingap

                    for k in range(len(directionalMakespans)):
                        if directionalMakespans[k] < ast:
                            directionalMakespans[k] = ast #Make sure we preserve cluster order, but don't overwrite stricter constraints such as from previously scheduling a long cluster in another lane
                    delay += clusters[surtracdata[light][i]["lanes"][j]][clusterind]["weight"]*((ast-ist)-1/2*(dur-newdur) ) #Delay += #cars * (actual-desired). 1/2(dur-newdur) compensates for the cluster packing together as it waits (I assume uniform compression)

                    try:
                        assert(delay >= schedule[5]) #Make sure delay doesn't go negative somehow
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

                    #TODO: This might not merge cars coming from different lanes in chronological order. Is this a problem?
                    #TODO: Are we handling splitty cars correctly?
                    newPredClusters = pickle.loads(pickle.dumps(schedule[8])) #Deep copy needed if I'm going to merge clusters #copy(schedule[8]) #Shallow copy is okay? This is a dict that points to lists of clusters, and I'm hopefully only ever changing the list of clusters (and the newly created clusters), never the old clusters. #pickle.loads(pickle.dumps(schedule[8]))
                    predLanes = []
                    for outlane in turndata[lane]: #lightoutlanes[light]: #Can't just be turndata[lane] since that might not have data for everything
                        arr = ast + fftimes[outlane]
                        assert(arr >= simtime)
                        if arr > simtime + predictionCutoff:
                            #Cluster is farther in the future than we want to predict; skip it
                            continue
                        newPredCluster = dict()
                        newPredCluster["endpos"] = 0
                        newPredCluster["time"] = ast
                        newPredCluster["arrival"] = arr
                        newPredCluster["departure"] = newPredCluster["arrival"] + newdur
                        newPredCluster["cars"] = []
                        newPredCluster["weight"] = 0
                        if not outlane in newPredClusters:
                            newPredClusters[outlane] = []
                        newPredClusters[outlane].append(newPredCluster)
                        predLanes.append(outlane) #Track which lanes' clusters are within the prediction cutoff

                    #Add cars to new clusters
                    edge = lane.split("_")[0]
                    for cartuple in clusters[lane][clusterind]["cars"]:
                        #cartuple[0] is name of car; cartuple[1] is departure time; cartuple[2] is debug info
                        if not cartuple[0] in isSmart or isSmart[cartuple[0]]: #It's possible we call this from QueueSim, at which point we split the vehicle being routed and wouldn't recognize the new names. Anything else should get assigned to isSmart or not on creation
                            #Split on "|" and "_" to deal with splitty cars correctly
                            route = currentRoutes[cartuple[0].split("|")[0].split("_")[0]] #traci.vehicle.getRoute(cartuple[0].split("|")[0]) #.split to deal with the possibility of splitty cars in QueueSim
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
                            
                            #Picking a random lane on the appropriate edge based on turndata
                            #Should probably do something cleverer based on the rest of the route, but hopefully this is fine
                            normprob = 0
                            for nextlaneind in range(lanenums[nextedge]):
                                nextlane = nextedge+"_"+str(nextlaneind) #Might not be a valid lane to transition to...
                                if lane in turndata and nextlane in turndata[lane]: #NOT predLanes; it's possible the car takes a path we don't care to predict, and we don't want to normalize that out
                                    normprob += turndata[lane][nextedge+"_"+str(nextlaneind)]
                            if normprob == 0:
                                #Might be happening if the car needs to make a last-minute lane change to stay on its route?
                                #TODO: Find a lane where it can continue with the route and go from there? Ignoring for now
                                #print("Warning, no data, having Surtrac prediction ignore this car instead of making something up")
                                #print(lane)
                                continue
                            for nextlaneind in range(lanenums[nextedge]):
                                nextlane = nextedge+"_"+str(nextlaneind)
                                if nextlane in predLanes: #Make sure we're predicting this cluster
                                    modcartuple = (cartuple[0], cartuple[1]+fftimes[nextlane], cartuple[2]*turndata[lane][nextlane] / normprob, cartuple[3])
                                    newPredClusters[nextlane][-1]["cars"].append(modcartuple)
                                    newPredClusters[nextlane][-1]["weight"] += modcartuple[2]
                        else:
                            for nextlane in predLanes:
                                modcartuple = (cartuple[0], cartuple[1]+fftimes[nextlane], cartuple[2]*turndata[lane][nextlane], cartuple[3])
                                newPredClusters[nextlane][-1]["cars"].append(modcartuple)
                                newPredClusters[nextlane][-1]["weight"] += modcartuple[2]
                    for outlane in predLanes:
                        if newPredClusters[outlane][-1]["weight"] == 0:
                            #Remove predicted clusters that are empty
                            newPredClusters[outlane].pop(-1)
                            continue

                        if len(newPredClusters[outlane]) >=2 and newPredClusters[outlane][-1]["arrival"] - newPredClusters[outlane][-2]["departure"] < clusterthresh:                            
                            #Merge this cluster with the previous one
                            #Pos and time don't do anything here
                            #Arrival doesn't change - previous cluster arrived first
                            newPredClusters[outlane][-2]["departure"] = max(newPredClusters[outlane][-2]["departure"], newPredClusters[outlane][-1]["departure"])
                            newPredClusters[outlane][-2]["cars"] += newPredClusters[outlane][-1]["cars"] # += concatenates
                            newPredClusters[outlane][-2]["weight"] += newPredClusters[outlane][-1]["weight"]
                            newPredClusters[outlane].pop(-1)

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
                    #8: predicted outflows (as clusters - arrival, departure, list of cars, weights, etc.)

                    newschedule = (schedule[0]+[(i, j)], newScheduleStatus, i, directionalMakespans, newMakespan, delay, newLastSwitch, newDurations, newPredClusters)
                    
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
                            assert(newschedule[1] == testschedule[1])
                            assert(newschedule[2] == testschedule[2])
                            
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
                    assert(len(scheduleHashDict[key]) > 0)

        schedules = sum(list(scheduleHashDict.values()), []) #Each key has a list of non-dominated partial schedules. list() turns the dict_values object into a list of those lists; sum() concatenates to one big list of partial schedules. (Each partial schedule is stored as a tuple)

    mindelay = np.inf
    bestschedule = [[]]
    for schedule in schedules:
        if schedule[5] < mindelay:
            mindelay = schedule[5]
            bestschedule = schedule
    
    if not bestschedule == [[]]:
        catpreds.update(bestschedule[8])
        bestschedules[light] = bestschedule
    else:
        print(light)
        print("No schedules anywhere? That shouldn't happen...")

#@profile
def doSurtrac(network, simtime, realclusters=None, lightphases=None, lastswitchtimes=None, predClusters=None):
    #print("Starting Surtrac")
    #print(simtime)

    toSwitch = []
    catpreds = dict()
    remainingDuration = dict()
    bestschedules = dict()

    surtracThreads = dict()

    inQueueSim = True
    if realclusters == None and lightphases == None:
        inQueueSim = False
        (realclusters, lightphases) = loadClusters(network)

    #predCutoff
    if inQueueSim:
        predictionCutoff = 0 #Routing
    else:
        predictionCutoff = 0 #Main simulation
    

    if not predClusters == None:
        clusters = mergePredictions(realclusters, predClusters)
    else:
        clusters = pickle.loads(pickle.dumps(realclusters))

    for light in lights:
        if multithreadSurtrac:
            surtracThreads[light] = threading.Thread(target=doSurtracThread, args=(network, simtime, light, clusters, lightphases, lastswitchtimes, inQueueSim, predictionCutoff, toSwitch, catpreds, remainingDuration, bestschedules))
            surtracThreads[light].start()
        else:
            doSurtracThread(network, simtime, light, clusters, lightphases, lastswitchtimes, inQueueSim, predictionCutoff, toSwitch, catpreds, remainingDuration, bestschedules)

    for light in lights:
        if multithreadSurtrac:
            surtracThreads[light].join()
    
        bestschedule = bestschedules[light]
        if not bestschedule[0] == []:
            spentDuration = simtime - lastswitchtimes[light]
            remainingDuration[light] = bestschedule[7]
            if len(remainingDuration[light]) > 0:
                remainingDuration[light][0] -= spentDuration

                if remainingDuration[light][0] >= 0 and not inQueueSim:
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
                    if not (simtime - lastswitchtimes[light] >= surtracdata[light][curphase]["minDur"] and simtime - lastswitchtimes[light] <= surtracdata[light][curphase]["maxDur"]+timestep):
                        print("Duration violation on light " + light + "; actual duration " + str(simtime - lastswitchtimes[light]))

                    lightphases[light] = (curphase+1)%nPhases #This would change the light if we're not in QueueSim
                    lastswitchtimes[light] = simtime

                    if len(remainingDuration[light]) == 0:
                        remainingDuration[light] = [lightphasedata[light][(lightphases[light]+1)%len(lightphasedata[light])].duration]

                    remainingDuration[light].pop(0)

                    if not inQueueSim: #Actually change the light
                        traci.trafficlight.setPhase(light, (curphase+1)%nPhases) #Increment phase, duration defaults to default
                        if len(remainingDuration[light]) > 0:
                            #And set the new duration if possible
                            traci.trafficlight.setPhaseDuration(light, remainingDuration[light][0]) #Update duration if we know it
                            #pass


    #Predict-ahead for everything else; assume no delays.
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
                        route = currentRoutes[cartuple[0].split("|")[0].split("_")[0]] #traci.vehicle.getRoute(cartuple[0].split("|")[0]) #.split to deal with the possibility of splitty cars in QueueSim
                        if not edge in route:
                            #Not sure if or why this happens - maybe the route is changing and predictions aren't updating?
                            #Can definitely happen for a splitty car inside QueueSim
                            #Regardless, don't predict this car forward and hope for the best?
                            if not "|" in cartuple[0]:
                                pass
                                #print("Warning, smart car on an edge that's not in its route. Assuming a mispredict and removing")
                            #TODO: Else should predict it goes everywhere? Does this happen??
                            continue
                        edgeind = route.index(edge)
                        if edgeind+1 == len(route):
                            #At end of route, don't care
                            continue
                        nextedge = route[edgeind+1]
                        
                        #Picking a random lane on the appropriate edge based on turndata
                        #Should probably do something cleverer based on the rest of the route, but hopefully this is fine
                        #TODO: Pick a lane according to where the car wants to go next next instead
                        normprob = 0
                        for nextlaneind in range(lanenums[nextedge]):
                            nextlane = nextedge+"_"+str(nextlaneind) #Might not be a valid lane to transition to...
                            if lane in turndata and nextlane in turndata[lane]: #NOT predLanes; it's possible the car takes a path we don't care to predict, and we don't want to normalize that out
                                normprob += turndata[lane][nextedge+"_"+str(nextlaneind)]
                        if normprob == 0:
                            #Might be happening if the car needs to make a last-minute lane change to stay on its route?
                            #TODO: Find a lane where it can continue with the route and go from there? Ignoring for now
                            #print("Warning, no data, having Surtrac prediction ignore this car instead of making something up")
                            #print(lane)
                            continue
                        for nextlaneind in range(lanenums[nextedge]):
                            nextlane = nextedge+"_"+str(nextlaneind)
                            modcartuple = (cartuple[0], cartuple[1]+fftimes[nextlane], cartuple[2]*turndata[lane][nextlane] / normprob, cartuple[3])
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
    goalcost = traci.lane.getLength(goal+"_0")/network.getEdge(goal).getSpeed()
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
            c = traci.lane.getLength(edge+"_0")/network.getEdge(edge).getSpeed()

            h = 0 #Heuristic not needed here - search is fast
            if succ in gvals and gvals[succ] <= gval+c:
                #Already saw this state, don't requeue
                continue

            #Otherwise it's new or we're now doing better, so requeue it
            gvals[succ] = gval+c
            heappush(pq, (gval+c+h, succ))
    return gvals
                    
##@profile
def run(network, rerouters, pSmart, verbose = True):
    global sumoPredClusters
    global currentRoutes
    global hmetadict
    global delay3adjdict
    global actualStartDict
    global laneDict
    #netfile is the filepath to the network file, so we can call sumolib to get successors
    #rerouters is the list of induction loops on edges with multiple successor edges
    #We want to reroute all the cars that passed some induction loop in rerouters using A*
    
    """execute the TraCI control loop"""
    
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
    while traci.simulation.getMinExpectedNumber() > 0:
        simtime += 1
        traci.simulationStep() #Tell the simulator to simulate the next time step

        #Check for lights that switched phase; update custom data structures and current phase duration
        for light in lights:
            temp = traci.trafficlight.getPhase(light)
            if not(light in remainingDuration and len(remainingDuration[light]) > 0):
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

        #Decide whether new vehicles use our routing
        for vehicle in traci.simulation.getDepartedIDList():
            isSmart[vehicle] = random.random() < pSmart
            timedata[vehicle] = [simtime, -1, -1]
            currentRoutes[vehicle] = traci.vehicle.getRoute(vehicle)
            routeStats[vehicle] = dict()
            routeStats[vehicle]["nCalls"] = 0
            routeStats[vehicle]["nSwaps"] = 0
            routeStats[vehicle]["swapped"] = False
            routeStats[vehicle]["nTimeouts"] = 0
            routeStats[vehicle]["nTeleports"] = 0

            goaledge = currentRoutes[vehicle][-1]
            if not goaledge in hmetadict:
                hmetadict[goaledge] = backwardDijkstra(network, goaledge)
            delayDict[vehicle] = -hmetadict[goaledge][currentRoutes[vehicle][0]] #I'll add the actual travel time once the vehicle arrives
            laneDict[vehicle] = traci.vehicle.getLaneID(vehicle)
            currentRoutes[vehicle] = traci.vehicle.getRoute(vehicle)

        for vehicle in traci.simulation.getArrivedIDList():
            timedata[vehicle][1] = simtime
            #print("Actual minus expected:")
            #print( (timedata[vehicle][1]-timedata[vehicle][0]) - timedata[vehicle][2])

        surtracFreq = 1 #Period between updates in main SUMO sim
        if simtime%surtracFreq >= (simtime+1)%surtracFreq:
            temp = doSurtrac(network, simtime, None, None, mainlastswitchtimes, sumoPredClusters)
            #Don't store toUpdate = temp[0], since doSurtrac has done that update already
            sumoPredClusters = temp[1]
            remainingDuration.update(temp[2])

        reroute(rerouters, network, simtime, remainingDuration) #Reroute cars (including simulate-ahead cars)
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
            if traci.vehicle.getRoadID(id) != locDict[id] and len(traci.vehicle.getRoadID(id)) > 0 and  traci.vehicle.getRoadID(id)[0] != ":":
                c0 = network.getEdge(locDict[id]).getFromNode().getCoord()
                c1 = network.getEdge(locDict[id]).getToNode().getCoord()
                theta0 = math.atan2(c1[1]-c0[1], c1[0]-c0[0])

                c2 = network.getEdge(traci.vehicle.getRoadID(id)).getToNode().getCoord()
                theta1 = math.atan2(c2[1]-c1[1], c2[0]-c1[0])

                if (theta1-theta0+math.pi)%(2*math.pi)-math.pi > 0:
                    leftDict[id] += 1
                laneDict[id] = traci.vehicle.getLaneID(id)
                locDict[id] = traci.vehicle.getRoadID(id)

                #Remove vehicle from predictions, since the next intersection should actually see it now
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

        
        for car in traci.simulation.getStartingTeleportIDList():
            routeStats[car]["nTeleports"] += 1
            print("Warning: Car " + car + " teleported, time=" + str(simtime))

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
            totalswaps = 0
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
                totalswaps += routeStats[id]["nSwaps"]
                if routeStats[id]["swapped"] == True:
                    nswapped += 1

        
            if verbose or not traci.simulation.getMinExpectedNumber() > 0:
                print(pSmart)
                print("\nCurrent simulation time: %f" % simtime)
                print("Total run time: %f" % (time.time() - tstart))
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
                    print("Proportion of cars that changed route at least once: %f" % (nswapped/nCars))
                    print("Average number of teleports: %f" % (nteleports/nCars))
                print("Among adopters:")
                print("Average delay: %f" % avgTimeSmart)
                print("Best delay: %f" % bestTimeSmart)
                print("Worst delay: %f" % worstTimeSmart)
                print("Average number of lefts: %f" % avgLeftsSmart)
                if nSmart > 0:
                    print("Average number of calls to routing: %f" % (totalcalls/nSmart))
                    if totalcalls > 0:
                        print("Proportion of timeouts in routing: %f" % (ntimeouts/totalcalls))
                    print("Average number of route changes: %f" % (totalswaps/nSmart))
                    print("Proportion of cars that changed route at least once: %f" % (nswapped/nSmart))
                    print("Average number of teleports: %f" % (nsmartteleports/nSmart))
                print("Among non-adopters:")
                print("Average delay: %f" % avgTimeNot)
                print("Best delay: %f" % bestTimeNot)
                print("Worst delay: %f" % worstTimeNot)
                print("Average number of lefts: %f" % avgLeftsNot)
                if nCars - nSmart > 0:
                    print("Average number of teleports: %f" % (nnotsmartteleports/(nCars-nSmart)))
                #print(len(nRight)/1200)
    return [avgTime, avgTimeSmart, avgTimeNot, avgTime2, avgTimeSmart2, avgTimeNot2, avgTime3, avgTimeSmart3, avgTimeNot3, avgTime0, avgTimeSmart0, avgTimeNot0]

    

#Tell all the detectors to reroute the cars they've seen
##@profile
def reroute(rerouters, network, simtime, remainingDuration):
    global toReroute
    global threads

    toReroute = []
    reroutedata = dict()
    threads = dict()
    
    #Test code from UESO, TODO delete
    # global nRight
    # detector = "IL_R1_0"
    # ids = traci.inductionloop.getLastStepVehicleIDs(detector) #All vehicles to be rerouted
    # if detector == "IL_R1_0":
    #     for id in ids:
    #         if not id in nRight:
    #             nRight.append(id)

    for r in rerouters:
        QueueReroute(r, network, reroutedata, simtime, remainingDuration)

    for vehicle in toReroute:
        if multithreadRouting:
            threads[vehicle].join()
        data = reroutedata[vehicle]
        
        newroute = data[0]

        routeStats[vehicle]["nCalls"] += 1

        if not tuple(newroute) == currentRoutes[vehicle] and not newroute == currentRoutes[vehicle][-len(newroute):]:
            #print(newroute)
            #print(currentRoutes[vehicle])
            routeStats[vehicle]["nSwaps"] += 1
            routeStats[vehicle]["swapped"] = True
        else:
            #print("NO CHANGE")
            #print(currentRoutes[vehicle])
            pass

        tcluster = data[1]
        #print(routes[vehicle])
        #print(edge)
        #print(tcluster)
        if timedata[vehicle][2] == -1:
            timedata[vehicle][2] = tcluster
            
        try:
            traci.vehicle.setRoute(vehicle, newroute)
            if not(newroute[-1] == currentRoutes[vehicle][-1]):
                print("AAAAAAAAAAAAAAAAAAAH! Route end changed!")
                print(newroute[-1])
                print(currentRoutes[vehicle][-1])
            assert(newroute[-1] == currentRoutes[vehicle][-1]) #This doesn't end code because try-catch. Is there a good way to fix this?
            currentRoutes[vehicle] = newroute
        except Exception as e:
            print("Routing fail fail")
            print(e)
            pass
            

##@profile
def QueueReroute(detector, network, reroutedata, simtime, remainingDuration):
    global toReroute
    global threads
    global delay3adjdict

    ids = traci.inductionloop.getLastStepVehicleIDs(detector) #All vehicles to be rerouted
    if len(ids) == 0:
        #No cars to route, we're done here
        return

    lane = traci.inductionloop.getLaneID(detector)
    #edge = lane.split[0]

    for vehicle in ids:
        # if detector == "IL_Ross-Seventh-Sixth_0" or detector == "IL_Ross-Seventh-Sixth_1":
        #     print("Yay!")
        try:
            if traci.vehicle.getLaneID(vehicle) != lane:
                #Vehicle isn't on same lane as detector. Stuff is going wrong, skip this.
                continue
        except: #Vehicle off network already??
            continue

        if detector in oldids and vehicle in oldids[detector]:
            #Just routed this, don't try again
            continue

        if vehicle not in delay3adjdict:
            delay3adjdict[vehicle] = simtime

        if isSmart[vehicle]:
            #Convert current state

            #tstart = time.time()
            toReroute.append(vehicle)
            reroutedata[vehicle] = [None]*2
            loaddata = loadClusters(network, vehicle)
            vehicles = traci.vehicle.getIDList()


            #Store routes once at the start to save time
            routes = deepcopy(currentRoutes)

            for vehicletemp in vehicles:
                if isSmart[vehicletemp]:
                    pass
                    #routes[vehicle] = traci.vehicle.getRoute(vehicle)
                else:
                    #Sample random routes for non-adopters
                    routes[vehicletemp] = sampleRouteFromTurnData(vehicletemp, traci.vehicle.getLaneID(vehicletemp), turndata)

            if multithreadRouting:
                threads[vehicle] = threading.Thread(target=doClusterSimThreaded, args=(lane, network, vehicle, simtime, remainingDuration, reroutedata[vehicle], deepcopy(loaddata), routes))
                threads[vehicle].start()
            else:
                doClusterSimThreaded(lane, network, vehicle, simtime, remainingDuration, reroutedata[vehicle], deepcopy(loaddata), routes) #If we want non-threaded
        
                #NEW: Copy-pasting route update data here to see if this improves stuff at 99% adoption

                #Check for route change before we update currentRoutes (other analysis can still happen in outer function)
                newroute = reroutedata[vehicle][0]
                if not tuple(newroute) == currentRoutes[vehicle] and not newroute == currentRoutes[vehicle][-len(newroute):]:
                    #print(newroute)
                    #print(currentRoutes[vehicle])
                    routeStats[vehicle]["nSwaps"] += 1
                    routeStats[vehicle]["swapped"] = True
                else:
                    #print("NO CHANGE")
                    #print(currentRoutes[vehicle])
                    pass

                currentRoutes[vehicle] = newroute

    oldids[detector] = ids

def doClusterSimThreaded(prevlane, net, vehicle, simtime, remainingDuration, data, loaddata, routes):
    global nRoutingCalls
    global routingTime
    starttime = time.time()
    temp = runClusters(net, simtime, remainingDuration, vehicle, prevlane, loaddata, routes)
    nRoutingCalls += 1
    routingTime += time.time() - starttime
    for i in range(len(temp)):
        data[i] = temp[i]

def loadClusters(net, VOI=None):
    #Load locations of cars and current traffic light states into custom data structures
    #If given, VOI is the vehicle triggering the routing call that triggered this, and needs to be unaffected when we add noise
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
                if len(clusters[lane]) > 0 and abs(clusters[lane][-1]["time"] - traci.simulation.getTime()) < clusterthresh and abs(clusters[lane][-1]["endpos"] - traci.vehicle.getLanePosition(vehicle))/speeds[edge] < clusterthresh:
                    #Add to cluster. pos and time track newest added vehicle to see if the next vehicle merges
                    #Departure time (=time to fully clear cluster) increases, arrival doesn't
                    clusters[lane][-1]["endpos"] = traci.vehicle.getLanePosition(vehicle)
                    clusters[lane][-1]["time"] = traci.simulation.getTime()
                    clusters[lane][-1]["departure"] = traci.simulation.getTime() + (lengths[lane]-clusters[lane][-1]["endpos"])/speeds[edge]
                    clusters[lane][-1]["cars"].append((vehicle, clusters[lane][-1]["departure"], 1, "Load append"))
                    clusters[lane][-1]["weight"] = len(clusters[lane][-1]["cars"])
                else:
                    #Else make a new cluster
                    newcluster = dict()
                    newcluster["startpos"] = traci.vehicle.getLanePosition(vehicle)
                    newcluster["endpos"] = traci.vehicle.getLanePosition(vehicle)
                    newcluster["time"] = traci.simulation.getTime()
                    newcluster["arrival"] = traci.simulation.getTime() + (lengths[edge+"_0"]-newcluster["endpos"])/speeds[edge]
                    newcluster["departure"] = newcluster["arrival"]
                    newcluster["cars"] = [(vehicle, newcluster["departure"], 1, "Load new")]
                    newcluster["weight"] = len(newcluster["cars"])
                    clusters[lane].append(newcluster)
    
    #Traffic light info
    lightphases = dict()
    for light in lights:
        lightphases[light] = traci.trafficlight.getPhase(light)
    #oldclusters = pickle.loads(pickle.dumps(clusters))
    #clusters = addNoise(clusters, VOI, 0.9, 2) #To simulate detector stuff
    return (clusters, lightphases)

def addNoise(clusters, VOI, detectprob, timeerr):
    #Randomly delete cars with probability noiseprob
    #Randomly clone non-deleted cars to make up for it

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
                #cluster = noisycluster
                clusters[lane][clusternum] = noisycluster
    return clusters


#NOTE: Multithreaded stuff doesn't get profiled...
##@profile
def runClusters(net, routesimtime, mainRemainingDuration, vehicleOfInterest, startlane, loaddata, routes):
    #global ntimeouts
    startedge = startlane.split("_")[0]

    goalEdge = routes[vehicleOfInterest][-1]
    splitinfo = dict()
    VOIs = [vehicleOfInterest]

    clusters = loaddata[0]
    lightphases = loaddata[1]

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
        #print(clusters)
        print(startlane)
        raise Exception("Can't find vehicle of interest init!")
    #End sanity check

    lastDepartTime = dict()

    starttime = routesimtime
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

    #Split initial VOI into all starting lanes
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
        vehicle = vehicleOfInterest+"_"+str(lanenum) #Name of copy of VOI to add
        VOIs.append(vehicle)
        for cluster in clusters[lane]:
            assert(cluster["startpos"] >= cluster["endpos"]) #Startpos is position of first car to leave, which is closest to end of edge, but 0 is start of edge, so startpos should be larger
            if cluster["startpos"] >= startdist and cluster["endpos"] <= startdist:
                #Found an appropriate cluster; prepend to it
                assert(cluster["time"] == routesimtime)
                ffdeparttime = routesimtime + (lengths[lane]-clusters[lane][-1]["endpos"])/speeds[startedge]
                clusters[lane][-1]["cars"].append((vehicle, ffdeparttime, 1, "VOI append clone"))
                clusters[lane][-1]["weight"] += 1
                VOIadded = True
                break
        if not VOIadded:
            #Else make a new cluster
            newcluster = dict()
            newcluster["startpos"] = startdist
            newcluster["endpos"] = newcluster["startpos"]
            newcluster["time"] = routesimtime
            newcluster["arrival"] = newcluster["time"] + (lengths[lane]-newcluster["endpos"])/speeds[startedge]
            newcluster["departure"] = newcluster["arrival"]
            newcluster["cars"] = [(vehicle, newcluster["departure"], 1, "VOI new clone")]
            newcluster["weight"] = len(newcluster["cars"])
            clusters[lane].append(newcluster)


    queueSimPredClusters = pickle.loads(pickle.dumps(sumoPredClusters)) #Initial predicted clusters are whatever SUMO's Surtrac thinks it is
    queueSimLastSwitchTimes = pickle.loads(pickle.dumps(mainlastswitchtimes)) #Initial last switch times are whatever they were in the main simulation
    remainingDuration = pickle.loads(pickle.dumps(mainRemainingDuration)) #Copy any existing schedules from main sim
    surtracFreq = 1 #Time between Surtrac updates, in seconds, during routing. (Technically the period between updates)

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
    startedgeind = routes[vehicleOfInterest].index(startedge)
    bestroute = routes[vehicleOfInterest][startedgeind:]
    toupgrade = routes[vehicleOfInterest][startedgeind+1:]

    blockingLinks = dict()
    while True:

        #Timeout if things have gone wrong somehow
        if time.time()-routestartwctime > timeout:
            print("Routing timeout: Edge " + startedge + ", time: " + str(starttime))
            routeStats[vehicleOfInterest]["nTimeouts"] += 1
            
            return (bestroute, -1)

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
            #print(clusters)
            print(startlane)
            raise Exception("Can't find vehicle of interest!")
        #End sanity check

        routesimtime += timestep

        #Update lights
        if surtracFreq <= timestep or routesimtime%surtracFreq >= (routesimtime+timestep)%surtracFreq:
            (_, queueSimPredClusters, newRemainingDuration) = doSurtrac(net, routesimtime, clusters, lightphases, queueSimLastSwitchTimes, queueSimPredClusters)
            remainingDuration.update(newRemainingDuration)
            for light in newRemainingDuration:
                assert(newRemainingDuration[light][0] > 0)
            for light in remainingDuration:
                if len(remainingDuration[light]) == 0:
                    #This should never happen
                    print(remainingDuration)
                    print(newRemainingDuration)
                    print(light)
                    assert(len(remainingDuration[light]) > 0)
        
        #Keep track of what lights change when, since we're not running Surtrac every timestep
        for light in lights:
            if light not in remainingDuration or len(remainingDuration[light]) == 0:
                print("Empty remainingDuration for light " + light + " in runClusters, which shouldn't happen; using the default value")
                remainingDuration[light] = [lightphasedata[light][lightphases[light]].duration]
            #All lights should have a non-zero length schedule in remainingDuration
            remainingDuration[light][0] -= timestep
            #Next TODO: Should this actually be <= 0 not < 0? Ex: Durations of 1, 1, 1, ...; first one lasts 1 timestep as desired, later ones last 2. Think I messed up the fencepost problem here.
            #Think I fixed this, TODO test this with fixed timing plans, make sure I didn't break this
            if remainingDuration[light][0] <= 0: #Note: Duration might not be divisible by timestep, so we might be getting off by a little over multiple phases??
                tosubtract = remainingDuration[light][0]
                remainingDuration[light].pop(0)
                lightphases[light] = (lightphases[light]+1)%len(lightphasedata[light])
                queueSimLastSwitchTimes[light] = routesimtime #Next TODO: Does this cause divisibility issues?
                if len(remainingDuration[light]) == 0:
                    remainingDuration[light] = [lightphasedata[light][lightphases[light]].duration + tosubtract] #tosubtract is negative
                #Main sim doesn't subtract for overruns if there's a Surtrac schedule, so we won't do that here either
        
        #Test code to make sure light schedules don't drift for large surtracFreq
        if not routesimtime in simdurations:
            simdurations[routesimtime] = pickle.loads(pickle.dumps(remainingDuration))
            if newsim == True:
                simdurations[routesimtime][lights[0]] = str(simdurations[routesimtime][lights[0]]) + " NEW SIM"
            newsim = False

        #blockingLinks = dict() #Moved to OUTSIDE the while loop so blockages persist between timesteps!!! TODO delete
        reflist = pickle.loads(pickle.dumps(edgelist)) #deepcopy(edgelist) #Want to reorder edge list to handle priority stuff, but don't want to mess up the for loop indexing
        for edge in reflist:

            for lanenum in range(lanenums[edge]):
                lane = edge + "_" + str(lanenum)

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
                            # print("upgrade test")
                            # print(edge)
                            # print(toupgrade)
                            # print(toupgrade.index(edge))
                            toupgrade = toupgrade[toupgrade.index(edge)+1:]
                            # print(toupgrade)
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

                        tempnextedges = pickle.loads(pickle.dumps(splitinfo[(cartuple[0], edge)])) #deepcopy(splitinfo[(cartuple[0], edge)])

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
                                            #NEXT TODO: Opposing traffic blocks links for mingap amount of time, not just one timestep
                                            #I think I have this working now
                                            for linktuple2 in prioritygreenlightlinks[node][lightphases[node]]+lowprioritygreenlightlinks[node][lightphases[node]]:
                                                conflicting = lightlinkconflicts[node][linktuple][linktuple2] #Precomputed to save time 

                                                if conflicting and (linktuple2 in blockingLinks[node] and blockingLinks[node][linktuple2] > routesimtime - mingap):
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
                            carlength = 7.5 #meters
                            for nlc in clusters[nextlane]:
                                totalcarnum += nlc["weight"]
                            if totalcarnum >= (lengths[nextlane]-10) / carlength:
                                continue

                            #Check append to previous cluster vs. add new cluster
                            if len(clusters[nextlane]) > 0 and abs(clusters[nextlane][-1]["time"] - routesimtime) < clusterthresh and abs(clusters[nextlane][-1]["endpos"])/speeds[nextedge] < clusterthresh:
                                
                                #Make sure there's no car on the new road that's too close
                                if not abs(clusters[nextlane][-1]["time"] - routesimtime) < mingap:
                                    #Add to cluster. pos and time track newest added vehicle to see if the next vehicle merges
                                    #Departure time (=time to fully clear cluster) increases, arrival doesn't
                                    #TODO eventually: Be more precise with time and position over partial timesteps, allowing me to use larger timesteps?
                                    clusters[nextlane][-1]["endpos"] = 0
                                    clusters[nextlane][-1]["time"] = routesimtime
                                    clusters[nextlane][-1]["departure"] = routesimtime + fftimes[nextedge]
                                    if cartuple[0] in VOIs:
                                        clusters[nextlane][-1]["cars"].append((cartuple[0]+"|"+nextlane, clusters[nextlane][-1]["departure"], 1, "Zipper append"))
                                        VOIs.append(cartuple[0]+"|"+nextlane)
                                    else:
                                        clusters[nextlane][-1]["cars"].append((cartuple[0], clusters[nextlane][-1]["departure"], 1, "Zipper append"))
                                    clusters[nextlane][-1]["weight"] += 1
                                else:
                                    #No space, try next lane
                                    continue
                            else:
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
                                else:
                                    newcluster["cars"] = [(cartuple[0], newcluster["departure"], 1, "Zipper new cluster")]
                                newcluster["weight"] = 1
                                clusters[nextlane].append(newcluster)

                            #We've added a car to nextedge_nextlanenum
                            
                            #Remove vehicle from predictions, since the next intersection should actually see it now
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
                            edgelist.append(edgelist.pop(edgelist.index(edge))) #Push edge to end of list to give it lower priority next time
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


def main(sumoconfig, pSmart, verbose = True):
    global lowprioritygreenlightlinks
    global prioritygreenlightlinks
    global edges
    global turndata
    global actualStartDict
    options = get_options()

    rngstate = random.getstate()
    with open("lastRNGstate.pickle", 'wb') as handle:
        pickle.dump(rngstate, handle, protocol=pickle.HIGHEST_PROTOCOL)

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

    #lights = traci.trafficlight.getIDList()
    edges = traci.edge.getIDList()
    #Grab stuff once at the start to avoid slow calls to traci in the routing

    lowprioritygreenlightlinks = dict()
    prioritygreenlightlinks = dict()

    for light in lights:
        lightlinkconflicts[light] = dict()
        lightphasedata[light] = traci.trafficlight.getCompleteRedYellowGreenDefinition(light)[0].phases
        lightlinks[light] = traci.trafficlight.getControlledLinks(light)
        lightphases[light] = traci.trafficlight.getPhase(light)

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
                surtracdata[light][i]["minDur"] = 5#1#3.5#5#lightphasedata[light][i].minDur
            # if "Y" in lightphasedata[light][i].state or "y" in lightphasedata[light][i].state:
            #     surtracdata[light][i]["minDur"] = 2#5#lightphasedata[light][i].minDur

            surtracdata[light][i]["lanes"] = []
            lightstate = lightphasedata[light][i].state
            lowprioritygreenlightlinks[light].append([])
            prioritygreenlightlinks[light].append([])
            lowprioritygreenlightlinksLE[light].append(dict())
            prioritygreenlightlinksLE[light].append(dict())
            
            linklistlist = lightlinks[light]
            for linklistind in range(len(linklistlist)):
                linkstate = lightstate[linklistind]
                if not linklistlist[linklistind][0][0] in lanephases:
                    lanephases[linklistlist[linklistind][0][0]] = []

                if linkstate == "G" and not linklistlist[linklistind][0][0] in surtracdata[light][i]["lanes"]: #NOTE: I'm being sloppy and assuming one-element lists of tuples, but I've yet to see a multi-element list here
                    surtracdata[light][i]["lanes"].append(linklistlist[linklistind][0][0]) #[0][x]; x=0 is from, x=1 is to, x=2 is via
                    
                    lanephases[linklistlist[linklistind][0][0]].append(i)

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

                if not (linkstate == "G" or linkstate == "g") and linklistlist[linklistind][0][0] in surtracdata[light][i]["lanes"]: #NOTE: I'm being sloppy and assuming one-element lists of tuples, but I've yet to see a multi-element list here
                    surtracdata[light][i]["lanes"].remove(linklistlist[linklistind][0][0])
                    lanephases[linklistlist[linklistind][0][0]].remove(i)
                
        for i in range(n):
            #Compute min transition time between the start of any two phases
            surtracdata[light][i]["timeTo"] = [0]*n
            for joffset in range(1, n):
                j = (i + joffset) % n
                jprev = (j-1) % n
                surtracdata[light][i]["timeTo"][j] = surtracdata[light][i]["timeTo"][jprev] + surtracdata[light][jprev]["minDur"]

    if pSmart < 1 or True:
        with open("Lturndata_"+routefile.split(".")[0]+".pickle", 'rb') as handle:
                turndata = pickle.load(handle)

    for lane in traci.lane.getIDList():
        if not lane[0] == ":":
            links[lane] = traci.lane.getLinks(lane)
            lengths[lane] = traci.lane.getLength(lane)

    for edge in edges:
        if not edge[0] == ":":
            lanenums[edge] = traci.edge.getLaneNumber(edge)
            speeds[edge] = network.getEdge(edge).getSpeed()
            fftimes[edge] = lengths[edge+"_0"]/speeds[edge]

    for lane in traci.lane.getIDList():
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

    outdata = run(network, rerouters, pSmart, verbose)
    traci.close()

    print("Routing calls: " + str(nRoutingCalls))
    print("Total routing time: " + str(routingTime))
    print("Average time per call: " + str(routingTime/nRoutingCalls))
    return [outdata, rngstate]


# this is the main entry point of this script
if __name__ == "__main__":
    if len(sys.argv) >= 3:
        pSmart = float(sys.argv[2])
    main(sys.argv[1], pSmart)
