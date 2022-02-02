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


#Changes from QueueSplit2: Somewhat respect lane-to-lane connections
#In QueueSplit2, given connection laneA -> laneB, I let any lane on laneA's edge go to any lane on laneB's edge
#I still don't want to deal with lane changes, so if laneA goes to laneB, I'll allow cars to go from laneA
#to any lane on the same edge as laneB (pretend you lane-change in the intersection or something?)
#Splitty cars split to all lanes
#Non-splitty cars go to first open lane that can go to next next edge

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

isSmart = dict(); #Store whether each vehicle does our routing or not
pSmart = 1.0; #Adoption probability

carsOnNetwork = []
max_edge_speed = 0.0

oldids = dict()

clusterthresh = 1 #Time between cars before we split to separate clusters
mingap = 2.5 #Minimum allowed space between cars
timestep = mingap #Amount of time between updates. In practice, mingap rounds up to the nearest multiple of this



timedata = dict()

def run(netfile, rerouters):
    #netfile is the filepath to the network file, so we can call sumolib to get successors
    #rerouters is the list of induction loops on edges with multiple successor edges
    #We want to reroute all the cars that passed some induction loop in rerouters using A*
    
    """execute the TraCI control loop"""
    network = sumolib.net.readNet(netfile)
    startDict = dict()
    endDict = dict()
    locDict = dict()
    leftDict = dict()

    #dontBreakEverything() #Run test simulation for a step to avoid it overwriting the main one or something??

    tstart = time.time()
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep() #Tell the simulator to simulate the next time step

        #Decide whether new vehicles use our routing
        for vehicle in traci.simulation.getDepartedIDList():
            isSmart[vehicle] = random.random() < pSmart
            timedata[vehicle] = [traci.simulation.getTime(), -1, -1]
        for vehicle in traci.simulation.getArrivedIDList():
            timedata[vehicle][1] = traci.simulation.getTime()
            #print("Actual minus expected:")
            #print( (timedata[vehicle][1]-timedata[vehicle][0]) - timedata[vehicle][2])

        reroute(rerouters, network, True) #Reroute cars (including simulate-ahead cars)
        carsOnNetwork.append(len(traci.vehicle.getIDList())) #Track number of cars on network (for plotting)
        
        t = traci.simulation.getTime()
        for id in traci.simulation.getDepartedIDList():
            startDict[id] = t
            locDict[id] = traci.vehicle.getRoadID(id)
            leftDict[id] = 0
        for id in traci.simulation.getArrivedIDList():
            endDict[id] = t
            locDict.pop(id)
        for id in locDict:
            if traci.vehicle.getRoadID(id) != locDict[id] and traci.vehicle.getRoadID(id)[0] != ":":
                c0 = network.getEdge(locDict[id]).getFromNode().getCoord()
                c1 = network.getEdge(locDict[id]).getToNode().getCoord()
                theta0 = math.atan2(c1[1]-c0[1], c1[0]-c0[0])

                assert(c1 == network.getEdge(traci.vehicle.getRoadID(id)).getFromNode().getCoord())
                c2 = network.getEdge(traci.vehicle.getRoadID(id)).getToNode().getCoord()
                theta1 = math.atan2(c2[1]-c1[1], c2[0]-c1[0])

                if (theta1-theta0+math.pi)%(2*math.pi)-math.pi > 0:
                    leftDict[id] += 1
                
                locDict[id] = traci.vehicle.getRoadID(id)

        if t%100 == 0 or not traci.simulation.getMinExpectedNumber() > 0:
            #After we're done simulating... 
            plt.figure()
            plt.plot(carsOnNetwork)
            plt.xlabel("Time (s)")
            plt.ylabel("Cars on Network")
            plt.title("Congestion, Adoption Prob=" + str(pSmart))
            #plt.show() #NOTE: Blocks code execution until you close the plot
            plt.savefig("Plots/Congestion, AP=" + str(pSmart)+".png")
            plt.close()

            avgTime = 0
            avgLefts = 0
            bestTime = inf
            worstTime = 0
            for id in endDict:
                ttemp = endDict[id] - startDict[id]
                avgTime += ttemp/len(endDict)
                avgLefts += leftDict[id]/len(endDict)
                if ttemp > worstTime:
                    worstTime = ttemp
                if ttemp < bestTime:
                    bestTime = ttemp
            print("\nCurrent simulation time: %f" % t)
            print("Total run time: %f" % (time.time() - tstart))
            print("Average time in network: %f" % avgTime)
            print("Best time: %f" % bestTime)
            print("Worst time: %f" % worstTime)
            print("Average number of lefts: %f" % avgLefts)

    

#Tell all the detectors to reroute the cars they've seen
def reroute(rerouters, network, rerouteAuto=True):
    doAstar = True #Set to false to stick with SUMO default routing

    if doAstar:
        for r in rerouters:
            QueueReroute(r, network, rerouteAuto)

def QueueReroute(detector, network, rerouteAuto=True):

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

        #Decide whether we route this vehicle
        if not vehicle in isSmart and rerouteAuto:
            print("Oops, don't know " + vehicle)
            isSmart[vehicle] = random.random() < pSmart
        if rerouteAuto and isSmart[vehicle]:
            #tstart = time.time()
            

            #Convert current state

            #tstart = time.time()

            data = doClusterSim(edge, network, vehicle)
            newroute = data[0]
            tcluster = data[1]
            #print(traci.vehicle.getRoute(vehicle))
            #print(edge)
            #print(tcluster)
            if timedata[vehicle][2] == -1:
                timedata[vehicle][2] = tcluster
                
            #traci.switch("main")
            traci.vehicle.setRoute(vehicle, newroute)
    if rerouteAuto:
        oldids[detector] = ids

def doClusterSim(prevedge, net, vehicle):
    loaddata = loadClusters(prevedge, net)
    simtime = traci.simulation.getTime()

    return runClusters(net, simtime, vehicle, prevedge, loaddata)

def loadClusters(prevedge, net):
    lightinfo = dict()
    clusters = dict()

    #Cluster data structures
    for edge in traci.edge.getIDList():
        if edge[0] == ":":
            #Skip internal edges (=edges for the inside of each intersection)
            continue
        for lanenum in range(traci.edge.getLaneNumber(edge)):
            lane = edge + "_" + str(lanenum)
            clusters[lane] = []
            for vehicle in reversed(traci.lane.getLastStepVehicleIDs(lane)): #Reversed so we go from end of edge to start of edge - first clusters to leave are listed first
                
                #Process vehicle into cluster somehow
                #If nearby cluster, add to cluster in sorted order (could probably process in sorted order)
                if len(clusters[lane]) > 0 and abs(clusters[lane][-1]["time"] - traci.simulation.getTime()) < clusterthresh and abs(clusters[lane][-1]["pos"] - traci.vehicle.getLanePosition(vehicle))/net.getEdge(edge).getSpeed() < clusterthresh:
                    #Add to cluster. pos and time track newest added vehicle to see if the next vehicle merges
                    #Departure time (=time to fully clear cluster) increases, arrival doesn't
                    clusters[lane][-1]["pos"] = traci.vehicle.getLanePosition(vehicle)
                    clusters[lane][-1]["time"] = traci.simulation.getTime()
                    clusters[lane][-1]["departure"] = traci.simulation.getTime() + (traci.lane.getLength(lane)-clusters[lane][-1]["pos"])/net.getEdge(edge).getSpeed()
                    clusters[lane][-1]["cars"].append((vehicle, clusters[lane][-1]["departure"], "Load append"))
                else:
                    #Else make a new cluster
                    newcluster = dict()
                    newcluster["pos"] = traci.vehicle.getLanePosition(vehicle)
                    newcluster["time"] = traci.simulation.getTime()
                    newcluster["arrival"] = traci.simulation.getTime() + (traci.lane.getLength(edge+"_0")-newcluster["pos"])/net.getEdge(edge).getSpeed()
                    newcluster["departure"] = newcluster["arrival"]
                    newcluster["cars"] = [(vehicle, newcluster["departure"], "Load new")]
                    clusters[lane].append(newcluster)
    
    #Traffic light info
    lightinfo = dict()
    for light in traci.trafficlight.getIDList():
        lightinfo[light] = dict()
        lightinfo[light]["state"] = traci.trafficlight.getRedYellowGreenState(light)
        lightinfo[light]["switchtime"] = traci.trafficlight.getNextSwitch(light)
        lightinfo[light]["index"] = traci.trafficlight.getPhase(light)
    return (clusters, lightinfo)

#@profile
def runClusters(net, time, vehicleOfInterest, startedge, loaddata):
    #global splitinfo
    goalEdge = traci.vehicle.getRoute(vehicleOfInterest)[-1]
    splitinfo = dict()
    VOIs = [vehicleOfInterest]

    clusters = loaddata[0]
    lightinfo = loaddata[1]

    starttime = time

    edgelist = list(traci.edge.getIDList())
    edgeind = 0
    while edgeind < len(edgelist):
        if edgelist[edgeind][0] == ":":
            edgelist.pop(edgeind)
        else:
            edgeind += 1
    
    while True:
        time += timestep

        #Update lights
        for light in traci.trafficlight.getIDList():
            while time >= lightinfo[light]["switchtime"]:
                phases = traci.trafficlight.getCompleteRedYellowGreenDefinition(light)[0].phases
                lightinfo[light]["index"] += 1
                if lightinfo[light]["index"] == len(phases):
                    #At end of program, loop back to 0
                    lightinfo[light]["index"] = 0
                phaseind = lightinfo[light]["index"]
                lightinfo[light]["switchtime"] += phases[phaseind].duration
                lightinfo[light]["state"] = phases[phaseind].state

        #Sanity check for debugging infinite loops where the vehicle of interest disappears
        #This shouldn't actually go off
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
        reflist = deepcopy(edgelist) #Want to reorder edge list to handle priority stuff, but don't want to mess up the for loop indexing
        for edge in reflist:

            for lanenum in range(traci.edge.getLaneNumber(edge)):
                lane = edge + "_" + str(lanenum)

                while len(clusters[lane]) > 0:
                    cluster = clusters[lane][0]
                #for cluster in clusters[lane]:
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
                        elif not cartuple[0] in VOIs and traci.vehicle.getRoute(cartuple[0])[-1] == edge:
                            cluster["cars"].pop(0) #Remove car from this edge
                            break

                        
                        #Add car to next edge. NOTE: Enforce merging collision etc. constraints here
                        node = net.getEdge(edge).getToNode().getID()
                        if not node in blockingLinks:
                            blockingLinks[node] = []
                        #print(node.getID()) #Matches the IDs on the traffic light list
                        #print(node.getType()) #zipper #traffic_light_right_on_red #dead_end
                        #print(traci.trafficlight.getIDList())
                        #https://sumo.dlr.de/docs/TraCI/Traffic_Lights_Value_Retrieval.html
                        #If light, look up phase, decide who gets to go, merge foe streams somehow
                        #Or just separate left turn phases or something? Would mean no need to merge
                        #If no light, zipper somehow

                        #print("Coordinate testing")
                        #print(net.getLane(lane).getShape()) #[(startx, starty), (endx, endy)]

                            
                        #Figure out where the car wants to go
                        if not (cartuple[0], edge) in splitinfo:
                            #Assume zipper
                            if cartuple[0] in VOIs:
                                nextedges = []
                                #Want all edges that current lane connects to
                                for nextlinktuple in traci.lane.getLinks(lane):
                                    nextedge = nextlinktuple[0].split("_")[0]
                                    if not nextedge in nextedges:
                                        nextedges.append(nextedge)
                            else:
                                route = traci.vehicle.getRoute(cartuple[0])
                                routeind = route.index(edge)
                                nextedges = [route[routeind+1]]

                            #nextlanes is going to loop over everything in nextedges
                            #Splitty cars want to go to everything in nextlanes
                            #Non-splitty cars only want to go to one
                            nextlanes = []
                            for nextedge in nextedges:
                                for nextlanenum in range(traci.edge.getLaneNumber(nextedge)):
                                    nextlane = nextedge + "_" + str(nextlanenum)

                                    #Apparently this works...
                                    # print(isIntersecting(net.getLane(lane).getShape(), net.getLane(nextlane).getShape()))
                                    # print(isIntersecting(net.getLane(lane).getShape(), net.getLane(lane).getShape()))
                                    
                                    #If non-splitty car and this nextlane doesn't go to nextnextedge, disallow it
                                    if not cartuple[0] in VOIs:
                                        #route = traci.vehicle.getRoute(cartuple[0])
                                        #routeind = route.index(edge)
                                        if routeind + 2 < len(route): #Else there's no next next edge, don't be picky
                                            nextnextedge = route[routeind+2]
                                            usableLane = False
                                            for nextnextlinktuple in traci.lane.getLinks(nextlane):
                                                if nextnextlinktuple[0].split("_")[0] == nextnextedge: #linktuple[0].split("_")[0] gives edge the link goes to
                                                    usableLane = True
                                                    break
                                            if not usableLane: #This nextlane doesn't connect to nextnextedge
                                                continue #So try a different nextlane
                                    nextlanes.append(nextedge + "_" + str(nextlanenum))
                            splitinfo[(cartuple[0], edge)] = nextlanes
                        #Else we've already figured out how we want to split this car

                        tempnextedges = deepcopy(splitinfo[(cartuple[0], edge)])
                        for nextlane in tempnextedges:
                            nextedge = nextlane.split("_")[0]
                            #for nextlanenum in range(traci.edge.getLaneNumber(nextedge)):
                            #    nextlane = nextedge+"_"+str(nextlanenum)

                            #Check light state
                            if node in traci.trafficlight.getIDList():
                                isGreenLight = False
                                linklistlist = traci.trafficlight.getControlledLinks(node)
                                for linklistind in range(len(linklistlist)):
                                    linkstate = lightinfo[node]["state"][linklistind]

                                    if linkstate == "G" or linkstate == "g": #Next TODO: g should be blockable by opposing G
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
                                                        
                                                            conflicting = isIntersecting( (net.getLane(linktuple[0]).getShape()[1], (net.getLane(linktuple[1]).getShape()[0])), 
                                                            (net.getLane(linktuple2[0]).getShape()[1], (net.getLane(linktuple2[1]).getShape()[0])) )

                                                            if not conflicting:
                                                                continue
                                                            blocking = (linklistind2 in blockingLinks[node]) #TODO: Not clear what it'd mean to have multiple tuples in a single link, this might need to change if that happens

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
                                                                    blockerroute = traci.vehicle.getRoute(blocker)
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
                            if len(clusters[nextlane]) > 0 and abs(clusters[nextlane][-1]["time"] - time) < clusterthresh and abs(clusters[nextlane][-1]["pos"])/net.getEdge(nextedge).getSpeed() < clusterthresh:
                                
                                #Make sure there's no car on the new road that's too close
                                if not abs(clusters[nextlane][-1]["time"] - time) < mingap:
                                    #Add to cluster. pos and time track newest added vehicle to see if the next vehicle merges
                                    #Departure time (=time to fully clear cluster) increases, arrival doesn't
                                    #TODO eventually: Be more precise with time and position over partial timesteps, allowing me to use larger timesteps?
                                    clusters[nextlane][-1]["pos"] = 0
                                    clusters[nextlane][-1]["time"] = time
                                    clusters[nextlane][-1]["departure"] = time + traci.lane.getLength(nextlane)/net.getEdge(nextedge).getSpeed()
                                    if cartuple[0] in VOIs:
                                        clusters[nextlane][-1]["cars"].append((cartuple[0]+"|"+nextlane, clusters[nextlane][-1]["departure"], "Zipper append"))
                                        VOIs.append(cartuple[0]+"|"+nextlane)
                                    else:
                                        clusters[nextlane][-1]["cars"].append((cartuple[0], clusters[nextlane][-1]["departure"], "Zipper append"))
                                else:
                                    #No space, try next lane
                                    continue
                            else:
                                #There is no cluster nearby
                                #So make a new cluster
                                newcluster = dict()
                                newcluster["pos"] = 0
                                newcluster["time"] = time
                                newcluster["arrival"] = time + traci.lane.getLength(nextlane)/net.getEdge(nextedge).getSpeed()
                                newcluster["departure"] = newcluster["arrival"]
                                if cartuple[0] in VOIs:
                                    newcluster["cars"] = [(cartuple[0]+"|"+nextlane, newcluster["departure"], "Zipper new cluster")]
                                    VOIs.append(cartuple[0]+"|"+nextlane)
                                else:
                                    newcluster["cars"] = [(cartuple[0], newcluster["departure"], "Zipper new cluster")]
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
    traci.start([checkBinary('sumo'), "-c", sumoconfig,
                             "--start", "--no-step-log", "true",
                             "--xml-validation", "never"], label="setup")

    net = sumolib.net.readNet(networkfile)
    rerouters = []
    global max_edge_speed

    # #Getting edge info from sumolib
    # for edge in net.getEdges(withInternal=False): 
    #     print(edge)

    with open("additional_autogen.xml", "w") as additional:
        print("""<additional>""", file=additional)
        for edge in traci.edge.getIDList():
            if edge[0] == ":":
                #Skip internal edges (=edges for the inside of each intersection)
                continue

            if (net.getEdge(edge).getSpeed() > max_edge_speed):
                max_edge_speed = net.getEdge(edge).getSpeed()

            #print(edge)
            for lanenum in range(traci.edge.getLaneNumber(edge)):
                lane = edge+"_"+str(lanenum)
                #print(lane)
                print('    <inductionLoop id="IL_%s" freq="1" file="outputAuto.xml" lane="%s" pos="-50" friendlyPos="true" />' \
                      % (lane, lane), file=additional)
                if len(net.getEdge(edge).getOutgoing()) > 1:
                    rerouters.append("IL_"+lane)
        print("</additional>", file=additional)
    
    return rerouters

# this is the main entry point of this script
if __name__ == "__main__":
    options = get_options()

    # this script has been called from the command line. It will start sumo as a
    # server, then connect and run
    if options.nogui:
        sumoBinary = checkBinary('sumo')
    else:
        sumoBinary = checkBinary('sumo-gui')

    #NOTE: Script name is zeroth arg
    sumoconfig = sys.argv[2]
    netfile = sys.argv[1]
    rerouters = generate_additionalfile(sumoconfig, netfile)

    # this is the normal way of using traci. sumo is started as a
    # subprocess and then the python script connects and runs
    traci.start([sumoBinary, "-c", sumoconfig,
                             "--additional-files", "additional_autogen.xml",
                             "--log", "LOGFILE", "--xml-validation", "never"], label="main")

    
    run(netfile, rerouters)
    traci.close()
