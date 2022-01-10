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

from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import optparse
import random
from numpy import inf
import time
import matplotlib.pyplot as plt
from heapq import * #priorityqueue
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

carsOnNetwork = [];
max_edge_speed = 0.0;

hmetadict = dict()

oldids = dict()

clusterthresh = 5 #Time between cars before we split to separate clusters
mingap = 2.5 #Minimum allowed space between cars
clusters = dict()
VOIs = []
splitinfo = dict()

timedata = dict()

def run(netfile, rerouters):
    #netfile is the filepath to the network file, so we can call sumolib to get successors
    #rerouters is the list of induction loops on edges with multiple successor edges
    #We want to reroute all the cars that passed some induction loop in rerouters using A*
    
    """execute the TraCI control loop"""
    network = sumolib.net.readNet(netfile)
    dontBreakEverything() #Run test simulation for a step to avoid it overwriting the main one or something??
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep() #Tell the simulator to simulate the next time step

        #Decide whether new vehicles use our routing
        for vehicle in traci.simulation.getDepartedIDList():
            isSmart[vehicle] = random.random() < pSmart
            timedata[vehicle] = [traci.simulation.getTime(), -1, -1]
        for vehicle in traci.simulation.getArrivedIDList():
            timedata[vehicle][1] = traci.simulation.getTime()
            print("Actual minus expected:")
            print( (timedata[vehicle][1]-timedata[vehicle][0]) - timedata[vehicle][2])

        reroute(rerouters, network, True) #Reroute cars (including simulate-ahead cars)
        carsOnNetwork.append(len(traci.vehicle.getIDList())) #Track number of cars on network (for plotting)

    #After we're done simulating... 
    plt.figure()
    plt.plot(carsOnNetwork)
    plt.xlabel("Time (s)")
    plt.ylabel("Cars on Network")
    plt.title("Congestion, Adoption Prob=" + str(pSmart))
    #plt.show() #NOTE: Blocks code execution until you close the plot
    plt.savefig("Plots/Congestion, AP=" + str(pSmart)+".png")
    

#Tell all the detectors to reroute the cars they've seen
def reroute(rerouters, network, rerouteAuto=True):
    doAstar = True #Set to false to stick with SUMO default routing

    if doAstar:
        for r in rerouters:
            QueueReroute(r, network, rerouteAuto)
            #BFReroute(r, network, rerouteAuto)

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
        if rerouteAuto and isSmart[vehicle]: #and detector[0:5]=="IL_in":
            #tstart = time.time()
            
            saveStateInfo(edge) #Saves the traffic state and traffic light timings

            #Swap to test sim, load current state
            traci.switch("test")
            #tstart = time.time()
            #tcluster = doClusterSim(edge, network, vehicle)
            data = doClusterSim(edge, network, vehicle)
            newroute = data[0]
            tcluster = data[1]
            #print(traci.vehicle.getRoute(vehicle))
            #print(edge)
            #print(tcluster)
            if edge == "start":
                timedata[vehicle][2] = tcluster
                
            traci.switch("main")
            traci.vehicle.setRoute(vehicle, newroute)

def doClusterSim(prevedge, net, vehicle):
    global VOIs, splitinfo
    loadClusters(prevedge, net)
    time = traci.simulation.getTime()
    starttime = time
    VOIs = [vehicle]
    splitinfo = dict()

    temp = "NOT DONE"
    while temp == "NOT DONE":
        temp = stepClusters(net, time, vehicle, prevedge)
        time+=1
    #print(temp)
    #return time-starttime
    return (temp, time-starttime)

def loadClusters(prevedge, net):
    loadStateInfo(prevedge, net)
    #Test clusters
    #Cluster data structures
    #print("Start load clusters")
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


def stepClusters(net, time, vehicleOfInterest, startedge):
    global splitinfo
    #print("Start stepClusters")
    #print(time)
    #print(vehicleOfInterest)
    #print(clusters)
    goalEdge = traci.vehicle.getRoute(vehicleOfInterest)[-1]
    
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

    reflist = deepcopy(edgelist) #Want to reorder edge list to handle priority stuff, but don't want to mess up the for loop indexing
    for edge in reflist:
        if edge[0] == ":":
            #Skip internal edges (=edges for the inside of each intersection)
            continue
        for lanenum in range(traci.edge.getLaneNumber(edge)):
            lane = edge + "_" + str(lanenum)
            #clusterscopy = deepcopy(clusters[lane])
            #for cluster in clusterscopy:

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
                        splitroute = cartuple[0].split("_")
                        splitroute.pop(0)
                        fullroute = [startedge]
                        for routepart in splitroute:
                            fullroute.append(routepart)
                        return fullroute
                    elif not cartuple[0] in VOIs and traci.vehicle.getRoute(cartuple[0])[-1] == edge:
                        cluster["cars"].pop(0) #Remove car from this edge
                        break

                    
                    #Add car to next edge. NOTE: Enforce merging collision etc. constraints here
                    node = net.getEdge(edge).getToNode()
                    #print(node.getID()) #Matches the IDs on the traffic light list
                    #print(node.getType()) #zipper #traffic_light_right_on_red #dead_end
                    #print(traci.trafficlight.getIDList())
                    #https://sumo.dlr.de/docs/TraCI/Traffic_Lights_Value_Retrieval.html
                    #If light, look up phase, decide who gets to go, merge foe streams somehow
                    #Or just separate left turn phases or something? Would mean no need to merge
                    #If no light, zipper somehow
                    if node.getID() in traci.trafficlight.getIDList():

                        #First pass: Figure out priority
                        #Second pass: Go through in priority order, shortcut when possible
                        #Then flip priority if zipper
                        
                        print("I'm a traffic light!")
                        print(traci.trafficlight.getControlledLanes(node.getID()))
                        print(traci.trafficlight.getControlledLinks(node.getID()))
                        print(traci.trafficlight.getCompleteRedYellowGreenDefinition(node.getID()))
                        #traci.trafficlight.getRedYellowGreenState
                        #Look up current road, next road, check if appropriate link is green
                        #If G green, add to next queue
                        #If g green, make sure no G green, then add to queue (second pass? Buffer variable?)
                        #This might be a problem later if I make too many assumptions about lights, but I'll ignore for now
                        #Else wait
                    else:
                        #Assume zipper
                        if cartuple[0] in VOIs:
                            nextedges = getSuccessors(edge, net)
                        else:
                            route = traci.vehicle.getRoute(cartuple[0])
                            nextedges = [route[route.index(edge)+1]]

                        if not (cartuple[0], edge) in splitinfo:
                            splitinfo[(cartuple[0], edge)] = nextedges
                        #Else we've already split this car

                        tempnextedges = deepcopy(splitinfo[(cartuple[0], edge)])
                        for nextedge in tempnextedges:
                            for nextlanenum in range(traci.edge.getLaneNumber(nextedge)):
                                nextlane = nextedge+"_"+str(nextlanenum)

                                if len(clusters[nextlane]) > 0 and abs(clusters[nextlane][-1]["time"] - time) < clusterthresh and abs(clusters[nextlane][-1]["pos"])/net.getEdge(nextedge).getSpeed() < clusterthresh:
                                    #Make sure time isn't too close
                                    if not abs(clusters[nextlane][-1]["time"] - time) < mingap:
                                        #Add to cluster. pos and time track newest added vehicle to see if the next vehicle merges
                                        #Departure time (=time to fully clear cluster) increases, arrival doesn't
                                        clusters[nextlane][-1]["pos"] = 0
                                        clusters[nextlane][-1]["time"] = time
                                        clusters[nextlane][-1]["departure"] = time + traci.lane.getLength(nextlane)/net.getEdge(nextedge).getSpeed()
                                        if cartuple[0] in VOIs:
                                            clusters[nextlane][-1]["cars"].append((cartuple[0]+"_"+nextedge, clusters[nextlane][-1]["departure"], "Zipper append"))
                                            VOIs.append(cartuple[0]+"_"+nextedge)
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
                                    newcluster["arrival"] = time + traci.lane.getLength(nextedge+"_"+str(nextlanenum))/net.getEdge(nextedge).getSpeed()
                                    newcluster["departure"] = newcluster["arrival"]
                                    if cartuple[0] in VOIs:
                                        newcluster["cars"] = [(cartuple[0]+"_"+nextedge, newcluster["departure"], "Zipper new cluster")]
                                        VOIs.append(cartuple[0]+"_"+nextedge)
                                    else:
                                        newcluster["cars"] = [(cartuple[0], newcluster["departure"], "Zipper new cluster")]
                                    clusters[nextlane].append(newcluster)

                                #We've added a car to nextedge_nextlanenum.
                                splitinfo[(cartuple[0], edge)].remove(nextedge)
                                break

                        #This is in the zipper merge logic in the cartuple < time loop, breaking it loops the cluster loop
                        #If zipper merge, need to alternate priority on things
                        if len(splitinfo[(cartuple[0], edge)]) == 0:
                            edgelist.append(edgelist.pop(edgelist.index(edge))) #Push edge to end of list to give it lower priority next time
                            cluster["cars"].pop(0) #Remove car from this edge
                            if len(cluster["cars"]) > 0:
                                cartuple = cluster["cars"][0]
                                continue
                                #...and loop back try to add the next car
                                #break #So we get to the outer loop to reassign cartuple
                            else:
                                #Cluster is empty, move on
                                break

                        #If we haven't added this car to all next edges, just stop?
                        else:
                            break
                        
                #Inside: while len(clusters[lane]) > 0
                if len(cluster["cars"]) == 0:
                    clusters[lane].pop(0) #Entire cluster is done, remove it
                    continue
                else:
                    #Something's left in the first cluster, so everyone's blocked
                    break

    return "NOT DONE"

# Gets successor edges of a given edge in a given network
# Parameters:
#   edge: an edge ID string
#   network: the nwtwork object from sumolib.net.readNet(netfile)
# Returns:
#   successors: a list of edge IDs for the successor edges (outgoing edges from the next intersection)
def getSuccessors(edge, network):
    ids = []
    for succ in list(network.getEdge(edge).getOutgoing()):
        ids.append(succ.getID())
    return ids

def saveStateInfo(edge):
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

def loadStateInfo(prevedge, net):#Load traffic state
    traci.simulation.loadState("savestates/teststate_"+prevedge+".xml")
    #Load light state
    with open("savestates/lightstate_"+prevedge+".pickle", 'rb') as handle:
        lightStates = pickle.load(handle)
    #Copy traffic light timings
    for light in traci.trafficlight.getIDList():
        traci.trafficlight.setPhase(light, lightStates[light][0])
        traci.trafficlight.setPhaseDuration(light, lightStates[light][1])
                        
                        
#Magically makes the vehicle lists stop deleting themselves somehow???
def dontBreakEverything():
    traci.switch("test")
    traci.simulationStep()
    traci.switch("main")

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
    #Second simulator for running tests. No GUI
    traci.start([checkBinary('sumo'), "-c", sumoconfig,
                             "--additional-files", "additional_autogen.xml",
                             "--start", "--no-step-log", "true",
                             "--xml-validation", "never",
                             "--step-length", "1"], label="test")
    edgelist = list(traci.edge.getIDList())
    for edge in edgelist:
        if edge[0] == ":":
            edgelist.remove(edge)
    run(netfile, rerouters)
    traci.close()
