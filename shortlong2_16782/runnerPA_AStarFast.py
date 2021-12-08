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
from numpy import Inf, inf
import time
import matplotlib.pyplot as plt
from heapq import * #priorityqueue
import math

# we need to import python modules from the $SUMO_HOME/tools directory
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

from sumolib import checkBinary
import traci  #To interface with SUMO simulations
import sumolib #To query node/edge stuff about the network
import libsumo
import pickle #To save/load traffic light states

isSmart = dict(); #Store whether each vehicle does our routing or not
pSmart = 1.0; #Adoption probability

carsOnNetwork = [];
max_edge_speed = 0.0;

hmetadict = dict()

def run(netfile, rerouters):
    #netfile is the filepath to the network file, so we can call sumolib to get successors
    #rerouters is the list of induction loops on edges with multiple successor edges
    #We want to reroute all the cars that passed some induction loop in rerouters using A*
    
    """execute the TraCI control loop"""
    network = sumolib.net.readNet(netfile)
    startDict = dict()
    endDict = dict()
    vehicleIDs = []
    # dontBreakEverything() #Run test simulation for a step to avoid it overwriting the main one or something??
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep() #Tell the simulator to simulate the next time step
        reroute(rerouters, network, True) #Reroute cars (including simulate-ahead cars)
        carsOnNetwork.append(len(traci.vehicle.getIDList())) #Track number of cars on network (for plotting)
        
        t = traci.simulation.getTime()
        for id in traci.simulation.getDepartedIDList():
            startDict[id] = t
            vehicleIDs.append(id)
        for id in traci.simulation.getArrivedIDList():
            endDict[id] = t

    #After we're done simulating... 
    plt.figure()
    plt.plot(carsOnNetwork)
    plt.xlabel("Time (s)")
    plt.ylabel("Cars on Network")
    plt.title("Congestion, A* Routing")
    plt.savefig("Plots/congestion_A_star.png")

    avgTime = 0
    bestTime = Inf
    worstTime = 0
    for id in vehicleIDs:
        t = endDict[id] - startDict[id]
        avgTime += t
        if t > worstTime:
            worstTime = t
        if t < bestTime:
            bestTime = t
    avgTime /= len(vehicleIDs)
    print("Average time in network: %f" % avgTime)
    print("Best time: %f" % bestTime)
    print("Worst time: %f" % worstTime)
    

#Tell all the detectors to reroute the cars they've seen
#Devolves into a 2-line for loop once we have A* working, should probably move into AstarReroute then
def reroute(rerouters, network, rerouteAuto=True):
    doAstar = True

    if doAstar:
        for r in rerouters:
            AstarReroute(r, network, rerouteAuto)
    else:
        #NOTE: This is hard-coded for fourroute.net.xml
        
        #Or the old brute-force-all-routes method
        rerouteDetector("IL_start_0", [
                        ["start", "R41", "R42", "R43", "goal"],
                        ["start", "M3", "R31", "R32", "R33", "N3", "goal"],
                        ["start", "M3", "M2", "R21", "R22", "R23", "N2", "N3", "goal"],
                        ["start", "M3", "M2", "M1", "R11", "R12", "R13", "N1", "N2", "N3", "goal"]
                        ], network, rerouteAuto)
        rerouteDetector("IL_M3_0", [
                    ["M3", "R31", "R32", "R33", "N3", "goal"],
                    ["M3", "M2", "R21", "R22", "R23", "N2", "N3", "goal"],
                    ["M3", "M2", "M1", "R11", "R12", "R13", "N1", "N2", "N3", "goal"]
                    ], network, rerouteAuto)
        rerouteDetector("IL_M2_0", [
                    ["M2", "R21", "R22", "R23", "N2", "N3", "goal"],
                    ["M2", "M1", "R11", "R12", "R13", "N1", "N2", "N3", "goal"]
                    ], network, rerouteAuto)

# Distance between the end points of the two edges as heuristic
def heuristic(net, curredge, goaledge):
    #return 0
    goalEnd = net.getEdge(goaledge).getToNode().getCoord() 
    currEnd = net.getEdge(curredge).getToNode().getCoord() 
    dist = math.sqrt((goalEnd[0] - currEnd[0])**2 + (goalEnd[1] - currEnd[1])**2)
    return dist / max_edge_speed

def backwardDijkstra(network, goal):
    gvals = dict()
    gvals[goal] = 0
    pq = []
    heappush(pq, (0, goal))

    while len(pq) > 0: #When the queue is empty, we're done
        #print(pq)
        stateToExpand = heappop(pq)
        #fval = stateToExpand[0]
        edge = stateToExpand[1]
        gval = gvals[edge]

        #Get predecessor IDs
        succs = []
        for succ in list(network.getEdge(edge).getIncoming()):
            succs.append(succ.getID())
        
        for succ in succs:
            c = traci.lane.getLength(edge+"_0")/network.getEdge(edge).getSpeed()
            #c = getEdgeCost(vehicle, succ, edgeToExpand, network, gval)

            # heuristic: distance from mid-point of edge to mid point of goal edge
            h = 0
            if succ in gvals and gvals[succ] <= gval+c:
                #Already saw this state, don't requeue
                continue

            #Otherwise it's new or we're now doing better, so requeue it
            gvals[succ] = gval+c
            heappush(pq, (gval+c+h, succ))
    return gvals

def AstarReroute(detector, network, rerouteAuto=True):
    #print("Warning: A* routing not implemented for router " + detector)

    ids = traci.inductionloop.getLastStepVehicleIDs(detector) #All vehicles to be rerouted
    if len(ids) == 0:
        #No cars to route, we're done here
        return

    # getRoadID: Returns the edge id the vehicle was last on
    edge = traci.vehicle.getRoadID(ids[0])
    
    for vehicle in ids:

        #Decide whether we route this vehicle
        if not vehicle in isSmart:
            isSmart[vehicle] = random.random() < pSmart
        if isSmart[vehicle] and rerouteAuto:
            saveStateInfo(edge) #Saves the traffic state and traffic light timings
        
            #Get goal
            route = traci.vehicle.getRoute(vehicle)
            goaledge = route[-1]

            if not goaledge in hmetadict:
                hmetadict[goaledge] = backwardDijkstra(network, goaledge)

            stateinfo = dict()
            stateinfo[edge] = dict()
            stateinfo[edge]['gval'] = 0
            stateinfo[edge]['path'] = [edge]
            #Store whatever else you need here

            pq = [] #Priority queue
            heappush(pq, (stateinfo[edge]['gval'], edge))

            while len(pq) > 0: #If the queue is empty, the route is impossible. This should never happen, but if it does we don't change the route.
                #print(pq)
                stateToExpand = heappop(pq)
                #fval = stateToExpand[0]
                edgeToExpand = stateToExpand[1]
                gval = stateinfo[edgeToExpand]['gval']

                #Check goal, update route, break out of loop
                if edgeToExpand == goaledge:
                    traci.vehicle.setRoute(vehicle, stateinfo[goaledge]['path'])
                    break #Done routing this vehicle

                succs = getSuccessors(edgeToExpand, network)
                for succ in succs:
                    if not succ in hmetadict[goaledge]:
                        #Dead end, don't bother
                        continue
                    
                    c = getEdgeCost(vehicle, succ, edgeToExpand, network, gval)

                    # heuristic: distance from mid-point of edge to mid point of goal edge
                    # h = heuristic(network, succ, goaledge)
                    h = hmetadict[goaledge][succ]
                    if succ in stateinfo and stateinfo[succ]['gval'] <= gval+c:
                        #Already saw this state, don't requeue
                        continue

                    #Otherwise it's new or we're now doing better, so requeue it
                    stateinfo[succ] = dict()
                    stateinfo[succ]['gval'] = gval+c
                    temppath = stateinfo[edgeToExpand]['path'].copy()
                    temppath.append(succ)
                    stateinfo[succ]['path'] = temppath
                    heappush(pq, (gval+c+h, succ))
                
        if not isSmart[vehicle]:
            #TODO: Turn randomly
            #Can't just steal from old rerouteDetector code if we don't know possible routes
            #Could just turn randomly and stop if you fall off the network...
            #Can deal with this later, for now I'll just set psmart=1
            print("TODO: Turn randomly")

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

def saveStateInfoLibsumo(edge):
    #Copy state from main sim to test sim
    libsumo.simulation.saveState("savestates/teststate_"+edge+".xml")
    #saveState apparently doesn't save traffic light states despite what the docs say
    #So save all the traffic light states and copy them over
    lightStates = dict()
    for light in libsumo.trafficlight.getIDList():
        lightStates[light] = [libsumo.trafficlight.getPhase(light), libsumo.trafficlight.getPhaseDuration(light)]
        #Why do the built-in functions have such terrible names?!
        lightStates[light][1] = libsumo.trafficlight.getNextSwitch(light) - libsumo.simulation.getTime()
    #Save lightStates to a file
    with open("savestates/lightstate_"+edge+".pickle", 'wb') as handle:
        pickle.dump(lightStates, handle, protocol=pickle.HIGHEST_PROTOCOL)

def loadStateInfo(prevedge):
    #Load traffic state
    libsumo.simulation.loadState("savestates/teststate_"+prevedge+".xml")
    #Load light state
    with open("savestates/lightstate_"+prevedge+".pickle", 'rb') as handle:
        lightStates = pickle.load(handle)
    #Copy traffic light timings
    for light in traci.trafficlight.getIDList():
        libsumo.trafficlight.setPhase(light, lightStates[light][0])
        libsumo.trafficlight.setPhaseDuration(light, lightStates[light][1])
    

#I think this works
#TODO: Consider stopping A* expansions and using current average speed for big g_value
def getEdgeCost(vehicle, edge, prevedge, network, g_value):
    # traci.switch("test")
    loadStateInfo(prevedge)

    #Tell the vehicle to drive to the end of edge
    libsumo.vehicle.setRoute(vehicle, [prevedge, edge])
    
    #Run simulation, track time to completion
    t = 0
    keepGoing = True
    while(keepGoing):
        libsumo.simulationStep()
        # reroute(rerouters, network, False) #Randomly reroute the non-adopters
        #NOTE: I'm modeling non-adopters as randomly rerouting at each intersection
        #So whether or not I reroute them here, I'm still wrong compared to the main simulation (where they will reroute randomly)
        #This is good - the whole point is we can't simulate exactly what they'll do
        t+=1
        for lanenum in range(libsumo.edge.getLaneNumber(edge)):
            ids = libsumo.inductionloop.getLastStepVehicleIDs("IL_"+edge+"_"+str(lanenum))
            if vehicle in ids:
                keepGoing = False
                break
    saveStateInfoLibsumo(edge) #Need this to continue the A* search
    # traci.switch("main")
    #print("Edge " + edge + " took " + str(t) + " seconds")
    return t

#Send all cars that hit detector down one of the routes in routes
#TODO: Once we have A* working, we don't need this function
def rerouteDetector(detector, routes, network, rerouteAuto=True):
    ids = traci.inductionloop.getLastStepVehicleIDs(detector)
    for i in range(len(ids)):
        #If we haven't decided whether to route it or not, decide now
        if not ids[i] in isSmart:
            isSmart[ids[i]] = random.random() < pSmart

        #If we're routing it, and we're recomputing routes, do so
        if isSmart[ids[i]]:
            traci.vehicle.setColor(ids[i], [0, 255, 0]) #Green = we're routing
            if detector == "RerouterL" or detector == "RerouterR":
                traci.vehicle.setColor(ids[i], [0, 255, 255]) #Blue = from side
            if rerouteAuto:
                #print(traci.vehicle.getRoadID(ids[i]))
                route = getShortestRoute(routes, ids[i], rerouters, network)
                #print(route)
                traci.vehicle.setRoute(ids[i], route)
            continue
        #If we're not routing it, randomly pick a route
        traci.vehicle.setColor(ids[i], [255, 0, 0]) #Red = random routing
        if detector == "RerouterL" or detector == "RerouterR":
                traci.vehicle.setColor(ids[i], [255, 0, 255]) #Blue = from side
        #Pick random route
        r = random.random()
        nroutes = len(routes)
        for j in range(nroutes):
            if r < 1.0/nroutes:
                traci.vehicle.setRoute(ids[i], routes[j])
                break
            else:
                r -= 1.0/nroutes

#Magically makes the vehicle lists stop deleting themselves somehow???
def dontBreakEverything():
    traci.switch("test")
    traci.simulationStep()
    traci.switch("main")

#TODO: Move simulate ahead logic into edgeCosts function. Once A* is working we don't need this function
def getShortestRoute(routes, vehicle, rerouters, network):
    #Save time on trivial cases
    if len(routes) == 1:
        return routes[0]
    
    #Copy state from main sim to test sim
    traci.simulation.saveState("teststate.xml")
    #saveState apparently doesn't save traffic light states despite what the docs say
    #So save all the traffic light states and copy them over
    lightStates = dict()
    for light in traci.trafficlight.getIDList():
        lightStates[light] = [traci.trafficlight.getPhase(light), traci.trafficlight.getPhaseDuration(light)]
        #Why do the built-in functions have such terrible names?!
        lightStates[light][1] = traci.trafficlight.getNextSwitch(light) - traci.simulation.getTime()

    traci.switch("test")
    
    bestroute = "None"
    besttime = float('inf')
    for route in routes:
        #Load traffic state
        traci.simulation.loadState("teststate.xml")
        #Copy traffic light timings
        for light in traci.trafficlight.getIDList():
            traci.trafficlight.setPhase(light, lightStates[light][0])
            traci.trafficlight.setPhaseDuration(light, lightStates[light][1])
        traci.vehicle.setRoute(vehicle, route)

        #Run simulation, track time to completion
        t = 0
        while(vehicle in traci.vehicle.getIDList() and t < besttime):
            traci.simulationStep()
            reroute(rerouters, network, False) #Randomly reroute the non-adopters
            #NOTE: I'm modeling non-adopters as randomly rerouting at each intersection
            #So whether or not I reroute them here, I'm still wrong compared to the main simulation (where they will reroute randomly)
            #This is good - the whole point is we can't simulate exactly what they'll do
            t+=1
        if t < besttime:
            besttime = t
            bestroute = route
    traci.switch("main")
    return bestroute

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
    print("MAX_EDGE_SPEED 2.0: {}".format(max_edge_speed))

    # this is the normal way of using traci. sumo is started as a
    # subprocess and then the python script connects and runs
    traci.start([sumoBinary, "-c", sumoconfig,
                             "--additional-files", "additional_autogen.xml",
                             "--log", "LOGFILE", "--xml-validation", "never"], label="main")
    #Second simulator for running tests. No GUI
    libsumo.start([checkBinary('sumo'), "-c", sumoconfig,
                             "--additional-files", "additional_autogen.xml",
                             "--start", "--no-step-log", "true",
                             "--xml-validation", "never",
                             "--step-length", "1"])
    run(netfile, rerouters)
    traci.close()
    libsumo.close()
