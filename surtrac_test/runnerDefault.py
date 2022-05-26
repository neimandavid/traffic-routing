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
AStarCutoff = 200;

hmetadict = dict()

oldids = dict()
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

    dontBreakEverything() #Run test simulation for a step to avoid it overwriting the main one or something??

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
    doAstar = False #Set to false to stick with SUMO default routing

    if doAstar:
        for r in rerouters:
            AstarReroute(r, network, rerouteAuto)

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
            tstart = time.time()
            saveStateInfo(edge) #Saves the traffic state and traffic light timings
        
            #Get goal
            route = traci.vehicle.getRoute(vehicle)
            goaledge = route[-1]

            if not goaledge in hmetadict:
                hmetadict[goaledge] = backwardDijkstra(network, goaledge)
                #print(hmetadict)

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
                    #h = heuristic(network, succ, goaledge)
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
            print(time.time() - tstart)
                
        if vehicle in isSmart and not isSmart[vehicle]: #TODO: Reconsider how we treat the vehicles that somehow haven't entered the network in main yet
            #TODO: Turn randomly
            #Can't just steal from old rerouteDetector code if we don't know possible routes
            #Could just turn randomly and stop if you fall off the network...
            #Or use Sumo default routing, but then we'd know what they're doing...
            #Can deal with this later, for now I'll just set psmart=1
            print("TODO: Turn randomly")
    if rerouteAuto:
        oldids[detector] = ids

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

def loadStateInfo(prevedge):
    #Load traffic state
    traci.simulation.loadState("savestates/teststate_"+prevedge+".xml")
    #Load light state
    with open("savestates/lightstate_"+prevedge+".pickle", 'rb') as handle:
        lightStates = pickle.load(handle)
    #Copy traffic light timings
    for light in traci.trafficlight.getIDList():
        traci.trafficlight.setPhase(light, lightStates[light][0])
        traci.trafficlight.setPhaseDuration(light, lightStates[light][1])
    

#Calls the simulator (or doesn't, if g_value > AStarCutoff)
#Timing notes: Sim is about 3x load, load is about 3x save
def getEdgeCost(vehicle, edge, prevedge, network, g_value):
    #If we're simulating way into the future, do math instead
    if g_value > AStarCutoff:
        print("Stopping A* and doing math")
        return traci.edge.getTraveltime(edge)

    traci.switch("test")
    #tstart = time.time()
    loadStateInfo(prevedge)
    #print("End load")
    #print(time.time() - tstart)

    #Tell the vehicle to drive to the end of edge
    traci.vehicle.setRoute(vehicle, [prevedge, edge])
    
    #Run simulation, track time to completion
    t = 0
    keepGoing = True
    #tstart = time.time()
    while(keepGoing):
        traci.simulationStep()
        reroute(rerouters, network, False) #Randomly reroute the non-adopters
        #NOTE: I'm modeling non-adopters as randomly rerouting at each intersection
        #So whether or not I reroute them here, I'm still wrong compared to the main simulation (where they will reroute randomly)
        #This is good - the whole point is we can't simulate exactly what they'll do
        t+=1
        for lanenum in range(traci.edge.getLaneNumber(edge)):
            ids = traci.inductionloop.getLastStepVehicleIDs("IL_"+edge+"_"+str(lanenum))
            if vehicle in ids:
                keepGoing = False
                break
    #print("End sim")
    #print(time.time() - tstart)
    #tstart = time.time()
    saveStateInfo(edge) #Need this to continue the A* search
    #print("End save")
    #print(time.time() - tstart)
    
    traci.switch("main")
    return t

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
    if len(sys.argv) == 2:
        sumoconfig = sys.argv[1]
        (netfile, junk) = readSumoCfg(sumoconfig)
    else:
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
    traci.start([checkBinary('sumo'), "-c", sumoconfig,
                             "--additional-files", "additional_autogen.xml",
                             "--start", "--no-step-log", "true",
                             "--xml-validation", "never",
                             "--step-length", "1"], label="test")
    run(netfile, rerouters)
    traci.close()
