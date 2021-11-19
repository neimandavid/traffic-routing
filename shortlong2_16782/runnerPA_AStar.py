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

def run(netfile, rerouters):
    #netfile is the filepath to the network file, so we can call sumolib to get successors
    #rerouters is the list of induction loops on edges with multiple successor edges
    #We want to reroute all the cars that passed some induction loop in rerouters using A*
    
    """execute the TraCI control loop"""
    dontBreakEverything() #Run test simulation for a step to avoid it overwriting the main one or something??
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep() #Tell the simulator to simulate the next time step
        reroute(rerouters, True) #Reroute cars (including simulate-ahead cars)
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
#Devolves into a 2-line for loop once we have A* working, should probably move into AstarReroute then
def reroute(rerouters, rerouteAuto=True):
    for r in rerouters:
        AstarReroute(r, rerouteAuto)
    
    #Bottom intersection
    rerouteDetector("IL_start_0", ["SLL0", "SLR0", "SRL0", "SRR0"], rerouteAuto)
    rerouteDetector("IL_start_1", ["SLL0", "SLR0", "SRL0", "SRR0"], rerouteAuto)
    #Left intersection
    rerouteDetector("IL_L_0", ["SLR", "SLL"], rerouteAuto)
    rerouteDetector("IL_startL_0", ["LR", "LL"], rerouteAuto)
    #Right intersection
    rerouteDetector("IL_R_0", ["SRL", "SRR"], rerouteAuto)
    rerouteDetector("IL_startR_0", ["RL", "RR"], rerouteAuto)

def AstarReroute(detector, rerouteAuto=True):
    #print("Warning: A* routing not implemented for router " + detector)

    ids = traci.inductionloop.getLastStepVehicleIDs(detector) #All vehicles to be rerouted
    if len(ids) == 0:
        #No cars to route, we're done here
        return
    edge = traci.vehicle.getRoadID(ids[0])
    saveStateInfo(edge) #Saves the traffic state and traffic light timings
    traci.switch("test")
    
    for vehicle in ids:
        #Decide whether we route this vehicle
        if not vehicle in isSmart:
            isSmart[vehicle] = random.random() < pSmart
        if isSmart[vehicle] and rerouteAuto:
            #TODO: Route these cars using A*.
            #INSERT A* CODE OR FUNCTION CALL TO A* CODE HERE
            #Note: You can use sumolib to get edges following edges or vertices. Ex: https://stackoverflow.com/questions/58753690/can-we-get-the-list-of-followed-edges-of-the-current-edge
            print("TODO: A* routing")
            #Test edge costs
            #print(getEdgeCost(vehicle, "L", edge, 0)) #Quick test of edge cost, errors out if next edge can't be "L"
        if not isSmart[vehicle]:
            #TODO: Turn randomly
            #Can't just steal from old rerouteDetector code if we don't know possible routes
            #Could just turn randomly and stop if you fall off the network...
            #Can deal with this later, for now I'll just set psmart=1
            print("TODO: Turn randomly")
            
    traci.switch("main")
    
def saveStateInfo(edge):
    #Copy state from main sim to test sim
    traci.simulation.saveState("teststate_"+edge+".xml")
    #saveState apparently doesn't save traffic light states despite what the docs say
    #So save all the traffic light states and copy them over
    lightStates = dict()
    for light in traci.trafficlight.getIDList():
        lightStates[light] = [traci.trafficlight.getPhase(light), traci.trafficlight.getPhaseDuration(light)]
        #Why do the built-in functions have such terrible names?!
        lightStates[light][1] = traci.trafficlight.getNextSwitch(light) - traci.simulation.getTime()
    #Save lightStates to a file
    with open("lightstate_"+edge+".pickle", 'wb') as handle:
        pickle.dump(lightStates, handle, protocol=pickle.HIGHEST_PROTOCOL)

def loadStateInfo(prevedge):
    #Load traffic state
    traci.simulation.loadState("teststate_"+prevedge+".xml")
    #Load light state
    with open("lightstate_"+prevedge+".pickle", 'rb') as handle:
        lightStates = pickle.load(handle)
    #Copy traffic light timings
    for light in traci.trafficlight.getIDList():
        traci.trafficlight.setPhase(light, lightStates[light][0])
        traci.trafficlight.setPhaseDuration(light, lightStates[light][1])
    

#I think this works
#TODO: Consider stopping A* expansions and using current average speed for big g_value
def getEdgeCost(vehicle, edge, prevedge, g_value):
    traci.switch("test")
    loadStateInfo(prevedge)

    #Tell the vehicle to drive to the end of edge
    traci.vehicle.setRoute(vehicle, [prevedge, edge])
    
    #Run simulation, track time to completion
    t = 0
    keepGoing = True
    while(keepGoing):
        traci.simulationStep()
        reroute(rerouters, False) #Randomly reroute the non-adopters
        #NOTE: I'm modeling non-adopters as randomly rerouting at each intersection
        #So whether or not I reroute them here, I'm still wrong compared to the main simulation (where they will reroute randomly)
        #This is good - the whole point is we can't simulate exactly what they'll do
        t+=1
        for lanenum in range(traci.edge.getLaneNumber(edge)):
            ids = traci.inductionloop.getLastStepVehicleIDs("IL_"+edge+"_"+str(lanenum))
            if vehicle in ids:
                keepGoing = False
                break
    saveStateInfo(edge) #Need this to continue the A* search
    traci.switch("main")
    #print("Edge " + edge + " took " + str(t) + " seconds")
    return t

#Send all cars that hit detector down one of the routes in routes
#TODO: Once we have A* working, we don't need this function
def rerouteDetector(detector, routes, rerouteAuto=True):
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
                traci.vehicle.setRouteID(ids[i], getShortestRoute(routes, ids[i], rerouters))
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
                traci.vehicle.setRouteID(ids[i], routes[j])
                break
            else:
                r -= 1.0/nroutes

#Magically makes the vehicle lists stop deleting themselves somehow???
def dontBreakEverything():
    traci.switch("test")
    traci.simulationStep()
    traci.switch("main")

#TODO: Move simulate ahead logic into edgeCosts function. Once A* is working we don't need this function
def getShortestRoute(routes, vehicle, rerouters):
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
        traci.vehicle.setRouteID(vehicle, route)

        #Run simulation, track time to completion
        t = 0
        while(vehicle in traci.vehicle.getIDList() and t < besttime):
            traci.simulationStep()
            reroute(rerouters, False) #Randomly reroute the non-adopters
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
    
    with open("additional_autogen.xml", "w") as additional:
        print("""<additional>""", file=additional)
        for edge in traci.edge.getIDList():
            if edge[0] == ":":
                #Skip internal edges (=edges for the inside of each intersection)
                continue
            print(edge)
            for lanenum in range(traci.edge.getLaneNumber(edge)):
                lane = edge+"_"+str(lanenum)
                print(lane)
                print('    <inductionLoop id="IL_%s" freq="1" file="outputAuto.xml" lane="%s" pos="%i" friendlyPos="true" />' \
                      % (lane, lane, traci.lane.getLength(lane)-50), file=additional)
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
        
    sumoconfig = "shortlong.sumocfg"
    netfile = "shortlong.net.xml" #A* people probably need this passed around in run() as well
    rerouters = generate_additionalfile(sumoconfig, netfile)
    print(rerouters)

    # this is the normal way of using traci. sumo is started as a
    # subprocess and then the python script connects and runs
    traci.start([sumoBinary, "-c", "shortlong.sumocfg",
                             "--additional-files", "additional_autogen.xml",
                             "--log", "LOGFILE", "--xml-validation", "never"], label="main")
    #Second simulator for running tests. No GUI
    traci.start([checkBinary('sumo'), "-c", "shortlong.sumocfg",
                             "--additional-files", "additional_autogen.xml",
                             "--start", "--no-step-log", "true",
                             "--xml-validation", "never",
                             "--step-length", "1"], label="test")
    run(netfile, rerouters)
    traci.close()
