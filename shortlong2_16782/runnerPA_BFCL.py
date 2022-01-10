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

hmetadict = dict()

oldids = dict()

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
            BFReroute(r, network, rerouteAuto)



def BFReroute(detector, network, rerouteAuto=True):

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
            loadStateInfo(edge)
    
            #Get goal
            route = traci.vehicle.getRoute(vehicle)
            goaledge = route[-1]

            t = 0
            keepGoing = True
            newids = dict()
            #tstart = time.time()

            #Initial split
            newvs = splitVehicle(vehicle, network)
            newids[detector] = newvs
            
            while(keepGoing):
                
                #Continue with counterfactual simulation
                traci.simulationStep()
                t+=1

                #Check if we're done
                for lanenum in range(traci.edge.getLaneNumber(goaledge)):
                    testids = traci.inductionloop.getLastStepVehicleIDs("IL_"+goaledge+"_"+str(lanenum))
                    for testv in testids:
                        if testv in newvs:
                            #Reroute the car, then say we're done
                            stuff = testv.split("_")
                            outroute = [edge]
                            for i in range(1,len(stuff)):
                                outroute.append(stuff[i])
                            keepGoing = False
                            break
                        
                #Check if we need to split anything
                for rerouter in traci.inductionloop.getIDList():
                    testids = traci.inductionloop.getLastStepVehicleIDs(rerouter)
                    for testv in testids:
                        if testv in newvs and not (rerouter in newids and testv in newids[rerouter]):
                            #print("Splitting")
                            #splittime = time.time()
                            newnewvs = splitVehicle(testv, network)
                            newvs.remove(testv)
                            newvs = newvs + newnewvs
                            if not rerouter in newids:
                                newids[rerouter] = []
                            newids[rerouter] += newnewvs
                            #print(time.time() - splittime)

            
            traci.switch("main")
            traci.vehicle.setRoute(vehicle, outroute)
            #print(time.time() - tstart)
                
        if vehicle in isSmart and not isSmart[vehicle]: #TODO: Reconsider how we treat the vehicles that somehow haven't entered the network in main yet
            #TODO: Turn randomly
            #Can't just steal from old rerouteDetector code if we don't know possible routes
            #Could just turn randomly and stop if you fall off the network...
            #Or use Sumo default routing, but then we'd know what they're doing...
            #Can deal with this later, for now I'll just set psmart=1
            print("TODO: Turn randomly")
    if rerouteAuto:
        oldids[detector] = ids

def splitVehicle(vehicle, network):
    newvs = []
    edge = traci.vehicle.getRoadID(vehicle)

    succs = getSuccessors(edge, network)
    #TODO make sure these are ordered CCW from current edge
    for succ in succs:
        route = [edge, succ]


        #rstart = time.time()
        if not str(route) in traci.route.getIDList():
            traci.route.add(str(route), route)
        else:
            #In case we already have the route, we'll get an error; ignore it
            pass
        #print(time.time() - rstart)

    
        lane = traci.vehicle.getLaneIndex(vehicle)
        #lane = 0
        pos = traci.vehicle.getLanePosition(vehicle)
        speed = traci.vehicle.getSpeed(vehicle)
        pos = -50
        #speed = "max"
    
        newv = vehicle+"_"+succ
        traci.vehicle.add(newv, str(route), typeID="ghost", departLane=lane, departPos=pos, departSpeed=speed)
        newvs.append(newv)
        traci.vehicle.setColor(newv, [0, 0, 255])
        traci.vehicle.setLength(newv, traci.vehicle.getLength(vehicle)/len(succs))

    for v in traci.edge.getLastStepVehicleIDs(edge):
        if traci.vehicle.getLanePosition(v) < traci.vehicle.getLanePosition(vehicle):
            traci.vehicle.setColor(v, [255, 0, 0])
            traci.vehicle.remove(v)
        elif traci.vehicle.getLanePosition(v) > traci.vehicle.getLanePosition(vehicle):
            traci.vehicle.setColor(v, [0, 255, 0])
            #traci.vehicle.remove(v)
    traci.vehicle.remove(vehicle)
    return newvs

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
    traci.start([checkBinary('sumo-gui'), "-c", sumoconfig,
                             "--additional-files", "additional_autogen.xml",
                             "--start", "--no-step-log", "true",
                             "--xml-validation", "never",
                             "--step-length", "1"], label="test")
    run(netfile, rerouters)
    traci.close()
