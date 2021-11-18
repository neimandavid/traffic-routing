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

from sumolib import checkBinary  # noqa
import traci  # noqa

isSmart = dict(); #Store whether each vehicle does our routing or not
pSmart = 1.0; #Adoption probability

carsOnNetwork = [];

def run():
    """execute the TraCI control loop"""
    step = 0
    dontBreakEverything()
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        reroute(True)
        carsOnNetwork.append(len(traci.vehicle.getIDList()))
        step+=1
        
    plt.figure()
    plt.plot(carsOnNetwork)
    plt.xlabel("Time (s)")
    plt.ylabel("Cars on Network")
    plt.title("Congestion, Adoption Prob=" + str(pSmart))
    #plt.legend(["Regret (left turn)", "Regret (right turn)"])
    #plt.show()
    plt.savefig("Plots/Congestion, AP=" + str(pSmart)+".png")
        
    
    traci.close()
    sys.stdout.flush()

#Tell all the detectors to reroute the cars they've seen
def reroute(rerouteAuto=True):
    rerouteDetector("RerouterS0", ["SLL0", "SLR0", "SRL0", "SRR0"], rerouteAuto)
    rerouteDetector("RerouterS1", ["SLL0", "SLR0", "SRL0", "SRR0"], rerouteAuto)
    rerouteDetector("RerouterSL", ["SLR", "SLL"], rerouteAuto)
    #rerouteDetector("RerouterSL", ["SLL"], rerouteAuto)

    rerouteDetector("RerouterL", ["LR", "LL"], rerouteAuto)
    #rerouteDetector("RerouterL", ["LL"], rerouteAuto)

    rerouteDetector("RerouterSR", ["SRL", "SRR"], rerouteAuto)
    #rerouteDetector("RerouterSR", ["SRR"], rerouteAuto)
    rerouteDetector("RerouterR", ["RL", "RR"], rerouteAuto)
    #rerouteDetector("RerouterR", ["RR"], rerouteAuto)


#Send all cars that hit detector down one of the routes in routes
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
                traci.vehicle.setRouteID(ids[i], getShortestRoute(routes, ids[i]))
            continue
        #If we're not routing it, randomly pick a route
        traci.vehicle.setColor(ids[i], [255, 0, 0]) #Red = random routing
        if detector == "RerouterL" or detector == "RerouterR":
                traci.vehicle.setColor(ids[i], [255, 0, 255]) #Blue = from side
        r = random.random()
        nroutes = len(routes)
        for j in range(nroutes):
            #if r < 1.0/nroutes:
            if r < 1.9/nroutes:
                traci.vehicle.setRouteID(ids[i], routes[j])
                break
            else:
                r -= 1.0/nroutes

#Magically makes the vehicle lists stop deleting themselves somehow???
def dontBreakEverything():
    #traci.simulation.saveState("teststate.xml")
    traci.switch("test")
    #traci.simulation.loadState("teststate.xml")
    traci.simulationStep()
    traci.switch("main")

def getShortestRoute(routes, vehicle):
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
    #traci.simulationStep() #So stuff doesn't break? Not sure we need this
    
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
            reroute(False) #Randomly reroute the non-adopters
            #NOTE: I'm modeling non-adopters as randomly rerouting at each intersection
            #So whether or not I reroute them here, I'm still wrong compared to the main simulation (where they will reroute randomly)
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


# this is the main entry point of this script
if __name__ == "__main__":
    options = get_options()

    # this script has been called from the command line. It will start sumo as a
    # server, then connect and run
    if options.nogui:
        sumoBinary = checkBinary('sumo')
    else:
        sumoBinary = checkBinary('sumo-gui')

    # first, generate the route file for this simulation
    #generate_routefile()

    # this is the normal way of using traci. sumo is started as a
    # subprocess and then the python script connects and runs
    traci.start([sumoBinary, "-c", "shortlong.sumocfg",
                             "--tripinfo-output", "tripinfo.xml",
                             "--additional-files", "additional.xml",
                             "--log", "LOGFILE", "--xml-validation", "never"], label="main")
    #Second simulator for running tests. No GUI
    traci.start([checkBinary('sumo'), "-c", "shortlong.sumocfg",
                             "--tripinfo-output", "tripinfo.xml",
                             "--additional-files", "additional.xml",
                             "--start", "--no-step-log", "true",
                             "--xml-validation", "never",
                             "--step-length", "1"], label="test")
    run()
