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

#Plotting travel times
entryTimes = dict();
nDone = 0;
totalTime = 0;
avgTravelTime = float("nan");

routerInfo = dict();
routerPlotInfo = dict();

def run():
    """execute the TraCI control loop"""
    step = 0
    dontBreakEverything()
    try: #Errors out if directory already exists
        os.mkdir("Plots/NEW")
    except OSError as error:
        print(error)

    #Set up plotting variables
    nCarsOnNetwork = [];
    avgTravelTimes = [];
    routers = ["RerouterS0", "RerouterS1", "RerouterSL", "RerouterL", "RerouterSR", "RerouterR"]
    for router in routers:
        routerPlotInfo[router] = dict()
        routerPlotInfo[router]["pchanged"] = []
        routerPlotInfo[router]["ntotal"] = []
        
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        step+=1 #Track time for plotting
        reroute(True) #Reroute everything
        updateTimes(["RerouterS0", "RerouterS1"], ["RerouterG0", "RerouterG1", "RerouterG2"], step)
        nCarsOnNetwork.append(len(traci.vehicle.getIDList()))
        avgTravelTimes.append(avgTravelTime)
        for router in routers:
            for route in routerInfo[router]:
                if route != "nchanged" and route != "ntotal" and route != "nsmart":
                    if not route in routerPlotInfo[router]:
                        #Copy appropriately-sized NaN array
                        routerPlotInfo[router][route] = routerPlotInfo[router]["pchanged"].copy()
                    if routerInfo[router]["ntotal"] == 0:
                        routerPlotInfo[router][route].append(float("nan"))
                    else:
                        routerPlotInfo[router][route].append(routerInfo[router][route]/routerInfo[router]["ntotal"])
            if routerInfo[router]["nsmart"] == 0:
                routerPlotInfo[router]["pchanged"].append(float("nan"))
            else:
                routerPlotInfo[router]["pchanged"].append(routerInfo[router]["nchanged"]/routerInfo[router]["nsmart"])
            routerPlotInfo[router]["ntotal"].append(routerInfo[router]["ntotal"])
        if step % 1000 == 0:
            plotThings(nCarsOnNetwork, avgTravelTimes, routerPlotInfo)

    
    plotThings(nCarsOnNetwork, avgTravelTimes, routerPlotInfo)
    
    traci.close()
    sys.stdout.flush()

def plotThings(nCarsOnNetwork, avgTravelTimes, routerPlotInfo):
    color='tab:blue'
    fig, ax1 = plt.subplots()
    ax1.plot(nCarsOnNetwork, color=color)
    ax1.set_xlabel("Simulation time step (s)")
    ax1.set_ylabel("Cars on Network", color=color)
    plt.title("Cars on Network and Travel Time,AP=" + str(pSmart))
    color='tab:red'
    ax2 = ax1.twinx()
    ax2.plot(avgTravelTimes, color=color)
    ax2.set_ylabel("Average travel time (s)", color=color)
    fig.tight_layout()
    plt.savefig("Plots/NEW/Congestion,AP=" + str(pSmart)+".png")

    for router in routerPlotInfo:
        plt.figure()
        #color='tab:blue'
        fig, ax1 = plt.subplots()
        legend = []
        for route in routerPlotInfo[router]:
            if route != "ntotal":
                ax1.plot(routerPlotInfo[router][route])
                legend.append(route)
        ax1.set_xlabel("Simulation time step (s)")
        ax1.set_ylabel("Proportion of cars")#, color=color)
        plt.title("Routing, AP=" + str(pSmart)+", Router="+router)
        color='tab:red'
        ax2 = ax1.twinx()
        ax2.plot(routerPlotInfo[router]["ntotal"], color=color)
        ax2.set_ylabel("Total number of cars", color=color)
        legend.append("ntotal")
        fig.legend(legend)
        fig.tight_layout()
        plt.savefig("Plots/NEW/Routing,AP=" + str(pSmart)+",Router="+router+".png")


##        plt.figure()
##        legend = []
##        for route in routerPlotInfo[router]:
##            plt.plot(routerPlotInfo[router][route])
##            legend.append(route)
##        plt.xlabel("Simulation time step (s)")
##        plt.ylabel("Proportion of cars")
##        plt.title("Routing, AP=" + str(pSmart)+", Router="+router)
##        plt.legend(legend)
##        #plt.show()
##        plt.savefig("Plots/NEW/Routing,AP=" + str(pSmart)+",Router="+router+".png")

def updateTimes(d_in, d_out, step):
    global nDone, totalTime, avgTravelTime
    for d in d_in:
        ids = traci.inductionloop.getLastStepVehicleIDs(d)
        for id in ids:
            entryTimes[id] = step
    for d in d_out:
        ids = traci.inductionloop.getLastStepVehicleIDs(d)
        for id in ids:
            if id in entryTimes:
                nDone += 1
                totalTime += step - entryTimes[id]
                avgTravelTime = totalTime/nDone
    

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
    if not detector in routerInfo:
        routerInfo[detector] = dict()
        routerInfo[detector]["ntotal"] = 0
        routerInfo[detector]["nsmart"] = 0
        routerInfo[detector]["nchanged"] = 0
        for route in routes:
            routerInfo[detector][route] = 0
    
    ids = traci.inductionloop.getLastStepVehicleIDs(detector)
    for i in range(len(ids)):
        #If we haven't decided whether to route it or not, decide now
        if not ids[i] in isSmart:
            isSmart[ids[i]] = random.random() < pSmart

        #If we're routing it, and we're recomputing routes, recompute
        if isSmart[ids[i]]:
            traci.vehicle.setColor(ids[i], [0, 255, 0]) #Green = we're routing
            if detector == "RerouterL" or detector == "RerouterR":
                traci.vehicle.setColor(ids[i], [0, 255, 255]) #Blue = from side
            if rerouteAuto:
                route = getShortestRoute(routes, ids[i])

                #Update plotting info
                routerInfo[detector]["ntotal"] += 1
                routerInfo[detector]["nsmart"] += 1
                routerInfo[detector][route] += 1
                if not traci.vehicle.getRouteID(ids[i]) == route:
                    routerInfo[detector]["nchanged"] += 1
                    
                traci.vehicle.setRouteID(ids[i], route)
            continue
        #If we're not routing it, randomly pick a route
        traci.vehicle.setColor(ids[i], [255, 0, 0]) #Red = random routing
        if detector == "RerouterL" or detector == "RerouterR":
                traci.vehicle.setColor(ids[i], [255, 0, 255]) #Blue = from side
        r = random.random()
        nroutes = len(routes)
        for j in range(nroutes):
            #if r < 1.0/nroutes:

            #This steals 90% of the probability from the last route and gives it to the first route
            #So the first decision point routes 72.5% left, 27.5% right
            #Second decision point routes 90% mid, 10% outside
            if r < 1.9/nroutes:
                route = routes[j]

                #Update plotting info
                if rerouteAuto:
                    routerInfo[detector]["ntotal"] += 1
                    routerInfo[detector][route] += 1
                
                traci.vehicle.setRouteID(ids[i], route)
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
