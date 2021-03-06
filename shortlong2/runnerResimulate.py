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

# we need to import python modules from the $SUMO_HOME/tools directory
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

from sumolib import checkBinary  # noqa
import traci  # noqa

def run():
    """execute the TraCI control loop"""
    step = 0
    dontBreakEverything()
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        
        step+=1
        
        #Smarter rerouting
        ids = traci.inductionloop.getLastStepVehicleIDs("RerouterS0")
        for i in range(len(ids)):
            traci.vehicle.setRouteID(ids[i], getShortestRoute(["SLL0", "SLR0", "SRL0", "SRR0"], ids[i]))

        ids = traci.inductionloop.getLastStepVehicleIDs("RerouterS1")
        for i in range(len(ids)):
            traci.vehicle.setRouteID(ids[i], getShortestRoute(["SLL0", "SLR0", "SRL0", "SRR0"], ids[i]))

        ids = traci.inductionloop.getLastStepVehicleIDs("RerouterSL")
        for i in range(len(ids)):
            traci.vehicle.setRouteID(ids[i], getShortestRoute(["SLL", "SLR"], ids[i]))

        ids = traci.inductionloop.getLastStepVehicleIDs("RerouterL")
        for i in range(len(ids)):
            traci.vehicle.setRouteID(ids[i], getShortestRoute(["LL", "LR"], ids[i]))

        ids = traci.inductionloop.getLastStepVehicleIDs("RerouterSR")
        for i in range(len(ids)):
            traci.vehicle.setRouteID(ids[i], getShortestRoute(["SRL", "SRR"], ids[i]))

        ids = traci.inductionloop.getLastStepVehicleIDs("RerouterR")
        for i in range(len(ids)):
            traci.vehicle.setRouteID(ids[i], getShortestRoute(["RL", "RR"], ids[i]))

    traci.close()
    sys.stdout.flush()

#Magically make the vehicle lists stop deleting themselves somehow???
def dontBreakEverything():
    #traci.simulation.saveState("teststate.xml")
    traci.switch("test")
    #traci.simulation.loadState("teststate.xml")
    traci.simulationStep()
    traci.switch("main")

def getShortestRoute(routes, vehicle):
    #Copy state from main sim to test sim
    temp = traci.trafficlight.getPhase("gneJ5")
    traci.simulation.saveState("teststate.xml")
    #TODO: saveState apparently doesn't save traffic light states despite what the docs say
    #So save all the traffic light states and copy them over
    lightStates = dict()
    for light in traci.trafficlight.getIDList():
        lightStates[light] = [traci.trafficlight.getPhase(light), traci.trafficlight.getPhaseDuration(light)]
        #Why do the built-in functions have such terrible names?!
        lightStates[light][1] = traci.trafficlight.getNextSwitch(light) - traci.simulation.getTime()

    traci.switch("test")
    traci.simulationStep()
    
    bestroute = "None"
    besttime = float('inf')
    for route in routes:
        traci.simulation.loadState("teststate.xml")
        for light in traci.trafficlight.getIDList():
            traci.trafficlight.setPhase(light, lightStates[light][0])
            traci.trafficlight.setPhaseDuration(light, lightStates[light][1])
        #assert(temp == traci.trafficlight.getPhase("gneJ5"))
        traci.vehicle.setRouteID(vehicle, route)
        t = 0
        while(vehicle in traci.vehicle.getIDList()):
            traci.simulationStep()
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
