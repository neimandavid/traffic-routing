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


def run():
    #netfile is the filepath to the network file, so we can call sumolib to get successors
    #rerouters is the list of induction loops on edges with multiple successor edges
    #We want to reroute all the cars that passed some induction loop in rerouters using A*
    
    """execute the TraCI control loop"""
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep() #Tell the simulator to simulate the next time step

        reroute() #Reroute cars (including simulate-ahead cars)
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
def reroute():
    for v in traci.vehicle.getIDList():
        #print(traci.vehicle.getLanePosition(v) - (traci.lane.getLength(traci.vehicle.getRoadID(v)+"_0") - 50) )
        print(traci.vehicle.getLanePosition(v))
        if traci.vehicle.getLanePosition(v) > traci.lane.getLength(traci.vehicle.getRoadID(v)+"_0") - 50:
            if random.random() < 0.5:
                traci.vehicle.setRoute(v, ["start", "down"])
            else:
                traci.vehicle.setRoute(v, ["start", "up"])

#Tell all the detectors to reroute the cars they've seen
def rerouteIL():
    print(traci.inductionloop.getIDList())
    vehicles = traci.inductionloop.getLastStepVehicleIDs("IL_start")
    for v in vehicles:
        if random.random() < 0.5:
            traci.vehicle.setRoute(v, ["start", "down"])
        else:
            traci.vehicle.setRoute(v, ["start", "up"])

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

    #NOTE: Script name is zeroth arg
    sumoconfig = sys.argv[2]
    netfile = sys.argv[1]
    print("MAX_EDGE_SPEED 2.0: {}".format(max_edge_speed))

    # this is the normal way of using traci. sumo is started as a
    # subprocess and then the python script connects and runs
    traci.start([sumoBinary, "-c", sumoconfig,
                             "--additional-files", "additional.xml",
                             "--mesosim", "true",
                             "--log", "LOGFILE", "--xml-validation", "never"], label="main")

    run()
    traci.close()
