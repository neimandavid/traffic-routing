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

def run():
    #netfile is the filepath to the network file, so we can call sumolib to get successors
    #rerouters is the list of induction loops on edges with multiple successor edges
    #We want to reroute all the cars that passed some induction loop in rerouters using A*
    
    """execute the TraCI control loop""" 
    carnum = 0
    traci.route.add("uproute", ["in", "up"])
    traci.route.add("downroute", ["in", "down"])
    while True:
        
        traci.simulationStep() #Tell the simulator to simulate the next time step
        carnum += 1
        traci.vehicle.add("car" + str(carnum), "uproute")
        carnum += 1
        traci.vehicle.add("car" + str(carnum), "downroute")


# this is the main entry point of this script
if __name__ == "__main__":


    # this is the normal way of using traci. sumo is started as a
    # subprocess and then the python script connects and runs
    traci.start([checkBinary('sumo-gui'), "-n", "test.net.xml",
                             "--log", "LOGFILE", "--xml-validation", "never"], label="main")
    run()
    traci.close()
