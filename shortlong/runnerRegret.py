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
import matplotlib.pyplot as plt
import numpy as np

# we need to import python modules from the $SUMO_HOME/tools directory
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

from sumolib import checkBinary  # noqa
import traci  # noqa


def runRegret(regrets):
    traci.load(['-c', 'shortlong.sumocfg', "--additional-files", "additional.xml",
                "--start"])
    """execute the TraCI control loop"""
    step = 0
    # we start with phase 2 where EW has green
    #traci.trafficlight.setPhase("0", 2)
    data = dict()
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        #Get travel times
        for v in traci.vehicle.getIDList():
            if not v in data:
                data[v] = [step, -1, 2]
        for v in data:
            if not v in traci.vehicle.getIDList() and data[v][1] == -1:
                data[v][1] = step

        ids = traci.multientryexit.getLastStepVehicleIDs("Rerouter")
        p = regrets[0][0]/np.sum(regrets[0])
        #p = 0.9
        for i in range(len(ids)):
            if random.random() < p:
                traci.vehicle.setRouteID(ids[i], "route_0")
                data[ids[i]][2] = 0
            else:
                traci.vehicle.setRouteID(ids[i], "route_1")
                data[ids[i]][2] = 1
        step += 1

    #traci.close()
    sys.stdout.flush()
    #Need a regret computation
    t = np.zeros([len(data[v])-2])
    nt = np.zeros([len(data[v])-2])
    tc = np.zeros([len(data[v])-2, 4])
    ntc = np.zeros([len(data[v])-2, 4])
    #Regret estimates: Know the total time from this policy
    #Estimate total time from pure strategy as average time from cars from that choice?
    #Going to make a list of categorical variables to track decisions made
    #Using 2 as "vehicle never encountered this choice"

    #So then tc is a 2D array? As are regrets?

    #Also full average time
    fullt = 0
    fullnt = 0
    for v in data:
        vt = data[v][1] - data[v][0] #Time taken by vehicle currently being processed
        fullt += vt
        fullnt += 1
        for decision in range(len(data[v])-2):
            tc[decision][data[v][decision+2]] += vt
            ntc[decision][data[v][decision+2]] += 1
            if data[v][decision+2] != 2:
                t[decision] += vt
                nt[decision] += 1
                
    for i in range(len(regrets)): #Decision points
        for j in range(len(regrets[i])): #Actions
            #Regrets should be a 1D vector, same len as tc
            regrets[i][j] += t[i]/nt[i] - tc[i][j]/ntc[i][j]
            if regrets[i][j] < 1: #1 to make sure there's always some exploration
                regrets[i][j] = 1
    print(tc[0][0]/ntc[0][0])
    print(ntc[0][0])
    print(tc[0][1]/ntc[0][1])
    print(ntc[0][1])
    return regrets, fullt/fullnt

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

    
    regrets = np.ones([1, 2])
    regrets[0][1] = 10
    r0 = []
    r1 = []
    p = []
    t = []
    traci.start([sumoBinary, "-c", "shortlong.sumocfg",
                             "--tripinfo-output", "tripinfo.xml",
                             "--additional-files", "additional.xml",
                             "--start"])
    while True:
        print(regrets)
        r0.append(regrets[0][0])
        r1.append(regrets[0][1])
        p.append(regrets[0][0]/np.sum(regrets[0]))

        regrets, avgtime = runRegret(regrets)
        t.append(avgtime)

        ##Plot regrets
        plt.figure()
        plt.plot(r0)
        plt.plot(r1)
        plt.xlabel("# Iterations")
        plt.ylabel("Regret")
        plt.title("Regrets")
        plt.legend(["Regret (left turn)", "Regret (right turn)"])
        #plt.show()
        plt.savefig("Regrets.png")
        
        #Plot probabilities
        color='tab:blue'
        fig, ax1 = plt.subplots()
        ax1.plot(p, color=color)
        ax1.set_xlabel("# Iterations")
        ax1.set_ylabel("Probability of left turn", color=color)
        plt.title("Probability and Travel Time")
        color='tab:red'
        ax2 = ax1.twinx()
        ax2.plot(t, color=color)
        ax2.set_ylabel("Average travel time", color=color)
        fig.tight_layout()
        plt.savefig("Probability.png")
        
        #traci.close()
