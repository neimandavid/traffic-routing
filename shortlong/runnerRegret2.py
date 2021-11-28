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


def runRegret(regrets, p):
    traci.load(['-c', 'shortlong.sumocfg', "--additional-files", "additional.xml",
                "--xml-validation", "never","--start"])
    """execute the TraCI control loop"""
    step = 0
    # we start with phase 2 where EW has green
    #traci.trafficlight.setPhase("0", 2)
    data = dict()

    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        #Get travel times
        for v in traci.simulation.getDepartedIDList():
            if not v in data:
                data[v] = dict()
                data[v]["start"] = step
        for v in traci.simulation.getArrivedIDList():
            if not v in traci.vehicle.getIDList() and not "stop" in data[v]:
                data[v]["stop"] = step
                data[v]["time"] = step - data[v]["start"]
        reroute("Rerouter", ["route_0", "route_1"], p, data)
        step += 1
    
    #Need a regret computation
    tc = dict()
    ntc = dict()
    #Regret estimates: Know the total time from this policy
    #Estimate total time from pure strategy as average time from cars from that choice?
    #Going to make a list of categorical variables to track decisions made
    #Using 2 as "vehicle never encountered this choice"

    #So then tc is a 2D array? As are regrets?

    #Also full average time
    fullt = 0
    fullnt = 0
    for v in data:
        vt = data[v]["time"] #Time taken by vehicle currently being processed
        fullt += vt
        fullnt += 1
        for detector in data[v]:
            if detector == "start" or detector == "stop" or detector == "time":
                continue
            if not detector in tc:
                tc[detector] = np.zeros([2])
                ntc[detector] = np.zeros([2])
            tc[detector][data[v][detector]] += vt
            ntc[detector][data[v][detector]] += 1

    for v in data:
        for detector in data[v]:
            if detector == "start" or detector == "stop" or detector == "time":
                continue
            for j in range(2):
                #Regrets should be a 1D vector, same len as tc
                regrets[detector][j] += data[v]["time"] - tc[detector][j]/ntc[detector][j]
                if regrets[detector][j] < 1000: #1 to make sure there's always some exploration
                    regrets[detector][j] = 1000


    return regrets, fullt/fullnt

def reroute(detector, routes, p, data):
    ids = traci.multientryexit.getLastStepVehicleIDs(detector)

    for i in range(len(ids)):
        if random.random() < p[detector]:
            traci.vehicle.setRouteID(ids[i], routes[0])
            data[ids[i]][detector] = 0
        else:
            traci.vehicle.setRouteID(ids[i], routes[1])
            data[ids[i]][detector] = 1

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

    
    regrets = dict()
    regrets["Rerouter"] = np.ones([2])
    #regrets["Rerouter"][1] = 10
    r0 = []
    r1 = []
    p = []
    psum = 0
    nps = 0
    pavg = []
    t = []
    traci.start([sumoBinary, "-c", "shortlong.sumocfg",
                             "--tripinfo-output", "tripinfo.xml",
                             "--additional-files", "additional.xml",
                             "--xml-validation", "never",
                             "--log", "LOGFILE", 
                             "--start"])
    while True:
        print(regrets)
        r0.append(regrets["Rerouter"][0])
        r1.append(regrets["Rerouter"][1])
        newp = regrets["Rerouter"][0]/(np.sum(regrets["Rerouter"]))
        p.append(newp)
        psum += newp
        nps += 1
        pavg.append(psum/nps)

        pdict = dict()
        pdict["Rerouter"] = newp #psum/nps #newp
        regrets, avgtime = runRegret(regrets, pdict)
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
        plt.close()
        
        #Plot probabilities
        color='tab:blue'
        fig, ax1 = plt.subplots()
        ax1.plot(p, color=color)
        ax1.plot(pavg, color=color)
        ax1.set_xlabel("# Iterations")
        ax1.set_ylabel("Probability of left turn", color=color)
        plt.title("Probability and Travel Time")
        color='tab:red'
        ax2 = ax1.twinx()
        ax2.plot(t, color=color)
        ax2.set_ylabel("Average travel time", color=color)
        fig.tight_layout()
        plt.savefig("Probability.png")
        plt.close()
        
        #traci.close()
