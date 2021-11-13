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


def runProb(p):
    traci.load(['-c', 'shortlong.sumocfg', "--additional-files", "additional.xml",
                "--start"])
    """execute the TraCI control loop"""
    step = 0
    # we start with phase 2 where EW has green
    #traci.trafficlight.setPhase("0", 2)
    data = dict()
    regrets = np.zeros([1, 2])

    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        #Get travel times
        #Newly spawned vehicle
        ids = traci.inductionloop.getLastStepVehicleIDs("Detector0")
        for v in ids:
            if True:
                data[v] = [step, -1, 2, 1]
        ids = traci.inductionloop.getLastStepVehicleIDs("Detector1")
        for v in ids:
            if True:
                data[v] = [step, -1, 2, 1]
                
        #Newly exited vehicle
        eta = 1.0/1000000
        ids = traci.inductionloop.getLastStepVehicleIDs("DetectorE0")
        for v in ids:
            data[v][1] = step
            regrets[0][data[v][2]] -= eta*(data[v][1] - data[v][0])/data[v][3]
        ids = traci.inductionloop.getLastStepVehicleIDs("DetectorE1")
        for v in ids:
            data[v][1] = step
            regrets[0][data[v][2]] -= eta*(data[v][1] - data[v][0])/data[v][3]

        #Routing
        ids = traci.multientryexit.getLastStepVehicleIDs("Rerouter")
        #p = np.exp(regrets[0][0])/np.sum(np.exp(regrets[0]))
        #ps.append(p)
        for i in range(len(ids)):
            if random.random() < p:
                traci.vehicle.setRouteID(ids[i], "route_0")
                data[ids[i]][2] = 0
                data[ids[i]][3] = p
            else:
                traci.vehicle.setRouteID(ids[i], "route_1")
                data[ids[i]][2] = 1
                data[ids[i]][3] = 1-p
        step += 1

    #traci.close()
    sys.stdout.flush()
    #Need a regret computation
    t = np.zeros([len(data[v])-2])
    nt = np.zeros([len(data[v])-2])
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

##    plt.figure()
##    plt.plot(ps)
##    plt.xlabel("Timestep")
##    plt.ylabel("P(left)")
##    plt.title("MWU, eta=" + str(eta))
##    plt.savefig("MWU, eta=" + str(eta) + ".png")
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

    

    ps = []
    ts = []
    traci.start([sumoBinary, "-c", "shortlong.sumocfg",
                             "--tripinfo-output", "tripinfo.xml",
                             "--additional-files", "additional.xml",
                             "--start"])


    for i in range(20):
        p = i/20.0
        ps.append(p)
        regrets, avgtime = runProb(p)
        ts.append(avgtime)
    plt.figure()
    plt.plot(ps, ts)
    plt.xlabel("P(left)")
    plt.ylabel("Avg. Time (s)")
    plt.title("Average time vs. P(left)")
    plt.savefig("PSweep.png")
        
    traci.close()
