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


def generate_routefile():
    random.seed(42)  # make tests reproducible
    N = 3600  # number of time steps
    # demand per second from different directions
    pWE = 1. / 10
    pEW = 1. / 11
    pNS = 1. / 30
    with open("data/cross.rou.xml", "w") as routes:
        print("""<routes>
        <vType id="typeWE" accel="0.8" decel="4.5" sigma="0.5" length="5" minGap="2.5" maxSpeed="16.67" \
guiShape="passenger"/>
        <vType id="typeNS" accel="0.8" decel="4.5" sigma="0.5" length="7" minGap="3" maxSpeed="25" guiShape="bus"/>

        <route id="right" edges="51o 1i 2o 52i" />
        <route id="left" edges="52o 2i 1o 51i" />
        <route id="down" edges="54o 4i 3o 53i" />""", file=routes)
        vehNr = 0
        for i in range(N):
            if random.uniform(0, 1) < pWE:
                print('    <vehicle id="right_%i" type="typeWE" route="right" depart="%i" />' % (
                    vehNr, i), file=routes)
                vehNr += 1
            if random.uniform(0, 1) < pEW:
                print('    <vehicle id="left_%i" type="typeWE" route="left" depart="%i" />' % (
                    vehNr, i), file=routes)
                vehNr += 1
            if random.uniform(0, 1) < pNS:
                print('    <vehicle id="down_%i" type="typeNS" route="down" depart="%i" color="1,0,0"/>' % (
                    vehNr, i), file=routes)
                vehNr += 1
        print("</routes>", file=routes)

# The program looks like this
#    <tlLogic id="0" type="static" programID="0" offset="0">
# the locations of the tls are      NESW
#        <phase duration="31" state="GrGr"/>
#        <phase duration="6"  state="yryr"/>
#        <phase duration="31" state="rGrG"/>
#        <phase duration="6"  state="ryry"/>
#    </tlLogic>

costs = dict()
def run():
    """execute the TraCI control loop"""
    step = 0
    # we start with phase 2 where EW has green
    #traci.trafficlight.setPhase("0", 2)
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        updateEdgeCosts()
        #if traci.trafficlight.getPhase("0") == 2:
            # we are not already switching
            #if traci.inductionloop.getLastStepVehicleNumber("0") > 0:
                # there is a vehicle from the north, switch
                #traci.trafficlight.setPhase("0", 3)
            #else:
                # otherwise try to keep green for EW
                #traci.trafficlight.setPhase("0", 2)
        #print() prints to console, not bottom of SUMO GUI
        #print("Detector data")
        #print(traci.multientryexit.getLastStepMeanSpeed("Detector"))
        #print(traci.multientryexit.getLastStepVehicleNumber("Detector"))

        #Smarter rerouting
        ids = traci.multientryexit.getLastStepVehicleIDs("RerouterS")
        for i in range(len(ids)):
            traci.vehicle.setRouteID(ids[i], getShortestRoute(["SLL0", "SLR0", "SRL0", "SRR0"]))

        ids = traci.multientryexit.getLastStepVehicleIDs("RerouterSL")
        for i in range(len(ids)):
            traci.vehicle.setRouteID(ids[i], getShortestRoute(["SLL", "SLR"]))

        ids = traci.multientryexit.getLastStepVehicleIDs("RerouterL")
        for i in range(len(ids)):
            traci.vehicle.setRouteID(ids[i], getShortestRoute(["LL", "LR"]))

        ids = traci.multientryexit.getLastStepVehicleIDs("RerouterSR")
        for i in range(len(ids)):
            traci.vehicle.setRouteID(ids[i], getShortestRoute(["SRL", "SRR"]))

        ids = traci.multientryexit.getLastStepVehicleIDs("RerouterR")
        for i in range(len(ids)):
            traci.vehicle.setRouteID(ids[i], getShortestRoute(["RL", "RR"]))

        step += 1
##        print("Estimating LR length with time/speed")
##        if(traci.edge.getLastStepMeanSpeed("LR") > 0):
##            print(traci.edge.getTraveltime("LR")*traci.edge.getLastStepMeanSpeed("LR"))
##        else:
##            print("Edge LR stopped?")
##        print("Actual LR length")
##        print(traci.lane.getLength("LR_0"))
    traci.close()
    sys.stdout.flush()

def updateEdgeCosts():
    edges = traci.edge.getIDList()
    w = 0 #Weight of previous cost value. Bigger = longer averaging
    #Weighted average seems bad, actually; it slows response to changes
    #0 just uses the current value
    #Outside +/-1 and a weighted average goes unstable.
    #Very negative and the sign flips (and magnitude grows) anytime anything changes.
    #Ironically, that's basically a coinflip, and does about as well as random
    for edge in edges:
        newcost = 0
        #newcost += traci.edge.getTraveltime(edge)
        newcost += 10*traci.edge.getLastStepVehicleNumber(edge)
        #newcost -= traci.edge.getLastStepMeanSpeed(edge)
            
        if edge in costs:
            costs[edge] = w*costs[edge] + (1-w)*newcost
        else:
            costs[edge] = newcost
            
def getShortestRoute(routes):
    route = "none"
    cost = inf
    for i in range(len(routes)):
        newroute = routes[i];
        newcost = 0;
        edges = traci.route.getEdges(newroute);
        for j in range(len(edges)):
            #newcost += traci.edge.getTraveltime(edges[j])
            #newcost += 10*traci.edge.getLastStepVehicleNumber(edges[j])
            #newcost -= traci.edge.getLastStepMeanSpeed(edges[j])/len(edges)
            newcost += costs[edges[j]] #Weighted average
        if newcost < cost:
            cost = newcost;
            route = newroute;
    return route

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
                             "--additional-files", "additional.xml"])
    run()
