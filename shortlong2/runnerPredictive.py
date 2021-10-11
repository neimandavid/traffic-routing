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


costs = dict()

preds = dict()
#To hold: Vehicle ID, arrival time, departure time
#Sorted by arrival time (note: arrival->departure is monotone)

knownvehicles = []
gap = 3 #Seconds between vehicles

def run():
    """execute the TraCI control loop"""
    step = 0
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        step+=1
        
        updateEdgeCosts(step)
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

    traci.close()
    sys.stdout.flush()

def updateEdgeCosts(step):
    vehicles = traci.vehicle.getIDList()
    for v in vehicles:
        if not v in knownvehicles:
            knownvehicles.append(v)
            #And add its route to the prediction
            tempedge = traci.vehicle.getRoute(v)[0]
            ind = addVehicle(tempedge, v, step)
            for i in range(1, len(traci.vehicle.getRoute(v))):
                #And iteratively add to future edges in order
                prevedge = tempedge
                tempedge = traci.vehicle.getRoute(v)[i]
                arrivalTime = preds[prevedge][ind][2]
                ind = addVehicle(tempedge, v, arrivalTime)
    
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

#Need to add, remove, and shift vehicles. Can maybe do shift as add+remove?

#Find ind in 2D array arr such that arr[ind][i] < value and arr[ind][i+1] >= value
#This assumes arr is sorted by values in index i, and just does binary search
def findIndexBefore(arr, i, value):
    #Implementing binary search
    m = 0
    n = len(arr)-1
    while m <= n:
        ind = m + (n-m)//2
        v = arr[ind][i]
        if v < value:
            #Check back half of array
            m = ind+1
        else:
            n = ind-1
    return n+1

def addVehicle(edge, vehicle, arrivalTime, oldArrivalTime=None):
    #If already here, should delete it and start by updating whatever was behind it
    #Stop that round of updates before we reinsert, then continue after reinsert
    #Is that just: Insert new, delete old, update from old (was +1), update from new(+1?)
    #How to find old? Do I know what the arrival time was??
    
    lane = edge + "_0"
    fftime = traci.lane.getLength(lane) / traci.lane.getMaxSpeed(lane)
    ind = findIndexBefore(preds[edge], 1, arrivalTime)
    
    if ind == 0: #Happens for empty list or nobody before you
        #Add to start; you're moving at freeflow speed
        preds[edge].insert(ind, [vehicle, arrivalTime, arrivalTime+fftime])
    else:
        #You leave no later than the previous car + gap
        preds[edge].insert(ind, [vehicle, arrivalTime, max(arrivalTime+fftime, preds[edge][ind-1][2]+gap)])

    #Delete old instance if oldArrivalTime was given
    if not oldArrivalTime == None:
        delind = findIndexBefore(preds[edge], 1, oldArrivalTime)
        preds[edge].pop(delind)
        if ind > delind:
            ind -= 1 #Adjust for deleted value
        #Update stuff starting after the deletion if possible, else the next car in merging lanes
        if delind + 1 < len(preds[edge]):
            #There's a following car that may be affected
            print("Recomputing next vehicle after delete")
            recomputeDeparture(edge, delind+1)
        else:
            print("TODO: Update merging lanes after delete, but whatever")

    #Now update the next car after the insertion, if there is one, else the next car in other lanes
    if ind + 1 < len(preds[edge]):
        #There's a following car that may be affected
        print("Recomputing next vehicle")
        recomputeDeparture(edge, ind+1)
    else:
        print("TODO: Update merging lanes, but whatever")
    #print(preds[edge])
    return ind

def recomputeDeparture(edge, ind):
    lane = edge + "_0"
    fftime = traci.lane.getLength(lane) / traci.lane.getMaxSpeed(lane)

    oldDeparture = preds[edge][ind][2]
    if ind > 0 and oldDeparture < preds[edge][ind-1][2] + gap:
        newDeparture = preds[edge][ind-1][2] + gap
        preds[edge][ind][2] = newDeparture
    else:
        if oldDeparture > preds[edge][ind][1] + fftime:
            newDeparture = preds[edge][ind][1] + fftime
            preds[edge][ind][2] = newDeparture
        else:
            return;

    #Propagate updated departure forward
    print("TODO: Propagate undelayed departure forward")
    #Find edge in route (so we know what edge comes next)
    #I'm not certain this terminates...
    vehicle = preds[edge][ind][0]
    route = traci.vehicle.getRoute(vehicle)
    print(route)
    print(edge)
    print("If this shows up before an error throws, I probably need to remove rerouted vehicles from their old routes!!!")
    routeedgeind = route.index(edge)
    #Adjust arrival time on next edge
    if routeedgeind + 1 < len(route):
        nextrouteedge = route[routeedgeind+1]
        addVehicle(nextrouteedge, vehicle, newDeparture, oldArrivalTime=oldDeparture)
    
    #Update car behind you on current edge
    if ind + 1 < len(preds[edge]):
        recomputeDeparture(edge, ind+1)
    else:
        print("TODO: Update merging lanes...")
            
            
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
    for edge in traci.edge.getIDList():
        preds[edge] = []
    run()
