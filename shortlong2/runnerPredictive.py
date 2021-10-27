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
import numpy as np

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
data = dict()
#Not sorted. Keys are edges, then vehicles.

knownvehicles = []
gap = 2 #Seconds between vehicles

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
        ids = traci.inductionloop.getLastStepVehicleIDs("RerouterS0")
        for i in range(len(ids)):
            reroute(ids[i], getShortestRoute(["SLL0", "SLR0", "SRL0", "SRR0"]))
            #reroute(ids[i], getShortestRoute(["SLR0"]))

        ids = traci.inductionloop.getLastStepVehicleIDs("RerouterS1")
        for i in range(len(ids)):
            reroute(ids[i], getShortestRoute(["SLL0", "SLR0", "SRL0", "SRR0"]))
            #reroute(ids[i], getShortestRoute(["SLR0"]))

        ids = traci.inductionloop.getLastStepVehicleIDs("RerouterSL")
        for i in range(len(ids)):
            reroute(ids[i], getShortestRoute(["SLL", "SLR"]))
            #reroute(ids[i], getShortestRoute(["SLL"]))

        ids = traci.inductionloop.getLastStepVehicleIDs("RerouterL")
        for i in range(len(ids)):
            reroute(ids[i], getShortestRoute(["LL", "LR"]))
            #reroute(ids[i], getShortestRoute(["LL"]))

        ids = traci.inductionloop.getLastStepVehicleIDs("RerouterSR")
        for i in range(len(ids)):
            reroute(ids[i], getShortestRoute(["SRL", "SRR"]))

        ids = traci.inductionloop.getLastStepVehicleIDs("RerouterR")
        for i in range(len(ids)):
            reroute(ids[i], getShortestRoute(["RL", "RR"]))

        #print(preds["goal"])
    traci.close()
    sys.stdout.flush()

def updateEdgeCosts(step):
    vehicles = traci.vehicle.getIDList()
    
    #Add new vehicles
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
                
        #Update data (when we actually arrived at edges)
        edge = traci.vehicle.getRoadID(v)
        if not edge in data and edge in preds:
            data[edge] = dict()
        if not v in data[edge]:# and edge in preds and findValue(preds[edge], 0, v) != None:
            #Just got onto a new edge
            data[edge][v] = [v, step, -1]
            edgeind = traci.vehicle.getRouteIndex(v)
            if edgeind > 0:
                prevedge = traci.vehicle.getRoute(v)[edgeind-1]
                #Without the if, I'm getting errors here for Lflow.7 at 91s, which is weird...
                #Supposedly it's due to teleporting after collision with Sflow.0?
                if prevedge in data and v in data[prevedge]:
                    data[prevedge][v][2] = step
            #Debug
##            print("Arrival on new edge")
##            print(v)
##            print(edge)
##            print("Predicted arrival time:")
##            ind = findValue(preds[edge], 0, v)
##            print(preds[edge][ind][1])
##            print("Actual arrival time:")
##            print(data[edge][v][1])
    for v in knownvehicles:
        if not v in vehicles:
            knownvehicles.remove(v)
            print(v + " left the map ")
            ind = findValue(preds["goal"], 0, v)
            #Can't print route, vehicle doesn't exist anymore. Grr...
            #print(preds["goal"])
            print("Estimated time:")
            print(preds["goal"][ind][2])
            print("Real time:")
            print(step)
            #Record actual departure time
            data["goal"][v][2] = step
    
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

#Find ind in 2D array arr such that arr[ind][i] < value and arr[ind+1][i] >= value
#This assumes arr is sorted by values in index i, and just does binary search
def findIndexBefore(arr, i, value):
    #Implementing binary search
    #Invariants: m <= index before, n >= index before
    m = -1
    n = len(arr)-1
    while m < n:
        ind = m + (n-m+1)//2
        v = arr[ind][i]
        if v < value:
            #Check back half of array
            m = ind
        else:
            n = ind-1
    assert(n < 0 or arr[n][i] < value)
    assert(n+1 >= len(arr) or arr[n+1][i] >= value)
    return n


def findIndexAfter(arr, i, value):
    #Implementing binary search
    #Invariants: m <= index after, n >= index after
    m = 0
    n = len(arr)
    while m < n:
        ind = m + (n-m)//2
        v = arr[ind][i]
        if v <= value:
            #Check back half of array
            m = ind+1
        else:
            n = ind
    assert(n-1 < 0 or arr[n-1][i] <= value)
    assert(n >= len(arr) or arr[n][i] > value)
    return n

def findValue(arr, i, value):
    for j in range(len(arr)):
        if arr[j][i] == value:
            return j
    return None

def addVehicle(edge, vehicle, arrivalTime):    
    lane = edge + "_0"
    fftimefudge = 1.1
    fftime = traci.lane.getLength(lane) / traci.lane.getMaxSpeed(lane) * fftimefudge
    ind = findIndexBefore(preds[edge], 1, arrivalTime)+1

    backind = findIndexAfter(preds[edge], 1, arrivalTime)
    
    if ind == 0: #Happens for empty list or nobody before you
        #Add to start; you're moving at freeflow speed
        preds[edge].insert(ind, [vehicle, arrivalTime, arrivalTime+fftime])
    else:
        #You leave no later than the previous car + gap
        departureTime = max(arrivalTime+fftime, preds[edge][ind-1][2]+gap)
        preds[edge].insert(ind, [vehicle, arrivalTime, departureTime])
            
    #Now update the next car after the insertion, if there is one, else the next car in other lanes
    if ind + 1 < len(preds[edge]):
        #There's a following car that may be affected
        recomputeDeparture(edge, ind+1)
    #else:
        #print("TODO: Update merging lanes, but whatever")
    #print(preds[edge])
    return ind

def removeVehicle(edge, vehicle, oldArrivalTime):
    
    delind = findIndexBefore(preds[edge], 1, oldArrivalTime)

    while delind < len(preds[edge]):
        if preds[edge][delind][0] == vehicle and preds[edge][delind][1] == oldArrivalTime:
            break
        delind += 1
        if delind == len(preds[edge]) or preds[edge][delind][1] > oldArrivalTime:
            #print("Car to be removed doesn't exist?")
            return
    
    oldDepartureTime = preds[edge][delind][2]
    preds[edge].pop(delind)
    #Update stuff starting after the deletion if possible, else the next car in merging lanes
    if delind + 1 < len(preds[edge]):
        #There's a following car that may be affected
        recomputeDeparture(edge, delind+1)
    #else:
        #print("TODO: Update merging lanes after delete, but whatever")
    return oldDepartureTime

def recomputeDeparture(edge, ind):
    lane = edge + "_0"
    fftime = traci.lane.getLength(lane) / traci.lane.getMaxSpeed(lane)

    oldDeparture = preds[edge][ind][2]

    #DEBUG
##    vehicle = preds[edge][ind][0]
##    route = traci.vehicle.getRoute(vehicle)
##    if not route.index(edge)+1 == len(route):
##        nextedge = route[route.index(edge)+1]
##        nextind = findIndexBefore(preds[nextedge], 1, oldDeparture)+1
##        print(vehicle)
##        print(oldDeparture)
##        print(nextind)
##        print(preds[nextedge])
##        print(preds[nextedge][nextind][1])
##        assert(preds[nextedge][nextind][1] == oldDeparture)

    
    if ind > 0 and oldDeparture < preds[edge][ind-1][2] + gap/traci.edge.getLaneNumber(edge):
        newDeparture = preds[edge][ind-1][2] + gap/traci.edge.getLaneNumber(edge)
        preds[edge][ind][2] = newDeparture
    else:
        if oldDeparture > preds[edge][ind][1] + fftime:
            newDeparture = preds[edge][ind][1] + fftime
            preds[edge][ind][2] = newDeparture
        else:
            return;

    #Propagate updated departure forward
    #Find edge in route (so we know what edge comes next)
    #I'm not certain this terminates, but probably fine...
    vehicle = preds[edge][ind][0]
    #print(preds[edge])
    #print(edge)
    #print(vehicle)
    route = traci.vehicle.getRoute(vehicle)
    try:
        routeedgeind = route.index(edge)
        #Adjust arrival time on next edge
        if routeedgeind + 1 < len(route):
            nextrouteedge = route[routeedgeind+1]
            removeVehicle(nextrouteedge, vehicle, oldDeparture)
            addVehicle(nextrouteedge, vehicle, newDeparture)
    except ValueError:
        #Error: Couldn't find edge on route
        #This shouldn't be happening, but apparently does.
        #This car isn't supposed to be on this edge anymore!
        assert(True)
        #print("More stuff going wrong?")
        #removeVehicle(edge, vehicle, preds[edge][ind][1])
        
    #Update car behind you on current edge
    if ind + 1 < len(preds[edge]):
        recomputeDeparture(edge, ind+1)
    #else:
        #print("TODO: Update merging lanes...")
            
            
def getShortestRoute(routes):
    route = "none"
    cost = np.inf
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

def reroute(id, newrouteID):
    oldroute = traci.vehicle.getRoute(id) #List
    newroute = traci.route.getEdges(newrouteID)
    routeIndex = traci.vehicle.getRouteIndex(id)
    #NOTE: DO NOT change route before here!!!
    routechanged = False
    arrivalTime = 0
    lastedge = None

    #Actually set the desired new route
    traci.vehicle.setRouteID(id, newrouteID)
    #Vehicle should be at start of new route. (For this hard-coded set of routes)
    assert(traci.vehicle.getRouteIndex(id) == 0)
    
    for i in range(len(oldroute)-routeIndex):
        if oldroute[i+routeIndex] != newroute[i]:
            routechanged = True
            break
        lastedge = oldroute[i+routeIndex] #After break, this is the last shared edge on both routes

    if routechanged:
        #This is the only case where we have to do anything

        #Find index of car on previous edge
        for lastind in range(len(preds[lastedge])):
            if preds[lastedge][lastind][0] == id:
                break
            if lastind == len(preds[lastedge]) - 1:
                print("Didn't find car, but we were supposed to")
                assert(False)
        
        #Add to new route
        ind = lastind
        tempedge = lastedge
        for j in range(i, len(newroute)):
            #And iteratively add to future edges in order
            prevedge = tempedge
            tempedge = newroute[j]
            arrivalTime = preds[prevedge][ind][2]
            ind = addVehicle(tempedge, id, arrivalTime)
            
        #Remove from old route
        arrivalTime = preds[lastedge][lastind][2]
        tempedge = lastedge
        for j in range(i+routeIndex, len(oldroute)):
            #Remove from previous edges in order
            tempedge = oldroute[j]
            arrivalTime = removeVehicle(tempedge, id, arrivalTime)
            if arrivalTime == None:
                break
        


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
