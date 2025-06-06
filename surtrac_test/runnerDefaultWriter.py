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
from collections import Counter #See https://www.geeksforgeeks.org/python-count-occurrences-element-list/

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
import xml.etree.ElementTree as ET

isSmart = dict(); #Store whether each vehicle does our routing or not
pSmart = 1.0; #Adoption probability

carsOnNetwork = [];
max_edge_speed = 0.0;
AStarCutoff = 200;

hmetadict = dict()

oldids = dict()
timedata = dict()
roadcarcounter = dict()

def run(netfile, rerouters, sumoconfig):

    (testnetfile, testroutefile) = readSumoCfg(sumoconfig)
    routefileparts = testroutefile.split(".")
    nonroutefileparts = testroutefile.split(".") #I'm scared of copy-by-location here...
    routefileparts[0] = sumoconfig.split(".")[0]+"_auto" #routefileparts[0]+"_auto"
    nonroutefileparts[0] = sumoconfig.split(".")[0]+"_noroutes_auto"
    routefilename = ""
    nonroutefilename = ""
    for i in range(len(routefileparts)):
        if i > 0:
            routefilename += "."
            nonroutefilename += "."
        nonroutefilename += nonroutefileparts[i]
        routefilename += routefileparts[i]
    
    configfilename = routefileparts[0]+".sumocfg"
    writeSumoCfg(configfilename, netfile, routefilename) #NOTE: We don't actually have the new routefile yet, but whatever...

    nonrouteconfigfilename = nonroutefileparts[0]+".sumocfg"
    writeSumoCfg(nonrouteconfigfilename, netfile, nonroutefilename) #NOTE: We don't actually have the new routefile yet, but whatever...

    tree = ET.parse(testroutefile)

    # get root element
    root = tree.getroot()

    intendedStartDict = dict()
    fromDict = dict()
    toDict = dict()
    routeDict = dict()
    # iterate news items
    for item in root.findall('./vehicle'):
        carname = item.attrib["id"]
        intendedStartDict[carname] = float(item.attrib["depart"])
        # print(item.attrib["id"])
        #BELOW IS BAD
        # fromDict[carname] = item.attrib["from"]
        # toDict[carname] = item.attrib["to"]
        #TODO: Set up fromDict and toDict to work with vehicle objects for non-route file

    for item in root.findall('./trip'):
        intendedStartDict[item.attrib["id"]] = float(item.attrib["depart"])
        fromDict[item.attrib["id"]] = item.attrib["from"]
        toDict[item.attrib["id"]] = item.attrib["to"]

    for item in root.findall('./flow'):
        dt = (float(item.attrib["end"]) - float(item.attrib["begin"])) / (float(item.attrib["number"]) - 1) #-1 because fencepost problem
        for ind in range(int(item.attrib["number"])):
            carname = item.attrib["id"] + "." + str(ind) #No +1 because 0-indexing everywhere
            intendedStartDict[carname] = float(item.attrib["begin"]) + dt*(ind) #No +1 because 0-indexing
            fromDict[carname] = item.attrib["from"]
            toDict[carname] = item.attrib["to"]

    #Sort intendedStartDict by depart time, because the output file needs to be in order
    #Code adapted from: https://www.freecodecamp.org/news/sort-dictionary-by-value-in-python/
    tempnotadict = sorted(intendedStartDict.items(), key=lambda x:x[1])
    intendedStartDict = dict(tempnotadict)

    #Print the non-route file now because we can
    with open(nonroutefilename, "w") as routefile:
        print("""<?xml version="1.0" encoding="UTF-8"?>""", file=routefile)
        print("""<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">
""", file=routefile)
        print("""<vType id="noVar" speedFactor="1.0" speedDev="0.0"/>""", file=routefile)
        for id in intendedStartDict:
                print("""<trip type="noVar" depart="%i" id="%s" from="%s" to="%s"/>""" % (intendedStartDict[id], id, fromDict[id], toDict[id]), file=routefile)
            
        #Close out non-route file
        print("""</routes>""", file=routefile)

    #Start printing the route file
    with open(routefilename, "w") as routefile:
        print("""<?xml version="1.0" encoding="UTF-8"?>""", file=routefile)
        print("""<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">
""", file=routefile)
        print("""<vType id="noVar" speedFactor="1.0" speedDev="0.0"/>""", file=routefile)
        transitiondata = dict()
        lanetransitiondata = dict()

        #Add psuedocounts everywhere...
        for lane in traci.lane.getIDList():
            if lane[0] == ":":
                #Skip internal edges (=edges for the inside of each intersection)
                continue
            links = traci.lane.getLinks(lane)
            if len(links) > 0:
                lanetransitiondata[lane] = []
                for link in links:
                    lanetransitiondata[lane].append(link[0])

        #netfile is the filepath to the network file, so we can call sumolib to get successors
        #rerouters is the list of induction loops on edges with multiple successor edges
        #We want to reroute all the cars that passed some induction loop in rerouters using A*
        
        """execute the TraCI control loop"""
        network = sumolib.net.readNet(netfile)
        startDict = dict()
        endDict = dict()
        locDict = dict()
        prevLaneDict = dict()
        laneDict = dict()
        leftDict = dict()

        tstart = time.time()
        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep() #Tell the simulator to simulate the next time step

            #Decide whether new vehicles use our routing
            for vehicle in traci.simulation.getDepartedIDList():
                isSmart[vehicle] = random.random() < pSmart
                timedata[vehicle] = [traci.simulation.getTime(), -1, -1]
            for vehicle in traci.simulation.getArrivedIDList():
                timedata[vehicle][1] = traci.simulation.getTime()
                #print("Actual minus expected:")
                #print( (timedata[vehicle][1]-timedata[vehicle][0]) - timedata[vehicle][2])

            carsOnNetwork.append(len(traci.vehicle.getIDList())) #Track number of cars on network (for plotting)
            
            t = traci.simulation.getTime()
            for id in traci.simulation.getDepartedIDList():
                startDict[id] = t
                locDict[id] = traci.vehicle.getRoadID(id)
                laneDict[id] = traci.vehicle.getLaneID(id)
                prevLaneDict[id] = None #Last lane you turned FROM
                leftDict[id] = 0

                #Repeaty stuff
                route = traci.vehicle.getRoute(id)
                routestring = ""
                for edgeind in range(len(route)):
                    routestring = routestring + " " + route[edgeind]
                    if edgeind > 0:
                        if route[edgeind-1] in transitiondata:
                            transitiondata[route[edgeind-1]].append(route[edgeind])
                        else:
                            transitiondata[route[edgeind-1]] = [route[edgeind]]
                routestring = routestring[1:] #Drop leading space
                routeDict[id] = routestring
                
            for id in traci.simulation.getArrivedIDList():
                endDict[id] = t
                locDict.pop(id)
                #Store the turn onto the exit road in lanetransitiondata
                if not id in traci.simulation.getEndingTeleportIDList():
                    if not (prevLaneDict[id] in lanetransitiondata): #Should always be true because psuedocount stuff, but apparently not???
                        print("Warning: prevLaneDict[id] isn't in lanetransitiondata, this shouldn't happen")
                        lanetransitiondata[prevLaneDict[id]] = []
                    if laneDict[id].split("_")[0] in getSuccessors(prevLaneDict[id].split("_")[0], network):
                        lanetransitiondata[prevLaneDict[id]].append(laneDict[id]) #Last lane you turned from to lane you're currently turning from
                    else:
                        print("We think we didn't teleport, but we ended up on a non-successor road?")

                laneDict.pop(id)
                prevLaneDict.pop(id)
            for id in locDict:
                road = traci.vehicle.getRoadID(id)
                if road != locDict[id] and road != "" and road[0] != ":":
                    #Track left turns
                    c0 = network.getEdge(locDict[id]).getFromNode().getCoord()
                    c1 = network.getEdge(locDict[id]).getToNode().getCoord()
                    theta0 = math.atan2(c1[1]-c0[1], c1[0]-c0[0])

                    #Confirm that the end of the previous edge is the start of the current edge
                    #Can fail for really short roads. Commenting the assert and hoping left turn counter is roughly correct
                    #assert(c1 == network.getEdge(traci.vehicle.getRoadID(id)).getFromNode().getCoord())
                    c2 = network.getEdge(road).getToNode().getCoord()
                    theta1 = math.atan2(c2[1]-c1[1], c2[0]-c1[0])

                    if (theta1-theta0+math.pi)%(2*math.pi)-math.pi > 0:
                        leftDict[id] += 1

                    locDict[id] = traci.vehicle.getRoadID(id)

                    #Track lane-by-lane turn data
                    if not id in traci.simulation.getEndingTeleportIDList():
                        if not prevLaneDict[id] in lanetransitiondata:
                            lanetransitiondata[prevLaneDict[id]] = []
                        if prevLaneDict[id] == None or laneDict[id].split("_")[0] in getSuccessors(prevLaneDict[id].split("_")[0], network):
                            lanetransitiondata[prevLaneDict[id]].append(laneDict[id]) #Last lane you turned from to lane you're currently turning from
                        else:
                            print("We think we didn't teleport, but we ended up on a non-successor road?")
                            
                    prevLaneDict[id] = laneDict[id]
                    
                    if not road in roadcarcounter:
                        roadcarcounter[road] = 0
                    roadcarcounter[road] += 1
                    
                # if traci.vehicle.getRoadID(id) == "":
                #     print(laneDict[id])
                if traci.vehicle.getRoadID(id) != "" and traci.vehicle.getLaneID(id)[0] != ":":
                    laneDict[id] = traci.vehicle.getLaneID(id) #Always keep this up to date

            if t%100 == 0 or not traci.simulation.getMinExpectedNumber() > 0:
                #After we're done simulating... 
                plt.figure()
                plt.plot(carsOnNetwork)
                plt.xlabel("Time (s)")
                plt.ylabel("Cars on Network")
                plt.title("Congestion, Adoption Prob=" + str(pSmart))
                #plt.show() #NOTE: Blocks code execution until you close the plot
                plt.savefig("Plots/Congestion, AP=" + str(pSmart)+".png")
                plt.close()

                avgTime = 0
                avgLefts = 0
                bestTime = inf
                worstTime = 0
                for id in endDict:
                    ttemp = endDict[id] - startDict[id]
                    avgTime += ttemp/len(endDict)
                    avgLefts += leftDict[id]/len(endDict)
                    if ttemp > worstTime:
                        worstTime = ttemp
                    if ttemp < bestTime:
                        bestTime = ttemp
                print("\nCurrent simulation time: %f" % t)
                print("Total run time: %f" % (time.time() - tstart))
                print("Average time in network: %f" % avgTime)
                print("Best time: %f" % bestTime)
                print("Worst time: %f" % worstTime)
                print("Average number of lefts: %f" % avgLefts)

        #Actually write the route file (couldn't do this before because cars might not have been inserted and those wouldn't have routes)
        #And route files need to be in departure time order, so just do everything at the end
        for id in intendedStartDict:
            print("""<route id="route_%s" edges="%s" />""" % (id, routeDict[id]), file=routefile)
            print("""<vehicle type="noVar" depart="%i" id="%s" route="route_%s" />""" % (intendedStartDict[id], id, id), file=routefile)
        
        #Simulation done, close out route file
        print("""</routes>""", file=routefile)

    turndata = dict()
    for road in transitiondata:
        turndata[road] = Counter(transitiondata[road])
        for nextroad in turndata[road]:
            turndata[road][nextroad] /= len(transitiondata[road])
    #print(turndata)
    with open("turndata_"+routefileparts[0]+".pickle", 'wb') as handle:
        pickle.dump(turndata, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Saved turn data")

    laneturndata = dict()
    for road in lanetransitiondata:
        laneturndata[road] = Counter(lanetransitiondata[road])
        for nextroad in laneturndata[road]:
            laneturndata[road][nextroad] /= len(lanetransitiondata[road])
    #print(laneturndata)
    with open("Lturndata_"+routefileparts[0]+".pickle", 'wb') as handle:
        pickle.dump(laneturndata, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open("Lturndata_"+nonroutefileparts[0]+".pickle", 'wb') as handle:
        pickle.dump(laneturndata, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Saved lane turn data")

    return roadcarcounter

def readSumoCfg(sumocfg):
    netfile = ""
    roufile = ""
    with open(sumocfg, "r") as cfgfile:
        lines = cfgfile.readlines()
        for line in lines:
            if "net-file" in line:
                data = line.split('"')
                netfile = data[1]
            if "route-files" in line: #This is scary - probably breaks if there's many of them
                data = line.split('"')
                roufile = data[1]
    return (netfile, roufile)
                

def writeSumoCfg(sumocfg, netfile, routefile):
    with open(sumocfg, "w") as cfgfile:
        print("""<?xml version="1.0" encoding="UTF-8"?>""", file=cfgfile)
        print("""<configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/sumoConfiguration.xsd">""", file=cfgfile)
        print("""<input>""", file=cfgfile)
        print("""<net-file value="%s"/>""" % (netfile), file=cfgfile)
        print("""<route-files value="%s"/>""" % (routefile), file=cfgfile)
        print("""</input>""", file=cfgfile)
        print("""</configuration>""", file=cfgfile)


#Generates induction loops on all the edges
def generate_additionalfile(sumoconfig, networkfile):
    #Create a third instance of a simulator so I can query the network
    print(sumoconfig)
    
    try:
        traci.start([checkBinary('sumo'), "-c", sumoconfig,
                            "--start", "--no-step-log", "true",
                            "--xml-validation", "never"], label="setup")
    except:
        #Worried about re-calling this without old main instance being removed
        #TODO: Something better than hard-coding a second name
        traci.start([checkBinary('sumo'), "-c", sumoconfig,
                            "--start", "--no-step-log", "true",
                            "--xml-validation", "never"], label=str(random.random()))


    net = sumolib.net.readNet(networkfile)
    rerouters = []
    global max_edge_speed

    with open("additional_autogen.xml", "w") as additional:
        print("""<additional>""", file=additional)
        for edge in traci.edge.getIDList():
            if edge[0] == ":":
                #Skip internal edges (=edges for the inside of each intersection)
                continue

            if (net.getEdge(edge).getSpeed() > max_edge_speed):
                max_edge_speed = net.getEdge(edge).getSpeed()

            #print(edge)
            for lanenum in range(traci.edge.getLaneNumber(edge)):
                lane = edge+"_"+str(lanenum)
                #print(lane)
                print('    <inductionLoop id="IL_%s" freq="1" file="outputAuto.xml" lane="%s" pos="-50" friendlyPos="true" />' \
                      % (lane, lane), file=additional)
                if len(net.getEdge(edge).getOutgoing()) > 1:
                    rerouters.append("IL_"+lane)
        print("</additional>", file=additional)
    
    return rerouters

# Gets successor edges of a given edge in a given network
# Parameters:
#   edge: an edge ID string
#   network: the network object from sumolib.net.readNet(netfile)
# Returns:
#   successors: a list of edge IDs for the successor edges (outgoing edges from the next intersection)
def getSuccessors(edge, network):
    ids = []
    for succ in list(network.getEdge(edge).getOutgoing()):
        ids.append(succ.getID())
    return ids

def main(netfile, sumoconfig):

    sumoBinary = checkBinary('sumo-gui')

    rerouters = generate_additionalfile(sumoconfig, netfile)
    print("MAX_EDGE_SPEED 2.0: {}".format(max_edge_speed))

    try:
        traci.start([sumoBinary, "-c", sumoconfig,
                            "--additional-files", "additional_autogen.xml",
                            "--log", "LOGFILE", "--xml-validation", "never", "--start", "--quit-on-end"], label="main")
    except:
        #Worried about re-calling this without old main instance being removed
        #TODO: Something better than just creating a second name and failing on the third...
        traci.start([sumoBinary, "-c", sumoconfig,
                            "--additional-files", "additional_autogen.xml",
                            "--log", "LOGFILE", "--xml-validation", "never", "--start", "--quit-on-end"], label="main2")
                                
    roadcarcounter = run(netfile, rerouters, sumoconfig)
    traci.close()
    return roadcarcounter


# this is the main entry point of this script
if __name__ == "__main__":

    #NOTE: Script name is zeroth arg
    if len(sys.argv) == 2:
        sumoconfig = sys.argv[1]
        (netfile, junk) = readSumoCfg(sumoconfig)
    else:
        sumoconfig = sys.argv[2]
        netfile = sys.argv[1]

    main(netfile, sumoconfig)