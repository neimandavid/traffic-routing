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

import sys
import random
import numpy as np
import pickle
import runnerDefaultWriter

preoptcarcounter = dict()
postoptcarcounter = dict()

#Assume we have a network, edge-to-edge turn data, and #vehicles/hr on input roads (pass these in)
#For each input road, generate n (=vehicles/hr) vehicles, pick starting times uniformly at random from [0, 3599].
#Sort ALL vehicles (combine all input roads) by start times.
#For each vehicle, from current road, pick next road using turn data. Stop when we pick an output road
#Then drop the intermediate roads, keep the origin destination pair. (Might bias toward shorter routes than reality, though)

def parseData(datafilepath, netfilename, routefilename, configfilename):
    nin = dict()
    nin["EXIT"] = 0
    nout = dict()
    turncounts = dict()
    with open(datafilepath, 'r') as datafile:
        for line in datafile:
            temp = line.split(">")
            if len(temp) == 1:
                continue #">" not found
            tempp = temp[1].split(":")
            if len(tempp) == 1:
                continue #":" not found
            splitline = [temp[0], tempp[0], int(tempp[1])]


            if splitline[0] in nout:
                nout[splitline[0]] += splitline[2]
            else:
                nout[splitline[0]] = splitline[2]

            if splitline[1] in nin:
                nin[splitline[1]] += splitline[2]
            else:
                nin[splitline[1]] = splitline[2]

            if not splitline[0] in turncounts:
                turncounts[splitline[0]] = dict()
            turncounts[splitline[0]][splitline[1]] = splitline[2]
    
    #We've read all the data now

    #Account for edges with some (but not all) cars exiting
    for edge in nout:
        if edge in nin:
            nToGenerate = nout[edge]-nin[edge]
            if nToGenerate < 0:
                print("Warning: " + str(-nToGenerate) + " cars disappear from edge " + edge + ". Double-check data to make sure that's intended.")
                turncounts[edge]["EXIT"] = -nToGenerate
                nin["EXIT"] -= nToGenerate

    #Compute edge>edge turn ratios
    edgeturnratios = pickle.loads(pickle.dumps(turncounts)) #Deep copy. Probably don't need to copy as long as we save turncountsum separately though
    for edge1 in turncounts:
        turncountsum = 0
        for edge2 in turncounts[edge1]:
            turncountsum += turncounts[edge1][edge2]
        for edge2 in edgeturnratios[edge1]:
            edgeturnratios[edge1][edge2] = turncounts[edge1][edge2] / turncountsum

    cars = []

    for edge in nout:
        if not edge in nin:
            nToGenerate = nout[edge]
            print("Generating " + str(nToGenerate) + " cars on edge " + edge)
            for carindex in range(nToGenerate):
                cars.append(makeCar(edge, carindex, edgeturnratios, nin, nout))
        else:
            nToGenerate = nout[edge]-nin[edge]
            if nToGenerate > 0:
                print("Warning: " + str(nToGenerate) + " cars appear on edge " + edge + ". Double-check data to make sure that's intended.")
                for carindex in range(nToGenerate):
                    cars.append(makeCar(edge, carindex, edgeturnratios, nin, nout))
    
    cars.sort(key = lambda y: y[1])

    #Start printing the route file
    with open(routefilename, "w") as routefile:
        print("""<?xml version="1.0" encoding="UTF-8"?>""", file=routefile)
        print("""<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">
""", file=routefile)
        print("""<vType id="noVar" speedFactor="1.0" speedDev="0.0"/>""", file=routefile)
        for car in cars:
            print("""<trip type="noVar" depart="%i" id="%s" from="%s" to="%s" />""" % (car[1], car[0], car[2], car[3]), file=routefile)
        print("""</routes>""", file=routefile)

    runnerDefaultWriter.writeSumoCfg(configfilename, netfilename, routefilename)

    postoptcarcounter = runnerDefaultWriter.main(netfilename, configfilename)

    preoptcountererr = dict()
    postoptcountererr = dict()
    for edge in preoptcarcounter:
        if not edge in postoptcarcounter:
            postoptcarcounter[edge] = 0
        assert(edge in nin)
        preoptcountererr[edge] = preoptcarcounter[edge] - nin[edge]
        postoptcountererr[edge] = postoptcarcounter[edge] - nin[edge]

    print("Preopt: Mean error " + str(np.mean(list(preoptcountererr.values()))) + ", std dev " + str(np.std(list(preoptcountererr.values()))))
    print("Postopt: Mean error " + str(np.mean(list(postoptcountererr.values()))) + ", std dev " + str(np.std(list(postoptcountererr.values()))))


            
def makeCar(edge, carindex, edgeturnratios, nin, nout):
    starttime = np.floor(random.random()*3600)
    startedge = edge
    prevedge = startedge
    while edge in nout:
        #Sample next edge using edgeturnratios
        r = random.random()
        for nextedge in edgeturnratios[edge]:
            r -= edgeturnratios[edge][nextedge]
            if r <= 0:
                #Use this as nextedge
                prevedge = edge
                edge = nextedge
                if not edge in preoptcarcounter:
                    preoptcarcounter[edge] = 0
                preoptcarcounter[edge] += 1
                break
    if edge == "EXIT":
        endedge = prevedge
    else:
        endedge = edge
    return (startedge+"."+str(carindex), starttime, startedge, endedge)


# this is the main entry point of this script
if __name__ == "__main__":
    datafile = sys.argv[2]
    netfilename = sys.argv[1]
    if len(sys.argv) > 3:
        routefilename = sys.argv[3]
    else:
        routefilename = datafile.split(".")[0]+".rou.xml"
    if len(sys.argv) > 4:
        configfilename = sys.argv[4]
    else:
        configfilename = datafile.split(".")[0]+".sumocfg"

    parseData(datafile, netfilename, routefilename, configfilename)