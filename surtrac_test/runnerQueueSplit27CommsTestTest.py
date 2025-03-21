#!/usr/bin/env python
#Adapted from TraCI tutorial here: https://github.com/eclipse/sumo/blob/main/tests/complex/tutorial/traci_tls/runner.py

#QueueSplit5 added a first iteration of a Surtrac model to QueueSplit4
#New in QueueSplit6: When there's multiple light phases a lane can go in, don't double-create clusters
#New in QueueSplit7: Adding Surtrac to simulate-ahead
#New in QueueSplit8: Adding communication between intersections. Which then requires re-compacting clusters, etc. For now, don't take advantage of known routes
#New in QueueSplit9: Take advantage of known routes from vehicle routing (NOTE: route may change as vehicle approaches intersection...)
#New in QueueSplit10: Optimize for speed (calling Surtrac less often in routing, merging predicted clusters)
#New in QueueSplit11: Compute delay, not just average travel time, for cars. Also fixed a bug with Surtrac code DP (was removing sequences it shouldn't have)
#QueueSplit12: Multithread the Surtrac code (it's really slow otherwise). Also, use the full Surtrac schedule rather than assuming we'll update every timestep
#QueueSplit13: Surtrac now (correctly) no longer overwrites all the finish times of other lanes with the start time of the currently scheduled cluster (leads to problems when a long cluster, then compatible short cluster, get scheduled, as the next cluster can then start earlier than it should). VOI now gets split into all lanes on starting edge
#14: Anytime routing, better stats on timeouts and teleports, added mingap to all cluster durations, using a timestep that divides mingap and surtracFreq, opposing traffic blocks for mingap not just one timestep, combining clusters at lights into a single queue
#15: Surtrac now runs once, assuming all vehicles travel their same routes, and then gets looked up during routing simulations. Also adding a flag to disable predicting clusters with Surtrac. REAL-TIME PERFORMANCE!!!
#16: Be lazy in routing simulations - don't need to check a lane until the front vehicle gets close to the end. (Didn't seem to give much speedup though.) Reusing Surtrac between multiple timesteps (and backported this to 15). Various attempts at cleanup and speedup.
#17: Move computation of Surtrac's predicted outflows to the end, after we already know what the schedule should be, rather than building it into the algorithm and generating predicted outflows for schedules we won't end up using. Fixed a bug where initially splitting the VOI into all lanes added it at the modified start time, and possibly in the wrong place
#18: Add imitation learning version of Surtrac
#19: Going back to A* in SUMO since ghost cars are causing problems. Telling all ghost cars to turn left, and spawning left-turning ghost cars when the turn completes so we account for oncoming traffic
#20: Storing Surtrac results for reuse
#21: Libsumo (optional hopefully), multithreaded routing, save files under different names so we can run two sets of code without making one error
#22: Did routing before Surtrac for more realism (in case we're not simultaneous so Surtrac's out of date). Not much effect there.
#23: 21, but now using a detector model to reconstruct the traffic state at the start of each routing simulation
#24: Detector model stops tracking specific names of non-adopters
#25: New plan for lane changes - blindly sample which lane stuff ends up in
#26: Detector model for Surtrac in routing as well (since the goal is to approximate what the main simulation would be doing)
#27: Support new SurtracNet (single network for all intersections, takes in intersection geometry and light phase info)

from __future__ import absolute_import
from __future__ import print_function

import sys
import xml.etree.ElementTree as ET

from sumolib import checkBinary

useLibsumo = True
if useLibsumo:
    try:
        import libsumo as traci
    except:
        print("Error using libsumo. Dropping back to traci instead. Hopefully this is fine")
        useLibsumo = False
if not useLibsumo:
    import traci  #To interface with SUMO simulations

import sumolib #To query node/edge stuff about the network

sumoconfig = None

pSmart = 1.0 #Adoption probability
useLastRNGState = False #To rerun the last simulation without changing the seed on the random number generator
appendTrainingData = False#True

isSmart = dict() #Store whether each vehicle does our routing or not

nRoutingCalls = 0
nSuccessfulRoutingCalls = 0
routingTime = 0

netfile = "UNKNOWN_FILENAME_OOPS"

def run(network, rerouters, pSmart, verbose = True):
    global actualStartDict
    startDict = dict()
    endDict = dict()
    delayDict = dict()

    simtime = 0

    while traci.simulation.getMinExpectedNumber() > 0 and (not appendTrainingData or simtime < 5000):
        simtime += 1
        traci.simulationStep() #Tell the simulator to simulate the next time step

        #Decide whether new vehicles use our routing
        for vehicle in traci.simulation.getDepartedIDList():
            isSmart[vehicle] = False

            delayDict[vehicle] = 0#-hmetadict[goaledge][currentRoutes[vehicle][0]] #I'll add the actual travel time once the vehicle arrives
            startDict[vehicle] = simtime

        #Check predicted vs. actual travel times
        for vehicle in traci.simulation.getArrivedIDList():
            endDict[vehicle] = simtime

        #Plot and print stats
        if simtime%100 == 0 or not traci.simulation.getMinExpectedNumber() > 0:
            
            #Stats
            avgTime = 0
            avgTime0 = 0
            nCars = 0
            nSmart = 0

            for id in endDict:
                if actualStartDict[id] >= 600 and actualStartDict[id] < 3000:
                    nCars += 1
                    if isSmart[id]:
                        nSmart += 1

            for id in endDict:
                #Only look at steady state - ignore first and last 10 minutes of cars
                if actualStartDict[id] < 600 or actualStartDict[id] >= 3000:
                    continue

                ttemp = (endDict[id] - startDict[id])+delayDict[id]
                avgTime += ttemp/nCars

                #Delay0 computation (start clock at intended entrance time)
                ttemp0 = (endDict[id] - actualStartDict[id])+delayDict[id]
                avgTime0 += ttemp0/nCars


            if verbose or not traci.simulation.getMinExpectedNumber() > 0 or (appendTrainingData and simtime == 5000):
                print(pSmart)
                print("\nCurrent simulation time: %f" % simtime)
                #print("Total run time: %f" % (time.time() - tstart))
                print("Number of vehicles in network: %f" % traci.vehicle.getIDCount())
                print("Total cars that left the network: %f" % len(endDict))
                print("Average delay: %f" % avgTime)
                print("Average delay0: %f" % avgTime0)
                
    return []

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

def main(sumoconfigin, pSmart, verbose = True, useLastRNGState = False, appendTrainingDataIn = False):
    global lanes
    global actualStartDict
    global sumoconfig
    global netfile

    sumoconfig = sumoconfigin

    # this script has been called from the command line. It will start sumo as a
    # server, then connect and run

    #sumoBinary = checkBinary('sumo')
    sumoBinary = checkBinary('sumo-gui')
    #NOTE: Script name is zeroth arg

    (netfile, routefile) = readSumoCfg(sumoconfig)

    network = sumolib.net.readNet(netfile)
    net = network

    # this is the normal way of using traci. sumo is started as a
    # subprocess and then the python script connects and runs
    if useLibsumo:
        traci.load(["-c", sumoconfig,
                                #"--additional-files", "additional_autogen.xml",
                                "--no-step-log", "true",
                                "--log", "LOGFILE", "--xml-validation", "never", "--start", "--quit-on-end"])
    else:
        try:
            traci.start([sumoBinary, "-c", sumoconfig,
                                    #"--additional-files", "additional_autogen.xml",
                                    "--no-step-log", "true",
                                    "--log", "LOGFILE", "--xml-validation", "never", "--start", "--quit-on-end"], label="main")
            #Second simulator for running tests. No GUI
            #traci.start([sumoBinary, "-c", sumoconfig, #GUI in case we need to debug
            traci.start([checkBinary('sumo'), "-c", sumoconfig, #No GUI
                                    #"--additional-files", "additionalrouting_autogen.xml",
                                    "--start", "--no-step-log", "true",
                                    "--xml-validation", "never", "--quit-on-end",
                                    "--step-length", "1"], label="test")
            dontBreakEverything()
        except:
            #Worried about re-calling this without old main instance being removed
            if not useLibsumo:
                traci.switch("main")
            traci.load([ "-c", sumoconfig,
                                    "--additional-files", "additional_autogen.xml",
                                    "--no-step-log", "true",
                                    #"--time-to-teleport", "-1",
                                    "--log", "LOGFILE", "--xml-validation", "never", "--start", "--quit-on-end"])
            if not useLibsumo:
                traci.switch("test")
            traci.load([ "-c", sumoconfig,
                                    "--additional-files", "additionalrouting_autogen.xml",
                                    "--no-step-log", "true",
                                    #"--time-to-teleport", "-1",
                                    "--log", "LOGFILE", "--xml-validation", "never", "--start", "--quit-on-end"])
            dontBreakEverything()


    #Parse route file to get intended departure times (to account for delayed SUMO insertions due to lack of space)
    #Based on: https://www.geeksforgeeks.org/xml-parsing-python/
    # create element tree object
    tree = ET.parse(routefile)

    # get root element
    root = tree.getroot()

    actualStartDict = dict()
    # iterate news items
    for item in root.findall('./vehicle'):
        actualStartDict[item.attrib["id"]] = float(item.attrib["depart"])
    for item in root.findall('./trip'):
        actualStartDict[item.attrib["id"]] = float(item.attrib["depart"])

    outdata = run(network, [], pSmart, verbose)
    
    #return [outdata, rngstate]

#Magically makes the vehicle lists stop deleting themselves somehow???
def dontBreakEverything():
    if not useLibsumo:
        traci.switch("test")
        traci.simulationStep()
        traci.switch("main")


# this is the main entry point of this script
if __name__ == "__main__":
    if len(sys.argv) >= 3:
        pSmart = float(sys.argv[2])
    if len(sys.argv) >= 4:
        useLastRNGState = sys.argv[3]
    if len(sys.argv) >= 5:
        appendTrainingData = sys.argv[4]
    main(sys.argv[1], pSmart, True, useLastRNGState, appendTrainingData)
