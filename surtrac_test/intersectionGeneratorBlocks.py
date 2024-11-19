import numpy as np
import random
import math
import torch
import sumolib
import pickle #To save/load traffic light states
import time
from copy import deepcopy, copy

multithreadSurtrac = False

if multithreadSurtrac:
    from multiprocessing import Process
    import multiprocessing
    manager = multiprocessing.Manager()

import itertools

# try:
#     multiprocessing.set_start_method("spawn")
# except:
#     pass

lightlanes = dict()
nLanes = dict()#[0, 0, 0, 0]
surtracdata = dict()
lightphasedata = dict()
lightoutlanes = dict()
lanephases = dict()
trainingdata = dict()

crossEntropyLoss = True#False #If false, mean-squared error on time before switch
nruns = 10000

mingap = 2.5 #Seconds between cars

#@profile
def intersectionGenerator():
    global nLanes
    global surtracdata

    nRoads = 4
    maxNLanes = 3
    nLanes = dict()#[0, 0, 0, 0]

    simtime = 0#RIR(0, 5000) #nninputsurtrac already subtracts off simtime from the arrival and departure time, which is how this had any hope of working back in the thesis proposal. Should either not use time at all or just hardcode 0
    isT = random.random() < 0.5

    for i in range(nRoads):
        nLanes[str(i)] = 2#RIR(0, maxNLanes) #n = somewhere between 0 and maxNLanes lanes on each road
        if i == 3:
            if not isT:
                nLanes[str(i)] = 2
            else:
                nLanes[str(i)] = 0

    #Literally just make up random nonsense for red-green phases
    maxNGreenPhases = 6
    nGreenPhases = RIR(1, maxNGreenPhases)
    if not isT:
        nGreenPhases = 4
    else:
        nGreenPhases = 3
    nPhases = nGreenPhases*2
    lightseqs = []
    clusters = dict()
    maxNClusters = 1#2#5
    minClusterGap = 5
    maxClusterGap = 10
    maxClusterWeight = 10

    lightlanes["light"] = []

    #Pretty sure 0th lane is right lane, but won't matter anyway unless we disable data augmentation
    for i in range(nRoads):
        lightseqs.append([])
        for j in range(nLanes[str(i)]):
            lightseqs[i].append([])
    if not isT:
        #Standard lagging left stuff
        lightseqs[0][0] = addYellows([1, 0, 0, 0])
        lightseqs[0][1] = addYellows([0.1, 1, 0, 0]) #Left lane: [g, G, r, r]
        lightseqs[1][0] = addYellows([0, 0, 1, 0])
        lightseqs[1][1] = addYellows([0, 0, 0.1, 1]) #Left lane: [r, r, g, G]
        lightseqs[2] = lightseqs[0]
        lightseqs[3] = lightseqs[1]
    else:
        #Road 3 doesn't exist, road 1 thus doesn't need a protected left
        #Road 2 can't turn left and road 1 can't go straight, thus in protected left phase road 1's right lane can still go
        lightseqs[0][0] = addYellows([1, 0, 0])
        lightseqs[0][1] = addYellows([0.1, 1, 0]) #Left lane: [g, G, r]
        lightseqs[2][0] = lightseqs[0][0] #Road 2 can't turn left, so treat its left turn lane as a straight lane instead
        lightseqs[2][1] = lightseqs[0][0]
        lightseqs[1][0] = addYellows([1, 0, 1]) #Road 1 right lane can go during the protected left
        lightseqs[1][1] = addYellows([0, 0, 1])
    assert(len(lightseqs[0][0]) == nPhases)

    for i in range(nRoads):
        #lightseqs.append([])
        for j in range(nLanes[str(i)]):
            #Get a random non-zero vector of length nGreenPhases and interpolate yellows
            #lightseqs[i].append(addYellows(nonZeroVector(nGreenPhases)))

            #clusters[lane][clusterind] is a dict, with startpos, endpos, time, arrival, departure, cars, weight
            #Probably only need arrival, departure, weight
            lane = str(i) + "_" + str(j)
            lightlanes["light"].append(lane)
            lanephases[lane] = []
            clusters[lane] = []
            nClusters = RIR(0, maxNClusters)
            lastdepart = simtime
            for k in range(nClusters):
                tempcluster = dict()
                tempcluster["arrival"] = lastdepart + RIR(minClusterGap, maxClusterGap)
                if k == 0:
                    tempcluster["arrival"] = lastdepart + RIR(0, maxClusterGap)
                tempcluster["weight"] = RIR(1, maxClusterWeight) #TODO consider making this log-uniform rather than straight uniform?
                assert(tempcluster["weight"] > 0)
                tempdur = (tempcluster["weight"]-1)*mingap
                tempcluster["departure"] = tempcluster["arrival"] + RIR(tempdur, 2*tempdur)
                lastdepart = tempcluster["departure"]
                clusters[lane].append(tempcluster)

    phase = RIR(0, (nGreenPhases-1))*2 #Current light phase - forced to be even, thus green. -1 because RIR is inclusive
    mindur = 5
    maxdur = 120 #Max duration of a green phase

    lightphases = dict()
    lightphases["light"] = phase
    lastswitchtimes = dict()
    lastswitchtimes["light"] = simtime - RIR(mindur, maxdur)
    lightoutlanes["light"] = []

    surtracdata["light"] = []
    lightphasedata["light"] = []
    for i in range(nPhases):
        lightphasedata["light"].append([])
        lightphasedata["light"][i] = sumolib.net.Phase(0, "")
        #lightphasedata["light"][i].state = ""
        surtracdata["light"].append(dict())
        surtracdata["light"][i]["minDur"] = mindur
        if i % 2 == 0: #Green phase
            surtracdata["light"][i]["maxDur"] = maxdur
        else:
            surtracdata["light"][i]["maxDur"] = 7#mindur #Because yellow phases
            surtracdata["light"][i]["minDur"] = 7#mindur #Because yellow phases
            
        surtracdata["light"][i]["lanes"] = []
        for j in range(nRoads):
            for k in range(nLanes[str(j)]):
                if lightseqs[j][k][i]:
                    surtracdata["light"][i]["lanes"].append(str(j) + "_" + str(k))
                    lightphasedata["light"][i].state += "G"
                    lanephases[str(j) + "_" + str(k)].append(i)
                else:
                    if lightseqs[j][k][i-1]: #Just turned from green to not-green, thus yellow
                        lightphasedata["light"][i].state += "Y"
                    else:
                        lightphasedata["light"][i].state += "R"

    for i in range(nPhases):
        #Compute min transition time between the start of any two phases
        surtracdata["light"][i]["timeTo"] = [0]*nPhases
        for joffset in range(1, nPhases):
            j = (i + joffset) % nPhases
            jprev = (j-1) % nPhases
            surtracdata["light"][i]["timeTo"][j] = surtracdata["light"][i]["timeTo"][jprev] + surtracdata["light"][jprev]["minDur"]
    
    # print(lightseqs)
    # print(clusters)
    # print(surtracdata)

    #convertToNNInputSurtrac(simtime, "light", clusters, lightphases, lastswitchtimes) #Runs now, yay!
    #bestschedules = dict()
    doSurtracThread("network", simtime, "light", clusters, lightphases, lastswitchtimes, False, 10, [], dict(), dict())
    #print(bestschedules["light"])
    #print("done")

def RIR(min, max):
    #Returns a random int between min and max, inclusive
    if min > max:
        print("min > max, things are probably wrong") #Though RIR itself probably still returns decent stuff. The +1 is probably the wrong direction, though
    return math.floor(random.random()*(max-min+1)) + min #floor cancels the +1. Ex: RIR(1, 6) = rand*6+1 = [0, 5] + 1

def nonZeroVector(length):
    if length == 0:
        #Technically everything's 0, but what else can we do here?
        return []
    out = []
    allZero = True
    for i in range(length):
        temp = RIR(0, 1)
        out.append(temp)
        if temp > 0:
            allZero = False
    if allZero:
        return nonZeroVector(length)
    else:
        return out

#Interpolates yellow phases between the input phase vector v. If green in successive phases, green in the middle, else yellow/red in the middle
def addYellows(v):
    l = len(v)
    w = []
    for i in range(l):
        w.append(int(v[i]))
        w.append(int(np.ceil(v[i])) & int(np.ceil(v[(i+1)%l])) & (int(v[i]) | int(v[(i+1)%l])))
    return w


#Blind copy-paste from RQS27:
#@profile
def convertToNNInputSurtrac(simtime, light, clusters, lightphases, lastswitchtimes, lightlanes = lightlanes):
    maxnlanes = 3 #Going to assume we have at most 3 lanes per road, and that the biggest number lane is left-turn only
    maxnroads = 4 #And assume 4-way intersections for now
    maxnclusters = 5 #And assume at most 10 clusters per lane
    ndatapercluster = 3 #Arrival, departure, weight
    maxnphases = 12 #Should be enough to handle both leading and lagging lefts
    phasevec = np.zeros(maxnphases)
    

    clusterdata = np.zeros(maxnroads*maxnlanes*maxnclusters*ndatapercluster)
    greenlanes = np.zeros(maxnphases*maxnroads*maxnlanes)

    phase = lightphases[light]
    phasevec[phase] = 1
    lastSwitch = lastswitchtimes[light]

    # maxfreq = max(routingSurtracFreq, mainSurtracFreq, timestep, 1)

    # if surtracdata[light][phase]["maxDur"]- maxfreq <= surtracdata[light][phase]["minDur"]:
    #     #Edge case where duration range is smaller than period between updates, in which case overruns are unavoidable
    #     if simtime - lastSwitch < surtracdata[light][phase]["minDur"]:
    #         phaselenprop = -1
    #     else:
    #         phaselenprop = 2
    # else:
    #     phaselenprop = (simtime - lastSwitch - surtracdata[light][phase]["minDur"])/(surtracdata[light][phase]["maxDur"]- maxfreq - surtracdata[light][phase]["minDur"])

    # #phaselenprop = (simtime - lastSwitch - surtracdata[light][phase]["minDur"])/surtracdata[light][phase]["maxDur"]
    # #phaselenprop is negative if we're less than minDur, and greater than 1 if we're greater than maxDur

    phaselenprop = (surtracdata[light][phase]["maxDur"] - (simtime - lastSwitch)) #Time until we hit max dur
    #phaselenprop = (maxdur - (simtime - lastSwitch)) #Time until we hit max dur

    prevRoad = None
    roadind = -1
    laneind = -1

    for lane in lightlanes[light]:
        temp = lane.split("_")
        road = temp[0] #Could have problems if road name has underscores, but ignoring for now...
        if nLanes[road] > maxnlanes:
            print("Warning: " + str(road) + " exceeds maxnlanes in convertToNNInput, ignoring some lanes")
        lanenum = int(temp[-1])
        if road != prevRoad or roadind < 0:
            roadind += 1
            assert(roadind < maxnroads)
            laneind = -1
            prevRoad = road
        laneind += 1
        assert(laneind < maxnlanes)

        #Not sharing weights so I'll skip this
        #Last lane on road assumed to be left-turn only and being inserted in last slot
        # if laneind + 1 == nLanes[road] or laneind + 1 >= maxnlanes:
        #     laneind = maxnlanes - 1

        for clusterind in range(len(clusters[lane])):
            if clusterind > maxnclusters:
                print("Warning: Too many clusters on " + str(lane) + ", ignoring the last ones")
                break
            clusterdata[((roadind*maxnlanes+laneind)*maxnclusters+clusterind)*ndatapercluster : ((roadind*maxnlanes+laneind)*maxnclusters+clusterind+1)*ndatapercluster] = [clusters[lane][clusterind]["arrival"]-simtime, clusters[lane][clusterind]["departure"]-simtime, clusters[lane][clusterind]["weight"]]


        for i in range(len(surtracdata[light])):
            assert(i < maxnphases)
            if lane in surtracdata[light][i]["lanes"]:
                greenlanes[roadind*maxnlanes*maxnphases+laneind*maxnphases+i] = 1
                #greenlanes should look like [road1lane1greenphases, road1lane2greenphases, etc] where each of those is just a binary vector with 1s for green phases


    #return torch.Tensor(np.array([np.concatenate(([phase], [phaselenprop]))]))
    return torch.Tensor(np.array([np.concatenate((clusterdata, greenlanes, phasevec, [phaselenprop], [simtime]))]))

#@profile
def doSurtracThread(network, simtime, light, clusters, lightphases, lastswitchtimes, inRoutingSim, predictionCutoff, toSwitch, catpreds, bestschedules):
    global totalSurtracRuns
    global totalSurtracClusters
    global totalSurtracTime

    # if inRoutingSim:
    #     freq = max(routingSurtracFreq, timestep)
    #     ttimestep = timestep
    # else:
    #     freq = max(mainSurtracFreq, 1)
    #     ttimestep = 1
    freq = 1
    ttimestep = 1
    learnYellow = False
    learnMinMaxDurations = False
    testNN = False
    testDumbtrac = False
    debugMode = False
    disableSurtracPred = True
    appendTrainingData = True #For now

        
    i = lightphases[light]
    if not learnYellow and ("Y" in lightphasedata[light][i].state or "y" in lightphasedata[light][i].state):
        #Force yellow phases to be min duration regardless of what anything else says, and don't store it as training data
        if simtime - lastswitchtimes[light] >= surtracdata[light][i]["minDur"]:
            dur = 0
        else:
            dur = (surtracdata[light][i]["minDur"] - (simtime - lastswitchtimes[light]))//ttimestep*ttimestep
        #Replace first element with remaining duration, rather than destroying the entire schedule, in case of Surtrac or similar
        if light in bestschedules:
            temp = pickle.loads(pickle.dumps(bestschedules[light][7]))
        else:
            temp = [0]
        temp[0] = dur
        bestschedules[light] = [None, None, None, None, None, None, None, temp]
        return

    if not learnMinMaxDurations:
        #Force light to satisfy min/max duration requirements and don't store as training data
        if simtime - lastswitchtimes[light] < surtracdata[light][i]["minDur"]:
            dur = 1e6
            if light in bestschedules:
                temp = pickle.loads(pickle.dumps(bestschedules[light][7]))
            else:
                temp = [0]
            temp[0] = dur
            bestschedules[light] = [None, None, None, None, None, None, None, temp]
            return
        if simtime - lastswitchtimes[light] + freq > surtracdata[light][i]["maxDur"]:
            #TODO this is slightly sloppy if freq > ttimestep - if we're trying to change just before maxDur this'll assume we tried to change at it instead
            dur = (surtracdata[light][i]["maxDur"] - (simtime - lastswitchtimes[light]))//ttimestep*ttimestep
            if light in bestschedules:
                temp = pickle.loads(pickle.dumps(bestschedules[light][7]))
            else:
                temp = [0]
            temp[0] = dur
            bestschedules[light] = [None, None, None, None, None, None, None, temp]
            return

    # if (testNN and (inRoutingSim or not noNNinMain)) or testDumbtrac: #If using NN and/or dumbtrac
    #     if (testNN and (inRoutingSim or not noNNinMain)): #If using NN
    #         nnin = convertToNNInputSurtrac(simtime, light, clusters, lightphases, lastswitchtimes)

    #         # if testDumbtrac: #And also dumbtrac
    #         #     nnin = convertToNNInputSurtrac(simtime, light, clusters, lightphases, lastswitchtimes)

    #         #     # nnin = convertToNNInput(simtime, light, clusters, lightphases, lastswitchtimes) #Obsolete - Surtrac architecture works for dumbtrac too!
    #         # else: #NN but not dumbtrac
    #         #     nnin = convertToNNInputSurtrac(simtime, light, clusters, lightphases, lastswitchtimes)
            
    #         # surtracStartTime = time.time()
    #         # totalSurtracRuns += 1
        
    #         temp = agents["light"](nnin).detach().cpu().numpy() # Output from NN
    #         # print(temp)
    #         # asdf
    #         outputNN = temp[0][1] - temp[0][0]

    #         # if debugMode:
    #         #     totalSurtracTime += time.time() - surtracStartTime

    #         if outputNN <= 0:
    #             actionNN = 1 #Switch
    #         else:
    #             actionNN = 0 #Stick

    #     if testDumbtrac and not (testNN and (inRoutingSim or not noNNinMain)): #Dumbtrac but not NN
    #         outputDumbtrac = dumbtrac(simtime, light, clusters, lightphases, lastswitchtimes)
    #         if outputDumbtrac <= 0: #Stick for <= 0 seconds
    #             actionDumbtrac = 1 #Switch
    #         else:
    #             actionDumbtrac = 0 #Stick
    #         actionNN = actionDumbtrac

    #     if actionNN == 0:
    #         dur = 1e6 #Something really big so we know the light won't change
    #     else:
    #         dur = 0
    #     testnnschedule = [None, None, None, None, None, None, None, [dur]] #Only thing the output needs is a schedule; returns either [0] for switch immediately or [1] for continue for at least another timestep
    #     assert(len(testnnschedule[7]) > 0)
    #     #return #Don't return early, might still need to append training data

    if (not (testNN and (inRoutingSim or not noNNinMain)) and not testDumbtrac) or (appendTrainingData and not testDumbtrac): #(No NN or append training data) and no dumbtrac - get the actual Surtrac result
        #print("Running surtrac, double-check that this is intended.")
        #We're actually running Surtrac

        # surtracStartTime = time.time()
        # totalSurtracRuns += 1

        sult = 3 #Startup loss time
        greedyDP = True

        #Figure out what an initial and complete schedule look like
        nPhases = len(surtracdata[light]) #Number of phases
        bestschedules[light] = [[]] #In case we terminate early or something??

        emptyStatus = dict()
        fullStatus = dict()
        nClusters = 0
        maxnClusters = 0

        #Does this vectorize somehow?
        for lane in lightlanes[light]:
            emptyStatus[lane] = 0
            fullStatus[lane] = len(clusters[lane])
            nClusters += fullStatus[lane]
            if maxnClusters < fullStatus[lane]:
                maxnClusters = fullStatus[lane]
        if debugMode:
            totalSurtracClusters += nClusters
        #If there's nothing to do, send back something we recognize as "no schedule"
        if nClusters == 0:
            bestschedules[light] = [[]]
            return
        if nClusters > 20:
            return #Else we get really really slow

        #Stuff in the partial schedule tuple
        #0: list of indices of the clusters we've scheduled, as tuples (phase, laneind)
        #1: schedule status (how many clusters from each lane we've scheduled)
        #2: current light phase
        #3: time when each direction will have finished its last scheduled cluster
        #4: time when all directions are finished with scheduled clusters ("total makespan" + starting time...)
        #5: total delay
        #6: last switch time
        #7: planned total durations of all phases
        #8: predicted outflows (as clusters - arrival, departure, list of cars, weights, etc.) Blank for now since I'll generate it at the end.
        #9: pre-predict data (cluster start times and compression factors) which I'll use to figure out predicted outflows once we've determined the best schedule

        emptyPreds = dict()
        for lane in lightoutlanes[light]:
            emptyPreds[lane] = []

        # emptyPrePreds = dict()
        # for lane in lightlanes[light]:
        #     emptyPrePreds[lane] = []
        lenlightlaneslight = len(lightlanes[light])
        assert(lenlightlaneslight > 0)
        emptyPrePreds = np.zeros((lenlightlaneslight, maxnClusters, 2))

        phase = lightphases[light]
        lastSwitch = lastswitchtimes[light]
        schedules = [([], emptyStatus, phase, [simtime]*len(surtracdata[light][phase]["lanes"]), simtime+mingap, 0, lastSwitch, [simtime-lastSwitch], [], emptyPrePreds)]

        #print("nClusters: " + str(nClusters))
        for _ in range(nClusters): #Keep adding a cluster until #clusters added = #clusters to be added
            scheduleHashDict = dict()
            for schedule in schedules:
                for laneindex in range(lenlightlaneslight):
                    lane = lightlanes[light][laneindex]
    
                    if schedule[1][lane] == fullStatus[lane]:
                        continue
                    #Consider adding next cluster from surtracdata[light][i]["lanes"][j] to schedule
                    newScheduleStatus = copy(schedule[1]) #Shallow copy okay? Dict points to int, which is stored by value
                    newScheduleStatus[lane] += 1
                    assert(newScheduleStatus[lane] <= maxnClusters)
                    phase = schedule[2]

                    #Now loop over all phases where we can clear this cluster
                    try:
                        assert(len(lanephases[lane]) > 0)
                    except Exception as e:
                        print(lane)
                        print("ERROR: Can't clear this lane ever?")
                        raise(e)
                        
                    for i in lanephases[lane]:
                        if not learnYellow and ("Y" in lightphasedata[light][i].state or "y" in lightphasedata[light][i].state):
                            continue
                        directionalMakespans = copy(schedule[3])

                        nLanes = len(surtracdata[light][i]["lanes"])
                        j = surtracdata[light][i]["lanes"].index(lane)

                        newDurations = copy(schedule[7]) #Shallow copy should be fine

                        clusterind = newScheduleStatus[lane]-1 #We're scheduling the Xth cluster; it has index X-1
                        ist = clusters[lane][clusterind]["arrival"] #Intended start time = cluster arrival time
                        dur = clusters[lane][clusterind]["departure"] - ist + mingap #+mingap because next cluster can't start until mingap after current cluster finishes
                        
                        mindur = max((clusters[lane][clusterind]["weight"] )*mingap, 0) #No -1 because fencepost problem; next cluster still needs 2.5s of gap afterwards
                        delay = schedule[5]

                        if dur < mindur:
                            #print("Warning, dur < mindur???")
                            dur = mindur

                        if phase == i:
                            pst = schedule[3][j]
                            newLastSwitch = schedule[6] #Last switch time doesn't change
                            ast = max(ist, pst)
                            newdur = max(dur - (ast-ist), mindur) #Try to compress cluster as it runs into an existing queue
                            currentDuration = max(ist, ast)+newdur-schedule[6] #Total duration of current light phase if we send this cluster without changing phase

                        if not phase == i or currentDuration > surtracdata[light][i]["maxDur"]: #We'll have to switch the light, possibly mid-cluster

                            if not phase == i:
                                #Have to switch light phases.
                                newFirstSwitch = max(schedule[6] + surtracdata[light][phase]["minDur"], schedule[4]-mingap) #Because I'm adding mingap after all clusters, but here the next cluster gets delayed
                            else:
                                #This cluster is too long to fit entirely in the current phase
                                newFirstSwitch = schedule[6] + surtracdata[light][phase]["maxDur"] #Set current phase to max duration
                                #Figure out how long the remaining part of the cluster is
                                tSent = surtracdata[light][i]["maxDur"] - (max(ist, ast)-schedule[6]) #Time we'll have run this cluster for before the light switches
                                if tSent < 0: #Cluster might arrive after the light would have switched due to max duration (ist is big), which would have made tSent go negative
                                    tSent = 0
                                    try:
                                        assert(mindur >= 0)
                                        assert(dur >= 0)
                                    except AssertionError as e:
                                        print(mindur)
                                        print(dur)
                                        raise(e)

                                if mindur > 0 and dur > 0: #Having issues with negative weights, possibly related to cars contributing less than 1 to weight having left the edge
                                    #We've committed to sending this cluster in current phase, but current phase ends before cluster
                                    #So we're sending what we can, cycling through everything else, then sending the rest
                                    #Compute the delay on the stuff we sent through, then treat the rest as a new cluster and compute stuff then
                                    delay += tSent/dur*clusters[surtracdata[light][i]["lanes"][j]][clusterind]["weight"]*((ast-ist)-1/2*(dur-newdur) ) #Weight of stuff sent through, times amount the start time got delayed minus half the squishibility
                                    mindur *= 1-tSent/dur #Assuming uniform density, we've sent tSent/dur fraction of vehicles through, so 1-tSent/dur remain to be handled
                                else:
                                    print("Negative weight, what just happened?")
                                dur -= tSent

                            newLastSwitch = newFirstSwitch + surtracdata[light][(phase+1)%nPhases]["timeTo"][i] #Switch right after previous cluster finishes (why not when next cluster arrives minus sult? Maybe try both?)                        
                            pst = newLastSwitch + sult #Total makespan + switching time + startup loss time
                            #Technically this sult implementation isn't quite right, as a cluster might reach the light as the light turns green and not have to stop and restart
                            directionalMakespans = [pst]*nLanes #Other directions can't schedule a cluster before the light switches

                            newDurations[-1] = newFirstSwitch - schedule[6] #Previous phase lasted from when it started to when it switched
                            tempphase = (phase+1)%nPhases
                            while tempphase != i:
                                newDurations.append(surtracdata[light][i]["minDur"])
                                tempphase = (tempphase+1)%nPhases
                            newDurations.append(0) #Duration of new phase i. To be updated on future loops once we figure out when the cluster finishes
                            assert(newDurations != schedule[7]) #Confirm that shallow copy from before is fine

                        ast = max(ist, pst)
                        newdur = max(dur - (ast-ist), mindur) #Compress cluster once cars start stopping

                        assert(ast >= simtime)
                        assert(mindur > 0)

                        newPrePredict = copy(schedule[9])#pickle.loads(pickle.dumps(schedule[9]))
                        # print(np.shape(newPrePredict))
                        # print(lightoutlanes[light])
                        # print(lane)
                        # print(laneindex)
                        # print(newScheduleStatus[lane]-1)
                        newPrePredict[laneindex][newScheduleStatus[lane]-1][0] = ast #-1 because zero-indexing; first cluster has newScheduleStatus[lane] = 1, but is stored at index 0
                        if dur <= mindur:
                            newPrePredict[laneindex][newScheduleStatus[lane]-1][1] = 0 #Squish factor = 0 (no squishing)
                        else:
                            newPrePredict[laneindex][newScheduleStatus[lane]-1][1] = (dur-newdur)/(dur-mindur) #Squish factor equals this thing
                            #If newdur = mindur, compression factor = 1, all gaps are 2.5 (=mindur)
                            #If newdur = dur, compression factor = 0, all gaps are original values
                            #Otherwise smoothly interpolate
                        
                        #Tell other clusters to also start no sooner than max(new ast, old directionalMakespan value) to preserve order
                        #That max is important, though; blind overwriting is wrong, as you could send a long cluster, then a short one, then change the light before the long one finishes
                        assert(len(directionalMakespans) == len(surtracdata[light][i]["lanes"]))
                        directionalMakespans[j] = ast+newdur+mingap

                        directionalMakespans = np.maximum(directionalMakespans, ast).tolist()
                        
                        delay += clusters[surtracdata[light][i]["lanes"][j]][clusterind]["weight"]*((ast-ist)-1/2*(dur-newdur) ) #Delay += #cars * (actual-desired). 1/2(dur-newdur) compensates for the cluster packing together as it waits (I assume uniform compression)
                        try:
                            assert(delay >= schedule[5] - 1e-10) #Make sure delay doesn't go negative somehow
                        except AssertionError as e:
                            print("Negative delay, printing lots of debug stuff")
                            #print(clusters)
                            print(light)
                            print(lane)
                            print(clusters[surtracdata[light][i]["lanes"][j]][clusterind])
                            print(ast)
                            print(ist)
                            print(dur)
                            print(newdur)
                            print((ast-ist)-1/2*(dur-newdur))
                            raise(e)

                        newMakespan = max(directionalMakespans)
                        currentDuration = newMakespan - newLastSwitch
                        assert(currentDuration > 0)

                        newDurations[-1] = currentDuration 
                        #Stuff in the partial schedule tuple
                        #0: list of indices of the clusters we've scheduled
                        #1: schedule status (how many clusters from each lane we've scheduled)
                        #2: current light phase
                        #3: time when each direction will have finished its last scheduled cluster
                        #4: time when all directions are finished with scheduled clusters ("total makespan" + starting time...)
                        #5: total delay
                        #6: last switch time
                        #7: planned total durations of all phases
                        #8: predicted outflows (as clusters - arrival, departure, list of cars, weights, etc.) Blank for now since I'll generate it at the end.
                        #9: pre-predict data (cluster start times and compression factors) which I'll use to figure out predicted outflows once we've determined the best schedule

                        newschedule = (schedule[0]+[(i, j)], newScheduleStatus, i, directionalMakespans, newMakespan, delay, newLastSwitch, newDurations, [], newPrePredict)
                        try:
                            assert(newschedule[7][0] >= simtime-lastSwitch)
                        except Exception as e:
                            print(newschedule)
                            print(simtime-lastSwitch)
                            raise(e)
                        #DP on partial schedules
                        key = (tuple(newschedule[1].values()), newschedule[2]) #Key needs to be something immutable (like a tuple, not a list)

                        if not key in scheduleHashDict:
                            scheduleHashDict[key] = [newschedule]
                        else:
                            keep = True
                            testscheduleind = 0
                            while testscheduleind < len(scheduleHashDict[key]):
                                testschedule = scheduleHashDict[key][testscheduleind]

                                #These asserts should follow from how I set up scheduleHashDict
                                if debugMode:
                                    assert(newschedule[1] == testschedule[1])
                                    assert(newschedule[2] == testschedule[2])
                                
                                #NOTE: If we're going to go for truly optimal, we also need to check all makespans, plus the current phase duration
                                #OTOH, if people seem to think fast greedy approximations are good enough, I'm fine with that
                                if newschedule[5] >= testschedule[5] and (greedyDP or newschedule[4] >= testschedule[4]):
                                    #New schedule was dominated; remove it and don't continue comparing (old schedule beats anything new one would)
                                    keep = False
                                    break
                                if newschedule[5] <= testschedule[5] and (greedyDP or newschedule[4] <= testschedule[4]):
                                    #Old schedule was dominated; remove it
                                    scheduleHashDict[key].pop(testscheduleind)
                                    continue
                                #No dominance, keep going
                                testscheduleind += 1

                            if keep:
                                scheduleHashDict[key].append(newschedule)
                        if debugMode:
                            assert(len(scheduleHashDict[key]) > 0)

            schedules = sum(list(scheduleHashDict.values()), []) #Each key has a list of non-dominated partial schedules. list() turns the dict_values object into a list of those lists; sum() concatenates to one big list of partial schedules. (Each partial schedule is stored as a tuple)

        mindelay = np.inf
        bestschedule = [[]]
        for schedule in schedules:
            if schedule[5] < mindelay:
                mindelay = schedule[5]
                bestschedule = schedule

        if not bestschedule == [[]]:
            #We have our best schedule, now need to generate predicted outflows
            if disableSurtracPred:
                newPredClusters = emptyPreds
            else:
                newPredClusters = pickle.loads(pickle.dumps(emptyPreds)) #Deep copy needed if I'm going to merge clusters

                nextSendTimes = [] #Priority queue
                clusterNums = dict()
                carNums = dict()
                #for lane in lightlanes[light]:
                for laneind in range(lenlightlaneslight):
                    lane = lightlanes[light][laneind]
                    clusterNums[lane] = 0
                    carNums[lane] = 0
                    if len(clusters[lane]) > clusterNums[lane]: #In case there's no clusters on a given lane
                        #heappush(nextSendTimes, (bestschedule[9][lane][clusterNums[lane]][0], lane)) #Pre-predict for appropriate lane for first cluster, get departure time, stuff into a priority queue
                        heappush(nextSendTimes, (bestschedule[9][laneind][clusterNums[lane]][0], laneind)) #Pre-predict for appropriate lane for first cluster, get departure time, stuff into a priority queue

                while len(nextSendTimes) > 0:
                    (nextSendTime, laneind) = heappop(nextSendTimes)
                    lane = lightlanes[light][laneind]
                    if nextSendTime + fftimes[light] > simtime + predictionCutoff:
                        #fftimes[light] is the smallest fftime of any output lane
                        #So if we're here, there's no way we'll ever want to predict this or any later car
                        break

                    cartuple = clusters[lane][clusterNums[lane]]["cars"][carNums[lane]]
                    if not cartuple[0] in isSmart or isSmart[cartuple[0]]: #It's possible we call this from QueueSim, at which point we split the vehicle being routed and wouldn't recognize the new names. Anything else should get assigned to isSmart or not on creation
                        #Split on "|" and "_" to deal with splitty cars correctly
                        route = currentRoutes[cartuple[0].split("|")[0].split("_")[0]] #.split to deal with the possibility of splitty cars in QueueSim
                        edge = lane.split("_")[0]
                        if not edge in route:
                            #Not sure if or why this happens - maybe the route is changing and predictions aren't updating?
                            #Can definitely happen for a splitty car inside QueueSim
                            #Regardless, don't predict this car forward and hope for the best?
                            if not "|" in cartuple[0] and not "_" in cartuple[0]:
                                #Smart car is on an edge we didn't expect. Most likely it changed route between the previous and current Surtrac calls. Get rid of it for now; can we be cleverer?
                                # print(cartuple[0])
                                # print(route)
                                # print(edge)
                                # print("Warning, smart car on an edge that's not in its route. This shouldn't happen? Assuming a mispredict and removing")
                                continue
                            #TODO: else should predict it goes everywhere?
                            continue
                        edgeind = route.index(edge)
                        if edgeind+1 == len(route):
                            #At end of route, don't care
                            continue
                        nextedge = route[edgeind+1]
                        
                        if not nextedge in normprobs[lane]:
                            #Means normprobs[lane] would be 0; nobody turned from this lane to this edge in the initial data
                            #Might be happening if the car needs to make a last-minute lane change to stay on its route?
                            #TODO: Find a lane where it can continue with the route and go from there? Ignoring for now
                            #NEXT TODO: Apparently still a thing even with splitting the initial VOI to multiple lanes???
                            continue

                        for nextlaneind in range(nLanes[nextedge]):
                            nextlane = nextedge+"_"+str(nextlaneind)
                            arr = nextSendTime + fftimes[nextlane]
                            if arr > simtime + predictionCutoff:
                                #Don't add to prediction; it's too far in the future. And it'll be too far into the future for all other lanes on this edge too, so just stop
                                break

                            if not nextlane in turndata[lane] or turndata[lane][nextlane] == 0:
                                #Car has zero chance of going here, skip
                                continue

                            if len(newPredClusters[nextlane]) == 0 or arr > newPredClusters[nextlane][-1]["departure"] + clusterthresh:
                                #Add a new cluster
                                newPredCluster = dict()
                                newPredCluster["endpos"] = 0
                                newPredCluster["time"] = ast
                                newPredCluster["arrival"] = arr
                                newPredCluster["departure"] = arr
                                newPredCluster["cars"] = []
                                newPredCluster["weight"] = 0
                                newPredClusters[nextlane].append(newPredCluster)

                            modcartuple = (cartuple[0], arr, cartuple[2]*predDiscount*turndata[lane][nextlane] / normprobs[lane][nextedge], cartuple[3])
                            newPredClusters[nextlane][-1]["cars"].append(modcartuple)
                            newPredClusters[nextlane][-1]["weight"] += modcartuple[2]
                            newPredClusters[nextlane][-1]["departure"] = arr
                    else:
                        for nextlane in turndata[lane]:
                            #Copy-paste previous logic for creating a new cluster
                            arr = nextSendTime + fftimes[nextlane]
                            if arr > simtime + predictionCutoff:
                                #Don't add to prediction; it's too far in the future. Other lanes may differ though
                                continue

                            if not nextlane in turndata[lane] or turndata[lane][nextlane] == 0:
                                #Car has zero chance of going here, skip
                                continue

                            if len(newPredClusters[nextlane]) == 0 or arr > newPredClusters[nextlane][-1]["departure"] + clusterthresh:
                                #Add a new cluster
                                newPredCluster = dict()
                                newPredCluster["endpos"] = 0
                                newPredCluster["time"] = ast
                                newPredCluster["arrival"] = arr
                                newPredCluster["departure"] = arr
                                newPredCluster["cars"] = []
                                newPredCluster["weight"] = 0
                                newPredClusters[nextlane].append(newPredCluster)

                            modcartuple = (cartuple[0], arr, cartuple[2]*turndata[lane][nextlane], cartuple[3])
                            newPredClusters[nextlane][-1]["cars"].append(modcartuple)
                            newPredClusters[nextlane][-1]["weight"] += modcartuple[2]
                            newPredClusters[nextlane][-1]["departure"] = arr
                    
                    #Added car to predictions, now set up the next car
                    carNums[lane] += 1
                    while len(clusters[lane]) > clusterNums[lane] and len(clusters[lane][clusterNums[lane]]["cars"]) == carNums[lane]: #Should fire at most once, but use while just in case of empty clusters...
                        clusterNums[lane] += 1
                        carNums[lane] = 0
                    if len(clusters[lane]) == clusterNums[lane]:
                        #Nothing left on this lane, we're done here
                        #nextSendTimes.pop(lane)
                        continue
                    if carNums[lane] == 0:
                        heappush(nextSendTimes, (bestschedule[9][laneind][clusterNums[lane]][0], laneind)) #Time next cluster is scheduled to be sent
                    else:
                        #Account for cluster compression
                        prevSendTime = nextSendTime #When we sent the car above
                        rawSendTimeDelay = clusters[lane][clusterNums[lane]]["cars"][carNums[lane]][1] - clusters[lane][clusterNums[lane]]["cars"][carNums[lane]-1][1] #Time between next car and this car in the original cluster
                        compFac = bestschedule[9][laneind][clusterNums[lane]][1] #Compression factor in case cluster is waiting at a red light
                        sendTimeDelay = compFac*mingap + (1-compFac)*rawSendTimeDelay #Update time delay using compression factor
                        newSendTime = prevSendTime + sendTimeDelay #Compute time we'd send next car
                        heappush(nextSendTimes, (newSendTime, laneind))

            #Predicting should be done now
            #bestschedule[8] = newPredClusters #I'd store this, but tuples are immutable and we don't actually use it anywhere...
        
            catpreds.update(newPredClusters)
            if len(bestschedule[7]) > 0:
                bestschedule[7][0] -= (simtime - lastswitchtimes[light])
            bestschedules[light] = bestschedule
        else:
            print(light)
            print("No schedules anywhere? That shouldn't happen...")


            
        #nnin = convertToNNInput(simtime, light, clusters, lightphases, lastswitchtimes) #If stuff breaks, make sure none of this gets changed as we go. (Tested when I first wrote this.)
        # actionSurtrac = 0
        # if bestschedule[7][0] <= simtime - lastswitchtimes[light]:
        #     actionSurtrac = 1
        #NN should take in nnin, and try to return action, and do backprop accordingly
        #target = torch.tensor([bestschedule[7][0] - (simtime - lastswitchtimes[light])]) # Target from expert

        # if debugMode:
        #     totalSurtracTime += time.time() - surtracStartTime

    if appendTrainingData:
        if testDumbtrac:
            outputDumbtrac = dumbtrac(simtime, light, clusters, lightphases, lastswitchtimes)
            if crossEntropyLoss:
                if (outputDumbtrac-0.25) < 0:
                    target = torch.LongTensor([0])
                else:
                    target = torch.LongTensor([1])
            else:
                target = torch.FloatTensor([outputDumbtrac-0.25]) # Target from expert
            nnin = convertToNNInputSurtrac(simtime, light, clusters, lightphases, lastswitchtimes)
        else:
            if crossEntropyLoss:
                if (bestschedule[7][0]-0.25) < 0:
                    target = torch.LongTensor([0])
                else:
                    target = torch.LongTensor([1])
            else:
                target = torch.FloatTensor([bestschedule[7][0]-0.25]) # Target from expert
            nnin = convertToNNInputSurtrac(simtime, light, clusters, lightphases, lastswitchtimes)

        if (testNN and (inRoutingSim or not noNNinMain)): #If NN
            trainingdata["light"].append((nnin, target, torch.tensor([[outputNN]])))
        else:
            nnin = convertToNNInputSurtrac(simtime, light, clusters, lightphases, lastswitchtimes, lightlanes)
            trainingdata["light"].append((nnin, target)) #Record the training data, but obviously not what the NN did since we aren't using an NN
            
            #Add all lanes from data augmentation - bad, overfits
            # for permlightlanes in dataAugmenter(lightlanes["light"]):
            #     templightlanes = dict()
            #     templightlanes["light"] = permlightlanes
            #     nnin = convertToNNInputSurtrac(simtime, light, clusters, lightphases, lastswitchtimes, templightlanes)
            #     trainingdata["light"].append((nnin, target)) #Record the training data, but obviously not what the NN did since we aren't using an NN
            
            #Add a random point from the data augmentation to try to learn robustness to lane permutations
            alldataaugment = dataAugmenter(lightlanes["light"])
            permlightlanes = alldataaugment[int(random.random()*len(alldataaugment))]
            templightlanes = dict()
            templightlanes["light"] = permlightlanes
            nnin = convertToNNInputSurtrac(simtime, light, clusters, lightphases, lastswitchtimes, templightlanes)
            trainingdata["light"].append((nnin, target)) #Record the training data, but obviously not what the NN did since we aren't using an NN
        
    
    if (testNN and (inRoutingSim or not noNNinMain)) or testDumbtrac:
        bestschedules[light] = testnnschedule

def main():
    global trainingdata
    try:
        with open("trainingdata/trainingdata_" + "IG" + ".pickle", 'rb') as handle:
            trainingdata = pickle.load(handle)
    except FileNotFoundError:
        print("Training data not found, starting fresh")
        for light in ["light"]:#lights:
            if multithreadSurtrac:
                trainingdata[light] = manager.list()#[]
            else:
                trainingdata[light] = []

    surtracThreads = dict()
    if multithreadSurtrac:
        nProcesses = multiprocessing.cpu_count()
    else:
        nProcesses = 1

    #print(multiprocessing.cpu_count())
            
    for i in range(nProcesses):
        print("Starting thread " + str(i))
        if multithreadSurtrac:
            #print("Starting vehicle routing thread")
            surtracThreads[i] = Process(target=loopIntersectionGenerator, args=(int(nruns/nProcesses),)) #Need the comma so args is a tuple not a single int
            surtracThreads[i].start()
        else:
            loopIntersectionGenerator(int(nruns/nProcesses))

    if multithreadSurtrac:
        for i in range(len(surtracThreads)):
            surtracThreads[i].join()
        #print(trainingdata)

    print("Saving training data")
    with open("trainingdata/trainingdata_" + "IG" + ".pickle", 'wb') as handle:
        pickle.dump(trainingdata, handle, protocol=pickle.HIGHEST_PROTOCOL)

def loopIntersectionGenerator(nLoops = 1):
    for i in range(nLoops):
        intersectionGenerator()

#lanelist = lightlanes["light"]
#Data augmentation: Permute lanes within each road, and order of roads
def dataAugmenter(lanelist):
    #Given a list of "road_lane" strings, want all reorderings, where we can change the order the roads appear and change the order within each road that the lanes appear
    #First, split to sublists by road
    listlist = []
    curRoad = None
    for lane in lanelist:
        if lane.split("_")[0] == curRoad:
            listlist[-1].append(lane)
        else:
            curRoad = lane.split("_")[0]
            listlist.append([lane])
    listlistlist = makeListListList(listlist)
    l3new = []
    for listlist in listlistlist:
        perms = itertools.permutations(listlist)
        for perm in perms:
            catlist = []
            for sublist in perm:
                catlist += sublist
            l3new.append(catlist)
    return l3new

def makeListListList(listlist):
    if len(listlist) == 0:
        return [[]] #TODO Check format here
    else:
        listlistlist = []
        perms = list(itertools.permutations(listlist[0]))
        if len(listlist) == 1:
            #Ex:
            #ll = [[1,2]]
            #perms = [[1,2],[2,1]]
            #Just return perms?
            #Think I'm missing some brackets: return [ [[1,2]], [[2,1]] ]
            #since I want a list of listlists
            for perm in perms:
                listlistlist.append([list(perm)])
        else:
            #Build stuff recursively. Ex:
            #ll = [[1,2], [3, 4]]
            #perms = [[1,2], [2,1]]
            #Take each thing in perms, and concat all the stuff from ll[1:]
            #lll = [[[1,2], [3, 4]], [[1,2],[4,3]], [[2,1],[3,4]], [[2,1],[4,3]]]
            for perm in perms:
                for utation in makeListListList(listlist[1:]):
                    listlistlist.append([list(perm)]+utation)
        return listlistlist

# this is the main entry point of this script
if __name__ == "__main__":
    main()
