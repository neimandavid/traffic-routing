import os
import sys
#import runnerQueueSplit19SUMOEverywhere as runnerQueueSplit12 #Change this line if using a newer code version
#import runnerQueueSplit19Ben as runnerQueueSplit12 #Change this line if using a newer code version
#import runnerQueueSplit18NN as runnerQueueSplit12 #Change this line if using a newer code version
#import runnerQueueSplit19SUMOEverywhereGC as runnerQueueSplit12 #Change this line if using a newer code version
import runnerQueueSplit23 as runnerQueueSplit12 #Change this line if using a newer code version

import pickle
import statistics
import matplotlib.pyplot as plt
from importlib import reload

import os
import sys
#import runnerQueueSplit11Threaded
import pickle
import statistics
import matplotlib.pyplot as plt
import numpy as np

nIters = 1
filterNonzeroTeleports = False

try:
    with open("delaydata/delaydata_" + sys.argv[1] + ".pickle", 'rb') as handle:
        data = pickle.load(handle)

        #Grab first 5 runs from data
        # for p in data:
        #     for q in data[p]:
        #         data[p][q] = data[p][q][0:5]
        # with open("delaydata/delaydata_" + sys.argv[1] + "new.pickle", 'wb') as handle:
        #     pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    for i in range(2, len(sys.argv)): #If given multiple files, grab all of them
        print("delaydata/delaydata_" + sys.argv[i] + ".pickle")
        with open("delaydata/delaydata_" + sys.argv[i] + ".pickle", 'rb') as handle:
            newdata = pickle.load(handle)
        for p in newdata:
            #print(p)
            if p in data:
                #data[p].append(newdata[p])
                for q in data[p]:
                    data[p][q] = data[p][q] + newdata[p][q]
            else:
                data[p] = newdata[p]
except Exception as e:
    print("Data not found")
    data = dict()
    raise(e)
#print(data)

filteredData = dict()

for p in data:
    print(p)
    for q in range(len(data[p]["NTeleports"])):
        print(q)
        if(data[p]["NTeleports"][q] == 0):
            print("No teleports")
        else:
            
            reload(runnerQueueSplit12)

            #Dump old RNG state to reuse
            rngstate = data[p]["RNGStates"][q]
            with open("lastRNGstate.pickle", 'wb') as handle:
                pickle.dump(rngstate, handle, protocol=pickle.HIGHEST_PROTOCOL)
                
            [newdata, newrngstate] = runnerQueueSplit12.main(sys.argv[1], p, False, True, False)

            
            if not p in filteredData:
                filteredData[p] = dict()
            for l in ["All", "Adopters", "Non-Adopters", "All2", "Adopters2", "Non-Adopters2", "All3", "Adopters3", "Non-Adopters3", "All0", "Adopters0", "Non-Adopters0", "Runtime", "NTeleports", "TeleportData", "RNGStates"]:
                if not l in filteredData[p]:
                    filteredData[p][l] = []

            filteredData[p]["All"].append(newdata[0])
            filteredData[p]["Adopters"].append(newdata[1])
            filteredData[p]["Non-Adopters"].append(newdata[2])
            filteredData[p]["All2"].append(newdata[3])
            filteredData[p]["Adopters2"].append(newdata[4])
            filteredData[p]["Non-Adopters2"].append(newdata[5])
            filteredData[p]["All3"].append(newdata[6])
            filteredData[p]["Adopters3"].append(newdata[7])
            filteredData[p]["Non-Adopters3"].append(newdata[8])
            filteredData[p]["All0"].append(newdata[9])
            filteredData[p]["Adopters0"].append(newdata[10])
            filteredData[p]["Non-Adopters0"].append(newdata[11])
            filteredData[p]["Runtime"].append(newdata[12])
            filteredData[p]["NTeleports"].append(newdata[13])
            filteredData[p]["TeleportData"].append(newdata[14])
            
            filteredData[p]["RNGStates"].append(newrngstate)
            with open("delaydata/delaydatateleportstuff_" + sys.argv[1] + ".pickle", 'wb') as handle:
                pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

for p in sorted(filteredData.keys()):
    print(p)
    for q in range(len(filteredData[p]["TeleportData"])):
        print(filteredData[p]["NTeleports"][q])
        print(filteredData[p]["TeleportData"][q])