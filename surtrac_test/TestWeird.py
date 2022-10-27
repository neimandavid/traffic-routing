import os
import sys
import runnerQueueSplit13 as runnerQueueSplit13
import pickle
import statistics
from importlib import reload
import random

try:
    with open("delaydata/delaydata_" + sys.argv[1] + ".pickle", 'rb') as handle:
        data = pickle.load(handle)
except:
    #If no data found, start fresh
    data = dict()
#print(data)

p = 0.01
ind = -1
with open("lastRNGstate.pickle", 'rb') as handle:
    rngstate = pickle.load(handle)  #data[p]["RNGStates"][ind]
random.setstate(rngstate)
[newdata, newrngstate] = runnerQueueSplit13.main(sys.argv[1], p, False)

print(newdata)
assert(newrngstate == rngstate) #Make sure we did in fact start with the correct RNG state