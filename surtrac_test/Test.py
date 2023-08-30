import threading
import time

def readThread():
    global superDict
    myDict = dict()
    timestep = 0
    while timestep < 5:
        while not timestep in superDict:
            time.sleep(0)
        newdata = superDict[timestep]
        myDict.update(newdata)

        print("Read, time = " + str(timestep) + ", superDict = " + str(superDict) + ", myDict = " + str(myDict))
        timestep += 1

def writeThread():
    global superDict
    superDict = dict()
    
    myDict = dict()
    timestep = 0
    while timestep < 5:
        superDict[timestep] = dict()
        superDict[timestep]["key"] = timestep**2
        while not timestep in superDict:
            time.sleep(0)
        newdata = superDict[timestep]
        myDict.update(newdata)

        print("Write, time = " + str(timestep) + ", superDict = " + str(superDict) + ", myDict = " + str(myDict))
        timestep += 1
        time.sleep(0.0001)
        
global superDict
superDict = dict()

mythreads = dict()

mythreads["read"] = threading.Thread(target=readThread)
mythreads["read"].start()

mythreads["write"] = threading.Thread(target=writeThread)
mythreads["write"].start()