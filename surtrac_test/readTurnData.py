import pickle
import os
import sys
import optparse

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

sumoconfig = sys.argv[1]
(netfile, routefile) = readSumoCfg(sumoconfig)
with open("Lturndata_"+routefile.split(".")[0]+".pickle", 'rb') as handle:
    turndata = pickle.load(handle)
    print(turndata)

    print("")
    print(turndata["10_1"])