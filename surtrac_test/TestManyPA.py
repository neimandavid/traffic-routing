import os
import sys
import runnerQueueSplit14 as runnerQueueSplit12
import pickle
import statistics
import matplotlib.pyplot as plt
from importlib import reload

nIters = 1

try:
    with open("delaydata/delaydata_" + sys.argv[1] + ".pickle", 'rb') as handle:
        data = pickle.load(handle)
except:
    #If no data found, start fresh
    data = dict()
#print(data[0.01]["All"])


for p in [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]*5:#[0.99]+[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]*4:#[0.01, 0.05, 0.25]*5:#[0.01, 0.05, 0.25, 0.5]*5:#[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]*5:
    reload(runnerQueueSplit12)
    print(p)
    if not p in data:
        data[p] = dict()
    for l in ["All", "Adopters", "Non-Adopters", "All2", "Adopters2", "Non-Adopters2", "All3", "Adopters3", "Non-Adopters3", "All0", "Adopters0", "Non-Adopters0", "RNGStates"]:
        if not l in data[p]:
            data[p][l] = []
    for i in range(0, nIters):
        print(i)
        [newdata, newrngstate] = runnerQueueSplit12.main(sys.argv[1], p, False)
        data[p]["All"].append(newdata[0])
        data[p]["Adopters"].append(newdata[1])
        data[p]["Non-Adopters"].append(newdata[2])
        data[p]["All2"].append(newdata[3])
        data[p]["Adopters2"].append(newdata[4])
        data[p]["Non-Adopters2"].append(newdata[5])
        data[p]["All3"].append(newdata[6])
        data[p]["Adopters3"].append(newdata[7])
        data[p]["Non-Adopters3"].append(newdata[8])
        data[p]["All0"].append(newdata[9])
        data[p]["Adopters0"].append(newdata[10])
        data[p]["Non-Adopters0"].append(newdata[11])
        data[p]["RNGStates"].append(newrngstate)
        with open("delaydata/delaydata_" + sys.argv[1] + ".pickle", 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # #Dump stats
        # for d in data:
        #     print(d)
        #     for label in ["All", "Adopters", "Non-Adopters", "All2", "Adopters2", "Non-Adopters2", "All3", "Adopters3", "Non-Adopters3", "All0", "Adopters0", "Non-Adopters0"]:
        #         print(label)
        #         if len(data[d][label]) > 1:
        #             print(str(statistics.mean(data[d][label])) + " +/- " + str(statistics.stdev(data[d][label])))
        #         else:
        #             print(str(statistics.mean(data[d][label])))

        plotdata = dict()
        for label in ["All", "Adopters", "Non-Adopters", "All2", "Adopters2", "Non-Adopters2", "All3", "Adopters3", "Non-Adopters3", "All0", "Adopters0", "Non-Adopters0"]:
            plotdata[label] = []
        for p in sorted(data.keys()):
            for w in ["All", "Adopters", "Non-Adopters", "All2", "Adopters2", "Non-Adopters2", "All3", "Adopters3", "Non-Adopters3", "All0", "Adopters0", "Non-Adopters0"]:
                plotdata[w].append(statistics.mean(data[p][w]))

        p = sorted(data.keys())
        #print(plotdata)
        plt.figure()
        for w in ["All", "Adopters", "Non-Adopters"]:
            plt.plot(p, plotdata[w], label=w)
        plt.xlabel("Adoption Probability")
        plt.ylabel("Average Delay (s)")
        plt.title("Delays")
        plt.legend()
        #plt.show() #NOTE: Blocks code execution until you close the plot
        plt.savefig("Plots/Delays" + sys.argv[1].split(".")[0] +".png")
        plt.close()

        plt.figure()
        for w in ["All2", "Adopters2", "Non-Adopters2"]:
            plt.plot(p, plotdata[w], label=w)
        plt.xlabel("Adoption Probability")
        plt.ylabel("Average Delay2 (s)")
        plt.title("Delays2")
        plt.legend()
        #plt.show() #NOTE: Blocks code execution until you close the plot
        plt.savefig("Plots/Delays2" + sys.argv[1].split(".")[0] +".png")
        plt.close()

        plt.figure()
        for w in ["All3", "Adopters3", "Non-Adopters3"]:
            plt.plot(p, plotdata[w], label=w)
        plt.xlabel("Adoption Probability")
        plt.ylabel("Average Delay3 (s)")
        plt.title("Delays3")
        plt.legend()
        #plt.show() #NOTE: Blocks code execution until you close the plot
        plt.savefig("Plots/Delays3" + sys.argv[1].split(".")[0] +".png")
        plt.close()

        plt.figure()
        for w in ["All0", "Adopters0", "Non-Adopters0"]:
            plt.plot(p, plotdata[w], label=w)
        plt.xlabel("Adoption Probability")
        plt.ylabel("Average Delay0 (s)")
        plt.title("Delays0")
        plt.legend()
        #plt.show() #NOTE: Blocks code execution until you close the plot
        plt.savefig("Plots/Delays0" + sys.argv[1].split(".")[0] +".png")
        plt.close()

for d in sorted(data.keys()):
    print(d)
    for label in ["All", "Adopters", "Non-Adopters"]:
        print(label)
        if len(data[d][label]) > 1:
            print(str(statistics.mean(data[d][label])) + " +/- " + str(statistics.stdev(data[d][label])))
        else:
            print(str(statistics.mean(data[d][label])))