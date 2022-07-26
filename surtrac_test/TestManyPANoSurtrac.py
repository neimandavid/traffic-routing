import os
import sys
import runnerQueueSplit11ThreadedNoSurtrac
import pickle
import statistics
import matplotlib.pyplot as plt

nIters = 1

try:
    with open("delaydataNoSurtrac_" + sys.argv[1] + ".pickle", 'rb') as handle:
        data = pickle.load(handle)
except:
    data = dict()
print(data)

for p in [0.01, 0.05, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]*3: #[0.99] + [0.01, 0.75, 0.9, 0.95, 0.99]*3:#[0.01, 0.05, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]*1:
    print(p)
    if not p in data:
        data[p] = dict()
        data[p]["All"] = []
        data[p]["Adopters"] = []
        data[p]["Non-Adopters"] = []
    for i in range(0, nIters):
        print(i)
        newdata = runnerQueueSplit11ThreadedNoSurtrac.main(sys.argv[1], p, False)
        data[p]["All"].append(newdata[0])
        data[p]["Adopters"].append(newdata[1])
        data[p]["Non-Adopters"].append(newdata[2])
        with open("delaydataNoSurtrac_" + sys.argv[1] + ".pickle", 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        #Dump stats
        for d in data:
            print(d)
            for label in ["All", "Adopters", "Non-Adopters"]:
                print(label)
                if len(data[d][label]) > 1:
                    print(str(statistics.mean(data[d][label])) + " +/- " + str(statistics.stdev(data[d][label])))
                else:
                    print(str(statistics.mean(data[d][label])))

        plotdata = dict()
        plotdata["All"] = []
        plotdata["Adopters"] = []
        plotdata["Non-Adopters"] = []
        for p in sorted(data.keys()):
            for w in ["All", "Adopters", "Non-Adopters"]:
                plotdata[w].append(statistics.mean(data[p][w]))

        p = sorted(data.keys())
        print(plotdata)
        plt.figure()
        for w in ["All", "Adopters", "Non-Adopters"]:
            plt.plot(p, plotdata[w], label=w)
        plt.xlabel("Adoption Probability")
        plt.ylabel("Average Delay (s)")
        plt.title("Delays")
        plt.legend()
        #plt.show() #NOTE: Blocks code execution until you close the plot
        plt.savefig("Plots/DelaysNoSurtrac" + sys.argv[1].split(".")[0] +".png")
        plt.close()

for d in sorted(data.keys()):
    print(d)
    for label in ["All", "Adopters", "Non-Adopters"]:
        print(label)
        if len(data[d][label]) > 1:
            print(str(statistics.mean(data[d][label])) + " +/- " + str(statistics.stdev(data[d][label])))
        else:
            print(str(statistics.mean(data[d][label])))