import os
import sys
import runnerQueueSplit11Threaded
import pickle
import statistics
import matplotlib.pyplot as plt
import numpy as np

nIters = 1

try:
    with open("delaydataNoSurtrac_" + sys.argv[1] + ".pickle", 'rb') as handle:
        data = pickle.load(handle)
except:
    data = dict()
print(data)

plotdata = dict()
plotdata["All"] = []
plotdata["Adopters"] = []
plotdata["Non-Adopters"] = []
sddata = dict()
sddata["All"] = []
sddata["Adopters"] = []
sddata["Non-Adopters"] = []
for p in sorted(data.keys()):
    for w in ["All", "Adopters", "Non-Adopters"]:
        plotdata[w].append(statistics.mean(data[p][w]))
        if len(data[p][w]) > 1:
            sddata[w].append(statistics.stdev(data[p][w]))
        else:
            sddata[w].append(0)

p = sorted(data.keys())
print(plotdata)
plt.figure()
maxwidth = 0.1
for w in ["All", "Adopters", "Non-Adopters"]:
    #plt.plot(p, plotdata[w], label=w)
    x = np.array(p)
    y = np.array(plotdata[w])

    if w == "All":
        thickness = 0*x+maxwidth
    elif w == "Adopters":
        thickness = maxwidth*x
    else:
        thickness = maxwidth - maxwidth*x

    #plt.fill_between(x, y - thickness, y + thickness, label=w) #Should divide by cos(atan(slope))

    #Error bars
    plt.errorbar(x, y, linestyle='None', markersize = 10.0, capsize = 3.0, yerr=np.array(sddata[w]))

    #Very wrong. Above code adds thickness vertically, which isn't right for non-horizontal lines.
    #Below code tried to account for that, but to get the angle of the line on the plot, I need to know the axes...
    #Variable thickness - need duplicate points so I can correctly divide out by cos(atan(slope))
    m = (y[1:] - y[:-1])/(x[1:]-x[:-1]) #Slope = delta y / delta x
    m = m * (plt.gca().get_xlim()[1] - plt.gca().get_xlim()[0])/(plt.gca().get_ylim()[1] - plt.gca().get_ylim()[0]) #Account for axis sizes
    print("test")
    print(w)
    print(y)
    print(x)
    print(m)
    print("edntest")
    #Duplicate data except first and last points
    repx = np.repeat(x, 2)
    repx = repx[1:-1]
    repy = np.repeat(y, 2)
    repy = repy[1:-1]
    repm = np.repeat(m, 2)
    
    if w == "All":
        thickness = (0*repx+maxwidth)/np.cos(np.arctan(repm))
    elif w == "Adopters":
        thickness = (maxwidth*repx)/np.cos(np.arctan(repm))
    else:
        thickness = (maxwidth - maxwidth*repx)/np.cos(np.arctan(repm))

    plt.fill_between(repx, repy - thickness, repy + thickness, label=w) #Should divide by cos(atan(slope))

    

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