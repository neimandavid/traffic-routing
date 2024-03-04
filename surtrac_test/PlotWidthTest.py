import os
import sys
#import runnerQueueSplit11Threaded
import pickle
import statistics
import matplotlib.pyplot as plt
import numpy as np

nIters = 1

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


# #Split to groups of 5 runs in case of concatenation with old data:
# #Blindly assume same amount of data in all runs (or less in last p)
# nruns = len(data[list(data)[-1]]["All"])
# nfiles = nruns//5
# print(nfiles)
# for i in range(0, nfiles):
#     tempdict = dict()
#     for p in data:

#         tempdict[p] = dict()
#         for q in data[p]:
#             tempdict[p][q] = []
        
#             for j in range(0, 5):
#                 tempdict[p][q].append(data[p][q][5*i+j])
            
            
#     with open("delaydata/delaydata_" + sys.argv[1] + str(i) + ".pickle", 'wb') as handle:
#         pickle.dump(tempdict, handle, protocol=pickle.HIGHEST_PROTOCOL)
# asdf

labels = []
for x in ["", "2", "3", "0"]:
    for w in ["All", "Adopters", "Non-Adopters"]:
        labels.append(w+x)

for p in sorted(data.keys()): 
    if len(data[p]) >= 13:
        labels.append("Runtime")
        break
#labels = ["All", "Adopters", "Non-Adopters", "All2", "Adopters2", "Non-Adopters2"]
plotdata = dict()
for l in labels:
    plotdata[l] = []
sddata = dict()
for l in labels:
    sddata[l] = []
for p in sorted(data.keys()):
    for w in labels:
        if w in data[p]:
            plotdata[w].append(statistics.mean(data[p][w]))
            if len(data[p][w]) > 1:
                sddata[w].append(statistics.stdev(data[p][w]))
            else:
                sddata[w].append(0)

p = sorted(data.keys())
print(plotdata)

for v in ["", "2", "3", "0"]:
    fig, ax = plt.subplots()
    for w in ["All", "Adopters", "Non-Adopters"]:
        #plt.plot(p, plotdata[w], label=w)
        x = np.array(p)
        y = np.array(plotdata[w+v])

        #Error bars
        ax.errorbar(x, y, linestyle='None', markersize = 10.0, capsize = 3.0, yerr=np.array(sddata[w+v]))
        ax.axis([0, 1, 130, 250]) #To standardize axes
        
        maxwidth = (ax.get_ylim()[1] - ax.get_ylim()[0])/500.0 #0.1 #0.99#1.0#

        #Can't just add thickness vertically; this isn't right for non-horizontal lines.
        #Below code tried to account for that, but to get the angle of the line on the plot, I need to know the axes...
        #Variable thickness - need duplicate points so I can correctly divide out by cos(atan(slope))
        m = (y[1:] - y[:-1])/(x[1:]-x[:-1]) #Slope = delta y / delta x
        m = m * (plt.gca().get_xlim()[1] - plt.gca().get_xlim()[0])/(plt.gca().get_ylim()[1] - plt.gca().get_ylim()[0]) #Account for axis sizes

        #Duplicate data except first and last points
        repx = np.repeat(x, 2)
        repx = repx[1:-1]
        repy = np.repeat(y, 2)
        repy = repy[1:-1]
        repm = np.repeat(m, 2)
        
        if w == "All":
            thickness = (0*repx+maxwidth)/np.cos(np.arctan(repm)) #Should divide by cos(atan(slope)) to convert diagonal thickness to vertical thickness
        elif w == "Adopters":
            thickness = (maxwidth*repx)/np.cos(np.arctan(repm))
        else: #Non-Adopters
            thickness = (maxwidth - maxwidth*repx)/np.cos(np.arctan(repm))

        ax.fill_between(repx, repy - thickness, repy + thickness, label=w)

    #Text box code from: https://matplotlib.org/3.3.4/gallery/recipes/placing_text_boxes.html
    s = "Unknown stuff, help!"
    if v == "":
        s = "[time leaving] - [time entering] - [minimum route time]"
    if v == "2":
        s = "[time leaving] - [time at first reroute] - [minimum route time]"
    if v == "3":
        s = "[time leaving] - [time at first intersection] - [minimum route time]"
    if v == "0":
        s = "[time leaving] - [intended time entering] - [minimum route time]"
    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 1, s, transform = ax.transAxes, fontsize=8, verticalalignment='top', bbox=props)

    plt.xlabel("Adoption Probability")
    plt.ylabel("Average Delay (s)")
    plt.title("Delays"+v)
    plt.legend()
    #plt.show() #NOTE: Blocks code execution until you close the plot
    plt.savefig("Plots/PWDelays" + v + sys.argv[1].split(".")[0] +".png")
    plt.close()

#Plot runtime stuff
if "Runtime" in labels:
    fig, ax = plt.subplots()
    w = "Runtime"
    v = ""
    #plt.plot(p, plotdata[w], label=w)
    x = np.array(p)
    y = np.array(plotdata[w+v])

    #Error bars
    ax.errorbar(x, y, linestyle='None', markersize = 10.0, capsize = 3.0, yerr=np.array(sddata[w+v]))
    #ax.axis([0, 1, 130, 250]) #To standardize axes

    maxwidth = (ax.get_ylim()[1] - ax.get_ylim()[0])/500.0 #0.1 #0.99#1.0#

    #Can't just add thickness vertically; this isn't right for non-horizontal lines.
    #Below code tried to account for that, but to get the angle of the line on the plot, I need to know the axes...
    #Variable thickness - need duplicate points so I can correctly divide out by cos(atan(slope))
    m = (y[1:] - y[:-1])/(x[1:]-x[:-1]) #Slope = delta y / delta x
    m = m * (plt.gca().get_xlim()[1] - plt.gca().get_xlim()[0])/(plt.gca().get_ylim()[1] - plt.gca().get_ylim()[0]) #Account for axis sizes

    #Duplicate data except first and last points
    repx = np.repeat(x, 2)
    repx = repx[1:-1]
    repy = np.repeat(y, 2)
    repy = repy[1:-1]
    repm = np.repeat(m, 2)

    thickness = (0*repx+maxwidth)/np.cos(np.arctan(repm)) #Should divide by cos(atan(slope)) to convert diagonal thickness to vertical thickness
    ax.fill_between(repx, repy - thickness, repy + thickness, label=w)

    #Text box code from: https://matplotlib.org/3.3.4/gallery/recipes/placing_text_boxes.html
    s = "Unknown stuff, help!"
    if v == "":
        s = "[time leaving] - [time entering] - [minimum route time]"
    if v == "2":
        s = "[time leaving] - [time at first reroute] - [minimum route time]"
    if v == "3":
        s = "[time leaving] - [time at first intersection] - [minimum route time]"
    if v == "0":
        s = "[time leaving] - [intended time entering] - [minimum route time]"
    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 1, s, transform = ax.transAxes, fontsize=8, verticalalignment='top', bbox=props)

    plt.xlabel("Adoption Probability")
    plt.ylabel("Average Delay (s)")
    plt.title(w)
    plt.legend()
    #plt.show() #NOTE: Blocks code execution until you close the plot
    plt.savefig("Plots/PW" + w + v + sys.argv[1].split(".")[0] +".png")
    plt.close()

for d in sorted(data.keys()):
    print(d)
    print(str(len(data[d][labels[0]])) + " runs")
    for label in labels:
        print(label)
        if len(data[d][label]) > 1:
            print(str(statistics.mean(data[d][label])) + " +/- " + str(statistics.stdev(data[d][label])))
        else:
            print(str(statistics.mean(data[d][label])))