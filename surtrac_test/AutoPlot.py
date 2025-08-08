import os
import sys
import pickle
import statistics
import matplotlib.pyplot as plt
import numpy as np
import time

filterNonzeroTeleports = False


directory = "delaydata/"

hasFiles = False
for filename in os.listdir(directory):
    if not filename.endswith(".pickle"):
        continue
    else:
        hasFiles = True
        break
    
if not hasFiles:
    print("No data in " + directory + "???")
    exit(0)

if not os.path.isdir("Plots/AutoPlot"):
    os.mkdir("Plots/AutoPlot")
presavedir = "Plots/AutoPlot/"+str(time.time())
os.mkdir(presavedir)

for filename in os.listdir(directory):
    filepath = directory+filename
    
    if not filepath.endswith(".pickle"):
        os.rename(filepath, savedir+"/"+filename)
        continue
    print(filepath)

    shortfilename = filename[10:].split(".")[0] #Drop the leading "delaydata_" and the file extensions

    savedir = presavedir + "/" + shortfilename
    os.mkdir(savedir)
    
    try:
        with open(filepath, 'rb') as handle:
            data = pickle.load(handle)

    except Exception as e:
        print("Data not found")
        data = dict()
        raise(e)

    if filterNonzeroTeleports:
        for p in data:
            i = 0
            while i < len(data[p]["NTeleports"]):
                if data[p]["NTeleports"][i] > 0:
                    for label in data[p]:
                        data[p][label].pop(i)
                else:
                    i+=1

    for p in data:
        print(p)
        print("Delay")
        print(data[p]["All"])
        #print(data[p]["All2"])
        if "NTeleports" in data[p]:
            print("NTeleports")
            print(data[p]["NTeleports"])
        if "TeleportData" in data[p]:
            print("TeleportData")
            print(data[p]["TeleportData"])

    labels = []
    for x in ["", "2", "3", "0"]:
        for w in ["All", "Adopters", "Non-Adopters"]:
            labels.append(w+x)

    for p in sorted(data.keys()): 
        if len(data[p]) >= 14 and not "Runtime" in labels:
            labels.append("Runtime")
        if len(data[p]) >= 15 and not "NTeleports" in labels:
            labels.append("NTeleports")

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


    for v in ["", "2", "3", "0"]:
        fig, ax = plt.subplots()
        for w in ["All", "Adopters", "Non-Adopters"]:
            #plt.plot(p, plotdata[w], label=w)
            x = np.array(p)
            y = np.array(plotdata[w+v])

            #Error bars
            ax.errorbar(x, y, linestyle='None', markersize = 10.0, capsize = 3.0, yerr=np.array(sddata[w+v]))
            #ax.axis([0, 1, 150, 330]) #To standardize axes
            
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
        ax.text(0, 1.1, shortfilename, transform = ax.transAxes, fontsize=8)

        plt.xlabel("Adoption Probability")
        plt.ylabel("Average Delay (s)")
        plt.title("Delays"+v)
        plt.legend()
        #plt.show() #NOTE: Blocks code execution until you close the plot
        plt.savefig(savedir+"/PWDelays" + v + shortfilename +".png")
        plt.close()

    #Plot runtime stuff
    for w in ["Runtime", "NTeleports"]:
        if w in labels:
            fig, ax = plt.subplots()
            #w = "Runtime"
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
            ax.text(0, 1.1, shortfilename, transform = ax.transAxes, fontsize=8)

            plt.xlabel("Adoption Probability")
            plt.ylabel(w)
            plt.title(w)
            plt.legend()
            #plt.show() #NOTE: Blocks code execution until you close the plot
            plt.savefig(savedir+"/PW" + w + v + shortfilename +".png")
            plt.close()
        else:
            print("Couldn't find data " + w)

    for d in sorted(data.keys()):
        print(d)
        print(str(len(data[d][labels[0]])) + " runs")
        for label in labels:
            print(label)
            if len(data[d][label]) > 1:
                print(str(statistics.mean(data[d][label])) + " +/- " + str(statistics.stdev(data[d][label])))
            else:
                print(str(statistics.mean(data[d][label])))


    #MAKE PLOTS WITHOUT TELEPORT POINTS
    prs = data.keys()
    prstoremove = []
    if "NTeleports" in labels:
        for pr in prs:
            n = 0
            while n < len(data[pr]["NTeleports"]):
                if data[pr]["NTeleports"][n] > 0:
                    for label in labels:
                        data[pr][label].pop(n)
                else:
                    n+=1
            if n == 0: #Ate all the data, oops
                prstoremove.append(pr)
        for pr in prstoremove:
            del data[pr]

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

    for v in ["", "2", "3", "0"]:
        fig, ax = plt.subplots()
        for w in ["All", "Adopters", "Non-Adopters"]:
            #plt.plot(p, plotdata[w], label=w)
            x = np.array(p)
            y = np.array(plotdata[w+v])

            #Error bars
            ax.errorbar(x, y, linestyle='None', markersize = 10.0, capsize = 3.0, yerr=np.array(sddata[w+v]))
            #ax.axis([0, 1, 150, 330]) #To standardize axes
            
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
        ax.text(0, 1.1, shortfilename, transform = ax.transAxes, fontsize=8)

        plt.xlabel("Adoption Probability")
        plt.ylabel("Average Delay (s)")
        plt.title("No Teleport Delays"+v)
        plt.legend()
        #plt.show() #NOTE: Blocks code execution until you close the plot
        plt.savefig(savedir+"/PWDelays" + v + shortfilename +"NoTeleports.png")
        plt.close()

    #Plot runtime stuff
    for w in ["Runtime", "NTeleports"]:
        if w in labels:
            fig, ax = plt.subplots()
            #w = "Runtime"
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
            ax.text(0, 1.1, shortfilename, transform = ax.transAxes, fontsize=8)

            plt.xlabel("Adoption Probability")
            plt.ylabel(w)
            plt.title("No Teleport " + w)
            plt.legend()
            #plt.show() #NOTE: Blocks code execution until you close the plot
            plt.savefig(savedir+"/PW" + w + v + shortfilename +"NoTeleports.png")
            plt.close()
        else:
            print("Couldn't find data " + w)

    #And finally, dump the data into the plot folder in case we need to reproduce later
    os.rename(filepath, savedir+"/"+filename)