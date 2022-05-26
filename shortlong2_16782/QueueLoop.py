import os
import sys
import runnerQueueSplit4PA
import pickle
import statistics

try:
    with open("statsdata_" + sys.argv[1] + ".pickle", 'rb') as handle:
        data = pickle.load(handle)
except:
    data = dict()
print(data)

for p in [0.99]:# [0.0, 0.01, 0.1, 0.15, 0.25, 0.5, 0.75, 1.0]:
    print(p)
    data[p] = dict()
    data[p]["All"] = []
    data[p]["Adopters"] = []
    data[p]["Non-Adopters"] = []
    for i in range(0, 20):
        print(i)
        newdata = runnerQueueSplit4PA.main(sys.argv[1], p, False)
        data[p]["All"].append(newdata[0])
        data[p]["Adopters"].append(newdata[1])
        data[p]["Non-Adopters"].append(newdata[2])
    for d in data:
        print(d)
        for label in ["All", "Adopters", "Non-Adopters"]:
            print(label)
            print(str(statistics.mean(data[d][label])) + " +/- " + str(statistics.stdev(data[d][label])))
    with open("statsdata_" + sys.argv[1] + ".pickle", 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

print(data)
for d in data:
    print(d)
    for label in ["All", "Adopters", "Non-Adopters"]:
        print(label)
        print(str(statistics.mean(data[d][label])) + " +/- " + str(statistics.stdev(data[d][label])))