import pickle

with open("delaydata/delaydata_PittsburghPMDataSmallerLongIn+10New_fixedroutes_auto.sumocfg.pickle", 'rb') as handle:
    temp = pickle.load(handle)
    print(temp[0.25]["All"][3])
    print(temp[0.25]["NTeleports"][3])
    #print(temp[0.25]["RNGStates"])
    rngstate = temp[0.25]["RNGStates"][3]

with open("lastRNGstate.pickle", 'wb') as handle:
    pickle.dump(rngstate, handle, protocol=pickle.HIGHEST_PROTOCOL)