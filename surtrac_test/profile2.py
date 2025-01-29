import pstats, cProfile
import re

import runnerQueueSplit27

try:
    cProfile.runctx("runnerQueueSplit27.main('PittsburghPMDataSmallerLongIn+10_fixedroutes_auto.sumocfg', 1)", globals(), locals(), "Profile.prof")
except:
    pass

s = pstats.Stats("Profile.prof")
s.strip_dirs().sort_stats("time").print_stats()