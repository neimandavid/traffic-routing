#!/usr/bin/env python
#Adapted from TraCI tutorial here: https://github.com/eclipse/sumo/blob/main/tests/complex/tutorial/traci_tls/runner.py

#QueueSplit5 added a first iteration of a Surtrac model to QueueSplit4
#New in QueueSplit6: When there's multiple light phases a lane can go in, don't double-create clusters
#New in QueueSplit7: Adding Surtrac to simulate-ahead
#New in QueueSplit8: Adding communication between intersections. Which then requires re-compacting clusters, etc. For now, don't take advantage of known routes
#New in QueueSplit9: Take advantage of known routes from vehicle routing (NOTE: route may change as vehicle approaches intersection...)
#New in QueueSplit10: Optimize for speed (calling Surtrac less often in routing, merging predicted clusters)
#New in QueueSplit11: Compute delay, not just average travel time, for cars. Also fixed a bug with Surtrac code DP (was removing sequences it shouldn't have)
#QueueSplit12: Multithread the Surtrac code (it's really slow otherwise). Also, use the full Surtrac schedule rather than assuming we'll update every timestep
#QueueSplit13: Surtrac now (correctly) no longer overwrites all the finish times of other lanes with the start time of the currently scheduled cluster (leads to problems when a long cluster, then compatible short cluster, get scheduled, as the next cluster can then start earlier than it should). VOI now gets split into all lanes on starting edge
#14: Anytime routing, better stats on timeouts and teleports, added mingap to all cluster durations, using a timestep that divides mingap and surtracFreq, opposing traffic blocks for mingap not just one timestep, combining clusters at lights into a single queue
#15: Surtrac now runs once, assuming all vehicles travel their same routes, and then gets looked up during routing simulations. Also adding a flag to disable predicting clusters with Surtrac. REAL-TIME PERFORMANCE!!!
#16: Be lazy in routing simulations - don't need to check a lane until the front vehicle gets close to the end. (Didn't seem to give much speedup though.) Reusing Surtrac between multiple timesteps (and backported this to 15). Various attempts at cleanup and speedup.
#17: Move computation of Surtrac's predicted outflows to the end, after we already know what the schedule should be, rather than building it into the algorithm and generating predicted outflows for schedules we won't end up using. Fixed a bug where initially splitting the VOI into all lanes added it at the modified start time, and possibly in the wrong place
#18: Add imitation learning version of Surtrac
#19: Going back to A* in SUMO since ghost cars are causing problems. Telling all ghost cars to turn left, and spawning left-turning ghost cars when the turn completes so we account for oncoming traffic
#20: Storing Surtrac results for reuse
#21: Libsumo (optional hopefully), multithreaded routing, save files under different names so we can run two sets of code without making one error
#22: Did routing before Surtrac for more realism (in case we're not simultaneous so Surtrac's out of date). Not much effect there.
#23: 21, but now using a detector model to reconstruct the traffic state at the start of each routing simulation
#24: Detector model stops tracking specific names of non-adopters
#25: New plan for lane changes - blindly sample which lane stuff ends up in
#26: Detector model for Surtrac in routing as well (since the goal is to approximate what the main simulation would be doing)
#27: Support new SurtracNet (single network for all intersections, takes in intersection geometry and light phase info)

from __future__ import absolute_import
from __future__ import print_function

import sys
import os

# we need to import python modules from the $SUMO_HOME/tools directory
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")
from sumolib import checkBinary

useLibsumo = True
if useLibsumo:
    try:
        import libsumo as traci
    except:
        print("Error using libsumo. Dropping back to traci instead. Hopefully this is fine")
        useLibsumo = False
if not useLibsumo:
    import traci  #To interface with SUMO simulations

def run():
    simtime = 0

    while traci.simulation.getMinExpectedNumber() > 0 :
        simtime += 1
        traci.simulationStep() #Tell the simulator to simulate the next time step

        # #Plot and print stats
        if simtime%100 == 0 or not traci.simulation.getMinExpectedNumber() > 0:
            print("\nCurrent simulation time: %f" % simtime)
                
def main(sumoconfig, verbose = True):

    # this script has been called from the command line. It will start sumo as a
    # server, then connect and run

    #sumoBinary = checkBinary('sumo')
    sumoBinary = checkBinary('sumo-gui')
    #NOTE: Script name is zeroth arg

    # this is the normal way of using traci. sumo is started as a
    # subprocess and then the python script connects and runs
    if useLibsumo:
        print("Using libsumo, yay!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        traci.load(["-c", sumoconfig,
                                #"--additional-files", "additional_autogen.xml",
                                "--no-step-log", "true",
                                "--log", "LOGFILE", "--xml-validation", "never", "--start", "--quit-on-end"])
    else:
        traci.start([sumoBinary, "-c", sumoconfig,
                                #"--additional-files", "additional_autogen.xml",
                                "--no-step-log", "true",
                                "--log", "LOGFILE", "--xml-validation", "never", "--start", "--quit-on-end"], label="main")

    outdata = run()

# this is the main entry point of this script
if __name__ == "__main__":
    main(sys.argv[1])
