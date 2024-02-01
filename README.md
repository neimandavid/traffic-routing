# traffic-routing

Current code is in surtrac_test, other folders are old versions

Main code file is runnerQueueSplit19SUMOEverywhereGC.py, set testNNdefault to run without needing a learned model for traffic lights. Command line arguments are the sumocfg file and the adoption probability

runnerDefaultWriter.py takes a sumocfg file and creates a bunch of files needed to run runnerQueueSplit19SUMOEverywhereGC (specifically, edge-to-edge and lane-to-lane turn ratios for generating random routes for non-adopters, and yourconfigfile_auto.sumocfg which unwraps flows into individual cars for computing delay)

trainNN2.py takes a sumocfg file and trains neural nets for all traffic lights (using the settings in runnerQueueSplit19SUMOEverywhereGC to decide whether it's fixed timing plans, actuated control, Surtrac, etc) using DAgger

scenarioBuilder.py takes a text file of hourly demand data and builds a route.xml file for it
