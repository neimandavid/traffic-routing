# traffic-routing

Current code is in surtrac_test, other folders are old versions or other tests

Main code file is runnerQueueSplit31Parallel.py, set testNNdefault to false if you want to run without needing a learned model for traffic lights. Command line arguments are the sumocfg file and the adoption probability. Dumps delay data as a pickle file to the delaydata folder (which is in surtrac_test)

trainNN4IS.py takes a sumocfg file and trains neural nets for all traffic lights (using the settings in runnerQueueSplit31Parallel to decide whether it's fixed timing plans, actuated control, Surtrac, etc) using DAgger. This is the intersection-specific version, meaning each traffic light gets its own neural net, because trying to make a single NN to handle any Surtrac case was too hard (couldn't get it to generalize from training data, and model size was getting large enough to negate any speedup we were trying to get); that aborted attempt is in trainNN4.py.

runnerDefaultWriter.py takes a sumocfg file and creates a bunch of files needed to run runnerQueueSplit31Parallel.py (specifically, edge-to-edge and lane-to-lane turn ratios for generating random routes for non-adopters, and yourconfigfilename_auto.sumocfg which unwraps flows into individual cars for computing delay). We used this to convert the synthetic and blocks demand data from a bunch of flows to individual vehicles

scenarioBuilder.py takes a text file of hourly demand data and builds a route.xml and sumocfg file for it, then calls runnerDefaultWriter; we used this to make the Pittsburgh demand data

AutoPlot.py grabs all the files in the delaydata folder, generates a bunch of plots, and stuffs the original delay data file and all the plots into a folder in Plots/AutoPlot. Autoplot.py is basically a wrapper for PlotWidthTest.py, which is the script to generate a single set of plots

For anyone trying to reproduce stuff, data from thesis/paper is in Plots/AutoPlot/ActuallyFinal. Pittsburgh sumocfg is PittsburghPMDataSmallerLongIn+15NewestesterYesLeftTurns_fixedroutes_auto.sumocfg; it and its associated .rou.xml were supposed to be 15% more vehicles than baseline demand, but apparently I messed up the arguments for scenarioBuilder and it's actually just standard baseline demand