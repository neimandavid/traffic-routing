#!/bin/bash

set -e
echo "Test script"

for n in {1..5};
do
for p in 0.01 0.05 0.25 0.5 0.75 0.95 0.99;
do

#Run everything!
python3 runnerQueueSplit30.py blocks51hNewestester_auto.sumocfg $p
python3 runnerQueueSplit30.py PittsburghPMDataSmallerLongIn+15NewestesterYesLeftTurns_fixedroutes_auto.sumocfg $p
python3 runnerQueueSplit30.py shortlongshort4_auto.sumocfg $p

done
done
