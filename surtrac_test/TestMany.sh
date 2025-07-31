#!/bin/bash

set -e #Error out if something goes wrong, useful for debug

echo "Moving previous data to backup folder"
mkdir -p Backup
NOW=$( date '+%F_%H:%M:%S' )
mkdir Backup/$NOW
[ -f delaydata/* ] mv delaydata/* Backup/$NOW

echo "Running code"
for n in {1..5};
do
for p in 0.01 0.05 0.25 0.5 0.75 0.95 0.99;
do

#Run everything!
python3 runnerQueueSplit27IntersectionSpecificGood.py blocks51hNewestester_auto.sumocfg $p
python3 runnerQueueSplit27IntersectionSpecificGood.py PittsburghPMDataSmallerLongIn+15NewestesterYesLeftTurns_fixedroutes_auto.sumocfg $p
python3 runnerQueueSplit27IntersectionSpecificGood.py shortlongshort4_auto.sumocfg $p

done
done
