#!/bin/bash

echo "Test script"

for p in 0.01 0.05 0.25 0.5 0.75 0.95 0.99;
do
for n in {1..5};
do

python3 runnerQueueSplit27IntersectionSpecific.py PittsburghPMDataSmallerLongIn+15Newestest_fixedroutes_auto.sumocfg $p
done
done
