#!/bin/bash

echo "Test script"

for n in {1..5};
do
for p in 0.01 0.05 0.25 0.5 0.75 0.95 0.99;
do

python3 runnerQueueSplit29IntersectionSpecific.py PittsburghPMDataSmallerLongIn+10_auto.sumocfg $p
done
done
