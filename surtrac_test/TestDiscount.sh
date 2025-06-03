#!/bin/bash

echo "Test script"

for n in 1..5
do
for d in 0 0.5 0.1 1 0.2 0.3 0.35 0.4 0.45 0.55 0.6 0.7 0.8;
do

python3 runnerQueueSplit27IntersectionSpecificDiscountTest.py PittsburghPMDataSmallerLongIn+15Newestest_fixedroutes_auto.sumocfg 0.01 False False $d
done
done