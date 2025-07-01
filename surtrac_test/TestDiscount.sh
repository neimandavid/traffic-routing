#!/bin/bash

echo "Test script"

for n in {1..5}
do
for d in 0 1 0.5 0.1 0.7 0.3 0.2 0.4 0.6 0.8 0.9;
do

python3 runnerQueueSplit27IntersectionSpecificDiscountTestGood.py PittsburghPMDataSmallerLongInNewestester_fixedroutes_auto.sumocfg 0.01 False False $d
done
done