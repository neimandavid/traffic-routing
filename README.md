# traffic-routing

Dumping my current traffic routing code here, in preparation for class projects, etc.
Uses Sumo as the traffic simulator: https://www.eclipse.org/sumo/

Current exploratory work in shortlong2:
- runnerAutoroute uses Sumo's default routing (Djikstra, edge weights based on current average speed), no rerouting). Alternates between sending too much traffic down each route due to lag on time estimates (current avg. speed isn't a good indicator of future travel time, leads to overreaction).
- runner is probably shortest path with avg. speed edge weights, rerouting at each intersection. Not much better
- runnerRandom routes cars randomly at each intersection, and does surprisingly well
- runner Braess removes some routes that were user optimal but not system optimal. Best overall performance, but the process wasn't automated at all
- runnerPredictive is hopefully going to use better travel time estimates based on queue models. WIP

Future work:
15888 project: Converge to some form of equilibrium using CFR (counterfactual regret minimization)
16811 project: ??? (planning through time with queue model?)