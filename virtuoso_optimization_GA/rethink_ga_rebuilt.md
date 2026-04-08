We need to rethink how GA works and rebuilt the ga algorith.

Based on the current SA updates we have updated several degree of freedom
1. added switchable inductors
2. added series caps pairs (we don't need to think about this right now)

Therefore we need update the solution to 1. This will be similar to how switches works right now. It's just we need to keep the inductor switch states consistent after mergeing. Basically if I merge two designs, if switch 0 of design 1 is kept, then it's switch states shoudn't change. I think the regular switches are already kind of doing this in GA, so we just need to make inductors also work in this way.

We dont' implement series caps pairs in GA right now.

Algorithm side, one insight might be one of the design might be good at certain frequency, and another design might be good at another frequency. In the evolution, can we keep the design that might be best for each state (on top of keeping best overall deisgn, for each state we keep in the best state wise design in the herd). Therefore we can hope to have better hybrid results hoping one will contribute some and another parents contribute to other good states.

Please redesign the GA algorithm to best utilize these insights and provide a plan.