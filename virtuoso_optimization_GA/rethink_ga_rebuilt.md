We need to rethink how GA works and rebuilt the ga algorith.

Based on the current SA updates we have updated several degree of freedom
1. added switchable inductors
2. added series caps pairs (we don't need to think about this right now)

Therefore we need update the solution to 1. This will be similar to how switches works right now. It's just we need to keep the inductor switch states consistent after mergeing. Basically if I merge two designs, if switch 0 of design 1 is kept, then it's switch states shoudn't change. I think the regular switches are already kind of doing this in GA, so we just need to make inductors also work in this way.

We dont' implement series caps pairs in GA right now.

Algorithm side, one insight might be one of the design might be good at certain frequency, and another design might be good at another frequency. In the evolution, can we keep the design that might be best for each state (on top of keeping best overall deisgn, for each state we keep in the best state wise design in the herd). Therefore we can hope to have better hybrid results hoping one will contribute some and another parents contribute to other good states.

Please redesign the GA algorithm to best utilize these insights and provide a plan.


### Newer update
loss function (cost function): change L1 loss to L2 loss.



Larger Initial population & population count scheduling.
To keep better genetic diversity, we start with a larger population size ~100, and a larger ellite group to keep, we can gradually cool down to 20 like: 100 -90 80 70 .. decrement by 10. And the ellite group count will also be kept to certain values. 

Best design exploitation. In each generation, before we hybrid and mutate, we can make a exploitation step. Basically we take the best design of this generation, and apply some special mutation change (action), with some unique new weights (separate from the current weight). Since switch states are important, we can make the default eploitation mutation weight as only tweaking the switch status, not affecting other part of the designs.Then we will fire by default 5 such kind of mutations to simulation and add to the pool.


### New update in response of simulator engine update 3

These description are mainly for GA but should be easy to adapt to SA. Please plan for the changes for GA and SA.

This update includes updating the simulator to support new pixel type: via ground. This should be managed similar to inductor but more simple. During initial generation, we filp several pixels to be vias and constrain by a initial count. 

After that we will have some action to flip this type of pixel. In this new action we will chose two numbers randomly: vias to remove and vias to add (lets say between 0 to 3) Then we will randomly remove the corresponding number of via, and randomly flip some non-via pixels to be vias.

The fix method after hybrid must consider if there are more than the max number of vias allowed. 

We also require spacing constraints where the surrounding 8 pixels can only be 0 and 1.

notice we include 3 hyperparameters for vias. n_max n_init and actions flip max count (name them better)


 
### Cost function update

This cost function aim to combine the S21 with S11 cost. For a lossy network optimizing S21 might not be efficient. We want to add the S11 optimization into the loss function. In S11 we want the pass band to go to negative infinity as much as possible (no reflection so we can aim a lower target like-30db). And stop band and cross bands to be closs to zero (as much reflection) . Please implement this cost using a similar way to the current s21 cost and add that to the s21. For each state. We can add S21 + S11 cost, reusing the same weight defined.
