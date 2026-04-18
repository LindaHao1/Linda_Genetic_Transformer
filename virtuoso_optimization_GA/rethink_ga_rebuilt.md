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


# updates

we want to save the best design for each state (indicies) (configuration).
Whenever we have a best design, can we record how it came? The action led to it? or the hybird, who hybrid this design, how?

Design optimization constraints: We want to have an option to enable a single inductor value during optimization. All the inductors will share the same inductor value during initialization and optimization. And when we flip and change inductor values, we have to ensure all inductor value changes. In the hybrid. If this optin is turned on, it will randomly inherit one of the parent inductor values if this option is turned on.

The above canges are done

# I want to make the following update to the ga

1. add back the constraint that the caps pairs must be horizontal or vertical

2. add an optional parameter in flipping pixels. This devide the pixel design into the inside and outside, defined by top left and bottom right index tuples. Within this range (middle of the pixels), we define a lower probability of flipping 0/1, and outside has a higher probability of flipping 0/1. This is useful in finetuning where we put a small design into a larger embedded design surrounded by zeros. 

3. Add a loss cost order. Currently all L1. But I might want something like L 1.5. So make this configurable.

# hybrid mode updates:
Currently we only use the block mode. We want to combine three mode. Randomly choosing one of the three
1. For the zig zag mode, I want to use a markovian structure. When we decide some starting point, the next point will be let's say 0.95 percent the same (this probability can be hardcoded not in configuration) Zig zag mode will also incur a horizontal cut (before only vertical ones)
In the horizontal mode. 

2. For the cluster and block mode. It basically will not change. But you need to make sure they work with all the cap pairs. I think block mode is already implementing the logic correctly. so the cluster mode need to follow that.

There should be some summary in each of the output condidate output file folder. These shows a. What are thier parents, are state champions used? b. what is the hybrid and the detail hypbrid paramenters

After modifying. Make sure all the configurations in ga are modified to represent this change.

Also, are we enabling state champions to be both parents? If they cannot, I want to enhance the probability of that.

GA cost and functions:

In the GA cost function I realized sometimes if there are many points (for instance in the cross band) The worst case will be diluted. Therefore, can we have some worst 5 average for each createrion like each cost components. So that the worst cases are considered.

# Cost function update

In GA cost function.Peak matching. In the current function, if the peaks shift, there will be catastrophic impact. But in reality peak shifting is not that servere and if leveraged  correctly can help optimization. Therefore, we will perform a peak matching according to the center of the passband, to match the peaks of S21 of the assigned states. Then shift every score categories according to this matching shift. After that we will compute the score normally as what we do now using the shifted ranges.However, these shifting scores are punished by how far the shift is. We can set some configurable threshold like +-20% of the centerfrequency, and punish the score maybe linearly about how large the shift is relative to the threshold.

# Fine tunning designs

Currently the fine tune of a design is by reading a design that I manually inserting empty rows/columns to embed a smaller design into a larger one. However, This creates a difficuties initializing design: The designs are not different from each other, and changing the outter designs might be highly inefficient. Therefore I want to modify to add a fine tune to larger mode, with its configurations in a separate section and can be enabled. This mainly works in the initialization stage: It will read a smaller design and insert columns/rows randomly untill the smaller design reaches our desired design size.

The insertion goes like this: It can insert rando columns or rows. When inserted, it can choose to copy the previous row or column (special case when inserting the left most column, since this is the feed line col and there is no previous col, we should copy the column 0 for the feedline extention).

All insertion mode
1. zero padding. We can zero padding evenly the smaller design to a larger one. when padding the top rows and bottom rows, they are relugar. However when we padding 
2. Random column/row content insertion. When we insert columns we generate a random 0/1 pair. 
3. Regular column duplicating

Break of constraints. When we add column. It is not possible to break inductor spacing or switch spacing constriants. But it's possible that the switch pair constraints be broken. Let's remove the switch pairs if such constraints are broken.