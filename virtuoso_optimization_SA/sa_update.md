In this new SA update, we want to add a new mode (action) to include the newly updated pixel pairs into the simulated annealing method.

This method has the following implications (added feasibility test):
This pixel pairs must be added to two "1" pixels (top metal pixels). It cannot be applied to switches or inductor pixels or ground pixels.
For routing convirience, this pixel pairs can only be horizontal or vertical. (the lines connecting two pixels must be horizontal or vertical.).
On the path that connect the pixel pairs (defined by a 3 by 3 region over the path, like moving a 3 by 3 grid from start to the end of path), there shoudn't be other pixels unless 0 and 1.  

Ranodm generation: When we generate the initial design, we will add several pairs (defined by the max number of pairs). We can choose to remove them in the action (please add new reasonable actions in SA to teak the number of pairs by adding new or removing old, and disturb the caps)


# SA with some initial point

I have already got some good designs. I want to perform SA to refine the design. For instance, I might want to begin with this design and refine it: [text](../../Sim_engine_results/ga_filter_results_block_hybrid/sims/gen9_ind53/worker_1/pixsimple_r21c27_ps25p0_gs20p0_frl13frr9_w1/config.yaml). Therefore I want to first load that one as the initial design.

Then the probability of the actions might be different, there fore we don't want to modify the default sa_config, you can create a new configuration, and in that one, mainly tune the cap paris. Make other decisions low prob.

Then in the cap pairs I want to make sure the action logic and feasibility logic. Can you explain me again how to check they are feasible. It is also important to make sure the connected like of the caps (including the cap pixels) all doesn't has any special pixels of chebshev distance smaller or equal to 2.