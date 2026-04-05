This is a optimization method similar to the flow mathcing one but with the below differences. The changes only apply in this folder 

Please implement:
1. Completely remove the flow matching framework in this example. We want to switch to diffusion model. but overall the ideas are the same. The agent and traning scheme will be switched to diffusion model. You can also remove the soft constraint function right now.

2. We will add a DDPO (using RL to train the diffusion model by letting the diffusion model to interact with the world model) step after training the generative model with offline sampled data. The diffusion model is given the ideal sparameter we want. 

In this step, we aim to give the reward based on the world model feedback of the diffusion generated design. Also, we should only give the RL feedback if the world model is confident about its prediction. It is ok if none of the designs are confident we will not update the diffusion.

If the diffusion generated design is infeasible, we will give large punishment based on how many rules it violates. If the design is infeasible, we can skip the world model prediction. The goal of this step is to finetune the diffusion model.


## New Guidance Mode
For most of these changes we already did in DDPO, now we want to repeat in this folder in the Flow matching

1. We want to have another guidance mode
Instead of using an ideal S parameter as the guidance (which is typically far ways from what is realizable, often create instability in learning). We want to use score at each states. We can use four scores to describe each state. Lower reject socre, pass band score, higher reject score, corss band score. We design a graduate guiding way: we record the currently best score (defined by the real simulation only). And guide the diffusion with a slightly better score (not exceeding the max score).

2. Accompanying this guidance mode. We have to not directly load the warm up check point as it has changed. Instead we will read the warm up sim folder and regenerating the warm up datasets according to the current configuration (for instance /disk/mcao53/Sim_engine_results/ml_filter_results_try_2/warmup_sims).

Pese make this guidance mode an option on top of the ideal spar guidance.

3. Then Feasible design poplulation in warm up.: We will populate N (in configuration, set to 1000 as default) number of random feasible designs, when traiing with these designs don't use any guidance (as this has no s parameters). Then discard these after warm up.
