The training script for machine learning is generally in a good shape, I want to do the follwoing changes to improve:

1. The current encoding sceheme use nearest mid point to determine the region, which is not good. We will still snap to the mid points, however, when we determine the regions, we have some dividing points manually defined to devide the pixel into regions, and then we will do region specific snapping or clipping.

2. The current feasibility check enforce a strict switch count equal to the defined number, which often lead to invalid design. So we will instead limit our switch count between some min and max value. During feasibility check, we should have a optional parameter to check switch this way. The decoding and encoding for the switch should also be flexible to match the switch count

3. When Saving into the replay buffer, we will also record the score of the design. We should log the best design if a new best design is found during trainig. when traning the world model in wm_correction_interval, we also give some priority to the best performant designs in the replay buffer that is simulated by the world model, to avoid cheating the best designs. 

### New updates

1. For each iteration, plot the best, and worst design according to world model
2. For each iteration, grab and plot the best simulation design (if any)
3. When we do the world model correction. First we will sample the best performant / and also the most confident design in the replay buffer (give priorities to both),  then we will plot what the world model actually output/against what the simulation output. After calibration from the real-sim, the replay buffer entry should be replaced by real simulation and marked real. 
4. Log of the losses through tensorboard. log the world model loss and generation model loss in each iteration.
5. during each traning sample of the generation model, output the number of each kind of sample: real, world-model simulation, infeasible.

# Critial updates

1. Confidence level scheduling. We schedule the confidence level requested at the beginning to be 1 and slowly schedule to 0.7 linearly and stay the same.
2. Better sampling scheme. There are some ways for efficient training of the generative model that incorporate the score. I want to first sample more infeasible ones as there are basically no infeasible design during training. So we will and that to the priority sample, but not too much. Secondly. I will weigh the update acoording to their scores, the samples with lower cost will have more weights in the gradient training (so basically normalize according to the cost). But infeasible ones are not affected, they have all uniform weights since they do not have a cost.


Construct warmup dataset directly from warmupsims folder. Currently the warmup dataset is check pointed. However, there might be some changes in coding after each run. So When we detect the warmup simulation folder, we directly recalculate from the sim all the parameters, and add them to the replay buffer.

New guidance mode: We want the guidance (condition to be injected to the generation model) 

## 