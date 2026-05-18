We will use parallel engine to first collect dataset for the world model training.

We will use a similar generation of random design as GA used. This random generation will be reading from a generation configuration that specify the grid information. we want basically reuse this configuration: /disk/mcao53/RFIC_RL/virtuoso_optimization_GA/configs/21x27/ps31_gs25_scratch.yaml.  However, we only use the random generation part. Moreover, we want to have a much larger number of state simulated for each deisgn (10). Actually lets copy the above mentioned configuration and do some modification to put in the world model test folder.

Pease wrtie a script to genrate this dataset. The dataset generation count can be an argument of the newly writen file data_generation.py. Please write this data collection file that uses the parallel engine. 


Dataset creation
we have created dataset, from the collector, with path like: /disk/mcao53/Sim_engine_results/world_model_data_21x27_ps31_gs25/sims/design_000000/worker_0/pixsimple_r21c27_ps31p0_gs25p0_frl15frr4_w0/emx_results/pixsimple_r21c27_ps31p0_gs25p0_frl15frr4_w0_state0.s2p. However this is not best suited for ML, we need to convert these s2ps and designs into dataset format.

World (surrogate) model Structure Training, and test.
This world model will read the encoding of one of the state, and produce the prediction of that state. The states includes the design matrix, capvalues for inductors and switches, series cap and their values. and state on/off. It would be best to embed this into a multi-channel picture, and use some sort of CNN architecture for prediction. Some mixture of pciture and vectors are also acceptable.

Training approach: develop training approach to train the world model

Stretch goal： we might need to use some active learning idea. Can the world model tell some confidence? Is this design well learned or not? 