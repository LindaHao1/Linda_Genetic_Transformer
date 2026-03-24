This is the ambitious goal of online trainig of generative modeling, combined with online training of a world model to provide rapid feedback.

## Data formatting
Before we start the ML part, Let's get some preliminaries for the design space and simulation.

The simulation is achieved in virtuoso_engine, it is basically feeding a configuration to the engine in a consumer producer framework.

The configurations include a pixel structure, a capacitor values list, a inductor values list, you can read the virtuoso engine for mare details.

We will embed all these design space into a one-channel picutre.

The design matrix has 4 options, which natually divide the one channel into four regions, for the left most and right most column, there are only one 1, representing the feedlines, so only 2 regions in these two columns.

We will embedde continuous value into the corresponding region, like capacitor value. The inductor is discrete so that inductor region will be discretized. The other regions are just discretized.

The switch state are encoded as additional 0/1 in columns, they will be appended to the design matrix. Depending on how many states we have.

to avoid ambiguity, we and add some spacing between regions, to avoid smooth transistion between regions.

With these, the design space can be translated as a single pictrure.

## Overall Framework

The framework includes a conditioned generative model. It receives ideal target s parameters (S21) (for all states concatenated), and generate a pixel design.

Then there is a world model that utilize ensemble learning for active learning. The core is it has a confidence value based on the agreement of the ensembles.

Training process:

0. Initilize the training dataset.
We randomly generate ~100 samples of valid designs. Use this to warm up the generative modeling and the world model. Add them to the dataset.

After 0. The algorithm iterate through 1-5 repeatedly

1. Data sampling through generative modeling.
We first give our ideal design curve to the diffusion model, then it will produce some design (initilally it will be poor but it's ok, we will let the model explore) which it think will be good. 

2. World model prediction of the generated design
The generated design will be feed into the world model. It will produce the s parameters for both states. Along with some confidence value. The high confidence samples will be directly relabeld by the world model prediction. This relabel is crucial as it aims to correct the generative model's understanding.

The low-confidence samples will be feed into the simulator to obtain real data. and added to the dataset.
The dataset will save (pixel, s params, confidence if predicted by the world model)

3. Generative model training with priority sampling
Give priority to 1. recent data 2. real simulation data 

4. World model training with real data.
We sample the buffer with only the real data. Train the world model with real generated data.

5. World model simulation correction
to avoid overfit, once in a while we need to sample some high confidence data in the dataset, and feed those data into the simulator. And then train the world model with the simulator data. Then replace them in the dataset.

## Checkpointing and logging

We will check point the dataset as well as the models once in a while. The dataset can be overwritten to save space. Use tensorboard to log crucial information.

## Generative modeling details

We want to employ a flow mathicng to generate the pixel designs. This generative method will be conditioned on the s parameter, rather than simply generating the design distribution randomly. To do this, you modify the neural network that predicts the vector field. Instead of just taking time $t$ and state $x$, the network $v_\theta$ also takes your design requirement $c$ as an input.The loss function remains almost identical, but your network is now conditional. We will not naively give the s parameter to the net there will be some encoder.

Since I am not that familiar with flow matching, you need to give me some formulation or plans how you gonna implement this.


## World model 

Use some encoder frame work to deal with the pixel designs. Out put s parameters for all satas. Use some proven good ensemble method for prediction and judging the confidence.

## Configuration files.

Create  a configuration of detials. (e.g. how many points are there in toal, pixel rows and cols, basically all parameters needed)

## Feasibility Labeling.

We will label the infeasible design with some mark (like some impossible values liek all 1 s parameters to let the network know it is not feasible). These designs will bypass world model and simulation. During sampling, give low priority to infeasible design. 

We don't want the infeasible design to dominate, 