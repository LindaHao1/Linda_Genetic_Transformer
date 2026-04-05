


1. New world model update:

We want to completely change the world model learning scheme. Instead of an ensemble of world model (So you can remove the ensemble completely), we will have one world model (surrogate model) remember our traget image size is around 20-50 with some convolutional encoder into a latent space. And from that latent to prediction, and this prediction will include both the mean and variance of the prediction. 

RND for OOD measure, we use a same world model structure siwh a similar learning rate for random network distillation. When a new sample comes, we evaluate how OOD is it.

New creteria to send to simulation: If either variance and RND difference is high.


2. New guidance for feasibility

First, completely remove the loss for feasibility checking. I think it is confusing and useless.
Second, we don't sample at all infeasible samples to train the generative model.
Third, we will train a binary classifier (by sampling both feasible and infeasible designs). We use this as additional guidance during training and sampling. Is this possible?