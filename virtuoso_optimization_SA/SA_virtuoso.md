# Simulated annealing for optimizing switchable filters.

## Design constraints and goals

We will optimize the pixelated design to achieve switchable filter at the desired frequency.

Input to the algorithm:
grid m, n (m by n grids)
total allowed number of switches
switch capacitor range (300fF-800fF)
total allowed number of inductors.

creterion function:
For each state, we have one desired S12 parameter curve. For instance we have have a two state switchable filter where the first state is 7G, the second state is 20G. for each state we and have some ideal s curve, for instance maybe in 7G we only look at 6-8G and hae a perfect square s curve. And calculate the distance or MSE error between the current curve and the perfect curve (we don't punish for better optimization, for instance if our desired rejection is -15dB, and it is smaller that -15dB it is even better)


## Matrix rule understanding and Random layout generation.

the design matrix example 
  pixel_state_matrix:
    - [1, 1, 2, 1, 0]
    - [0, 3, 1, 3, 1]
    - [0, 0, 2, 1, 0]


For the internal matrix generation it will just generate random integer matrix with 0 1 2 3

feasibility check: 
The design rule for matrix are as follows:
a. connected path from input to output: out of the 8 direction (including diagnal), there is a path from the left most 1 to the right most 1 without any 0 (0 means there are to top metal so disconnected)
b. Switch Spacing check: There cannot be other switches or inductors around the 8 pixels of the switch pixel (3), so there can only be 0 and 1s
c. Inductor Spacing check: similar to switch spacing check
d. Inductor count check: we allow a max number of total inductors (10) in this example
e. Switch count check: for simplicity we force the total number of switches to be exactlyt the requried count

The left most and right most column indicate the position of the feedline. so there can be only one 1 in the left and right most column
When we generate a random matrix we can first generate two random numbers corresponding to these two locations
To improve the process of generation of valid random pixels. We can employ different pobablility of pixel generation. Largest is 1, then 0, then 2. Let's don't put any switch (pixel 3) yet.
After a random generation we do the connectivity test and inductor count test (d), if not, redo the random generation untill good.

Then we generate the switch. This is acheived by flipping the existing 1 to 3. We randomly flip N_switch times. Don't choose if it violates c or e.

Generate random switch states for N_state number of states (2 default).

Generate capacitance states using unifor distribution

This complete the random generation part.

Then we do a final feasibility check (2-5).   

## Parallel Simulated Annealing:

We generate N_parallel designs in each step in simulated anealing (you don't really need to do the SA in parallel, since each step should be fast, but we leverage the parallel simulation).

Initially generate one random designs.
We randomly choose one of the following actions to generate the next design based on the current one:

1. 0/1 flip, choose up to N_regular_flip count of 0 or 1 pxiels, flip them (0->1, 1->0)
2. Switch location change: randomly cancel one switch (3->1) and choose another place with pixel 1, and make it a switch (1->3) make sure it doesn't violate c and e.
3. Inductor filp: two stage, first choose an action add/remove inductor. If current inductor count is full, don't add. If add action, flip an random 1 to 2, if remove filp a random 2 to 1.
4. Cap value optimization, add random gaussian noise to the current cap value state (clip after adding to valid ranges)
5. switch State optimization: Rnadomly flip one of the state value (one number in the total state)

After action, we do a feasibility check (a-e) to see if anything is violated, if violation, we redo the action

The loop is one design -> N independt next design -> accept one -> next round

## Implementation

Please implement the SA framework in modular design and easy to configure. please implement in this path: virtuoso_optimization_SA