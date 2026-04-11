# New hybrid method

Weighted clustering hybrid method. To have better hybrid between parents we propose weighted clustering blender. (as a option and keep the current unchanged, we can switch between)

This blender is inspired by k means clusering, but each clusrter centroid has a weight for the distance.

We first initialize some number of centroids (2-4 by default), and then assign each entroids to some parents, meaning that we'll take the design of the cluster that belong to that centroid form that assigned parents. Make sure that two different parents occurs otherwise its meaningless.

Cluster bourndary deciding. We assign each pixel to its closest centroid to decide the clusters. But the distance is timed by the weights (randomly initialized from (0.5-1.0)). so that we have richer boundaries. we can break ties arbirarily

Repairing. Notice this boundary assign ment will also include the  input and output ports, in some cases we might need to randomly kill input/output ports if there are two.

Other repariments are as usual, like spacing or count repairments, they are easy.

You can think of one eazy connectivety fix, we can only ensure input port to output port connectivity. If it is hard, it's also ok to discard and redo.


We should keep these codes separated from the current hybrid implementation for better maintainance and compatibility.