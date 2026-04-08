In this new SA update, we want to add a new mode (action) to include the newly updated pixel pairs into the simulated annealing method.

This method has the following implications (added feasibility test):
This pixel pairs must be added to two "1" pixels (top metal pixels). It cannot be applied to switches or inductor pixels or ground pixels.
For routing convirience, this pixel pairs can only be horizontal or vertical. (the lines connecting two pixels must be horizontal or vertical.).
On the path that connect the pixel pairs (defined by a 3 by 3 region over the path, like moving a 3 by 3 grid from start to the end of path), there shoudn't be other pixels unless 0 and 1.  

Ranodm generation: When we generate the initial design, we will add several pairs (defined by the max number of pairs). We can choose to remove them in the action (please add new reasonable actions in SA to teak the number of pairs by adding new or removing old, and disturb the caps)