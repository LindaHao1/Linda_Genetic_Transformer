# new cross over options 

This is a block based cross over option. In this method, we generte two same sized bounding box, a random direction (parent 1 -> parent 2 or parent 2 -> parent 1). And then we move the block from the starting parent, to replace the bounding block of the destination parent.

Constraints and special cases:
Both of the x and y size of the bounding box are randomly generated. can constrained within a range (5-15 by default).

Boxes at the edge. Since the left and right most columns are designed for the feedlines. We can not move a box that touches the edge to other places. So if one box is at the edge (containing the feelines), the other box must also be at the same edge.

There need to be some fixes. We might have two inputs or outputs ports, we can fix by removing any of them to kept one input one outpu. Ohter fixes are similar to what we already have.

