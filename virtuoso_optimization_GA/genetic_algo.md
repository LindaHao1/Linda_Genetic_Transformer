Similar to the SA, I want to implement a GA approach for optimization. In the GA optimization the most important is how to hybrid two designs. Let's assume we have two designs, then we draw some random cuts between the pixels (might be a zig zag cut), and we will take left part of one parent and right part of another parent.


A reminder of the constraints:
a. connected path from input to output: out of the 8 direction (including diagnal), there is a path from the left most 1 to the right most 1 without any 0 (0 means there are to top metal so disconnected)
b. Switch Spacing check: There cannot be other switches or inductors around the 8 pixels of the switch pixel (3), so there can only be 0 and 1s
c. Inductor Spacing check: similar to switch spacing check
d. Inductor count check: we allow a max number of total inductors (10) in this example
e. Switch count check: for simplicity we force the total number of switches to be exactlyt the requried count


After that the pixel design will likely be invalid, we will do the following to fix:
First remove redundant inductor: remove inductor over the count limit. remove inductor violate spacing.

We then fix the switches:
    we check b, if b is violated please remove switch/inductor that caused the violation.
Then we add back switch to the count if switch is below requirement, or we remove the switches back if switch is over limit. (done randomly)

Then the only possible error should be the connectivity. Since we divide the pixels into two halves, we will start from teh left port, find the right most connecting pixel. Then we start from the right pixel port, find the left connecting pixels, then we will create pixels that connect both parts, or you can suggest a better connectivity fix.

After that we will do a final round check just to ensure it is valid. We can discard if it is not valid.


Overall logging and plotting: Please have some progression plot of how the herd evolves.

The mutation is similar to SA so I will not explain here.
Please look at how SA is implemented and do the GA implementation.