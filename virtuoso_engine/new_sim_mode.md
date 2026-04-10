Now I want to introduce a new simulation mode in the virtuoso simulation engine. This is named capacitor pixel pairs. This is a simulation adding to some pixels. For instance, this can be controlled by [(pix1 x, pix1 y), (pix2 x, pix2 y)], [(), ()]  as a list of list of tuples. for each tuple pair of pixels, this will create some port with some name for the port, similar to what we did for inductor and switches. It will connnect a capicitor between two pixels. For instance, it will be controlled by a cap value list [capval 1, capval2, ...]. This will be simulated as port only in the emx, and connect the cap to the corresponding ports in the sepectre simulation. Very similar to what we did for the switch embedded in the pixels. 


### Update 2

In this simulation update, we implement the switchable inductor. Currently, inductor are implemented as a fixed s par file connected to pixel ports. However, we want to implement the switable inductor, similar to how the normal switch are implemented. When the inductor is turned off, we should connect nothing to the indutor port. To control it, we have states similarly, We will define an inductor state list, which is similar to the switch states. So the inductor also has several states. The states is controlled simultaneously with the switch state. Basically if swtiches are at state 0, then inductors are also at state 0. 

Please figure out the plan and let me know how to impelent the open port in the simulator.


### Update 3

We supprot a new simulation component. This is a via for shorting pixel and gnd. to do this we need to insert the port on the metal layer, and during the spectre simulation, connect this kind of port directly to gnd. Similar to how inductor works but more simple. You can treat this as a new pixel type (index 4). During the plotting, we want to use a new color to represent this type of pixel.