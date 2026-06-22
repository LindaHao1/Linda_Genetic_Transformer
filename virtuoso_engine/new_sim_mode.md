Now I want to introduce a new simulation mode in the virtuoso simulation engine. This is named capacitor pixel pairs. This is a simulation adding to some pixels. For instance, this can be controlled by [(pix1 x, pix1 y), (pix2 x, pix2 y)], [(), ()]  as a list of list of tuples. for each tuple pair of pixels, this will create some port with some name for the port, similar to what we did for inductor and switches. It will connnect a capicitor between two pixels. For instance, it will be controlled by a cap value list [capval 1, capval2, ...]. This will be simulated as port only in the emx, and connect the cap to the corresponding ports in the sepectre simulation. Very similar to what we did for the switch embedded in the pixels. 


### Update 2

In this simulation update, we implement the switchable inductor. Currently, inductor are implemented as a fixed s par file connected to pixel ports. However, we want to implement the switable inductor, similar to how the normal switch are implemented. When the inductor is turned off, we should connect nothing to the indutor port. To control it, we have states similarly, We will define an inductor state list, which is similar to the switch states. So the inductor also has several states. The states is controlled simultaneously with the switch state. Basically if swtiches are at state 0, then inductors are also at state 0. 

Please figure out the plan and let me know how to impelent the open port in the simulator.


### Update 3

We supprot a new simulation component. This is a via for shorting pixel and gnd. to do this we need to insert the port on the metal layer, and during the spectre simulation, connect this kind of port directly to gnd. Similar to how inductor works but more simple. You can treat this as a new pixel type (index 4). During the plotting, we want to use a new color to represent this type of pixel.

## Another new sim mode
We already have the pixelated design simple, where we don't build the full layout but use a combination of simulated pixels and spectre components. We want to build a full simulation mode. It will Read some configuration same as the configuration for pixelated deisgn  simple,  but will generate a full layout.  Let's do not worry about EMX simulation and spectre for now. Let's just build the layout first.

The cell name and lib name specification are similar to what we have right now, we should meanually specify them in the configuration.

The configuration for this sim mode will be put into one section of the configuration file. It should not be interfering other configuratios. If this simulation mode is not selected all the previous mode will not be affected.

This mode will involve creating cap pairs, inductors and switches, and vias according to the configuration file. For the switch design,we already have that single pixel Demo. the goal is to convert the cap values to cap dimensions through a table (these tables can be csv files and referenced in the configuration). You can make up this table for now and i will provide later. Then we will use the switch script that already exists to generate the pixels.

when creating cap pairs, we have to adapt to horizontal and vertical caps. We need to rotate them respectively and make sure the wiring wires both ends to the pixel centers.

For the cap pairs we need to read from a different table to get the cap dimensions since we will have a different aspect ratio for the cap pairs. This part of the script does not exist yet. So you will need to write the skill script to connect the center of the pixel to one end of the cap, then connet the other end of the cap pair to the other pixel. You need to create vias and adaptive to the size of the cap pair.

For the via pixel, we just need a via size configuration and create the via form the pixel all the way to the ground.

For the active inductor pixel, this one requires reading from a already built active inductor. This inductor is drawn in virtuoso layout. So I will need to know  what files do i need (provide that in your plan). The job of this part is to duplicate and  move this layout to its corresponding location.


Testing: Let's create a single configuration for testing right now. Let's have some simple deisgn that can fit in one horizontal pair,  one vertical cap pair,  two inductors and two switches one vias. And then we can run this simple configuration to test.


Cap pair description:
Series cap logic change: The correct series cap placement: If the series if defined horizontal, the cap need to be rotated 90 degree. If the series cap is vertical, it does not need to be rotated. The series cap need to be placed  in the middle point of the line connectng the two pixel centers. after placing the pixels we will wire it in the M8 layer. We first find the PLUS like how we find it in switches, wire it to the center of the left pixel (or top pixel). Take horizontal as example. This will align in the x direction with the pix center. We need  another wire to start from x to the corresponding y, then use via to connect to the top metal layer. It is like a Ninety-degree zigzag line. SImilarly we do for the minus. For the bulk we will connect it to the ground like how we do to the switch.



Differential mode filter design and optimization.

We want to add a differential mode filter design, that might require another template like pixelated design simple. This is still a filter design but the input is a differential mode. The input is on the left edge, and the entire design is symmetric with respect to the x axis. The design matrix is only half of the design, since the other part is just symmetric. Notice that the bottom row will be the symmetric line, e.g. the bottom row will not be duplicated and the top part is mirrored under. So we will have input still in the left most column, but with the restriction that the input port cannot be on the bottom two rows, since after mirroring it will be too tight together. The output has only one port and it will always be in the middle. We can simply force the output column to have an 1 on the bottom row.

As for s par simulation, the logic is different. We will first add a capacitor in between the two input ports. This cap value will be 61fF (can be configured). Then we will add a 60 ohm port in between the input, and a regular 50ohm port on the output side. Therefore our spar will look into the input port (with the capacitor) and we still get the S21, S11, ...

The optimization will only optimize the half design using GA (remember the constraint of the input port), you can create yet another folder to store those configurations for differential filter design. We will have shunt caps, series caps available. But we can start with a scratch with only 0/1 pixels. 



Differential mode multi layer update

This will update the differential filter design with multi layer option. Single layer differential filter will fail, since it is symmetric and the current won't flow through the middle. so we look to add multi layer ability to the pxielated design structure of differential filtering. We have to let the output be in the second metal to leverage the induction current

To achieve this we enable two new pixel types: metal2, and double metal. Easy to under stand, we enable another metal layer (to be configured default as M9). 

The simulation change is that: instead of putting only one single output in the middle, we will still have a flexible output simialr to the input (that is indicated on the last column of the design). This will results in two outputs. In the spectre simulation, we will read the output from one port (let's say the top one), and the other output port will be grounded. Notice that we will force the output to be on the M9 layer (for the differential design only).

After the first few simulation changes is done, we look for adaptive changes in the port simulation and advanced pixel types. (we do not consider switch for now). The first one is shunt inductor, this inductor can be connected to either M10 or M9, depending on the base pixel type. If the base pixel type includes M9 layer, then it port is on M9, if not then its port is on M10 as usual.  

Vias has similar idea to shunt caps.

The most tricky one is the series caps, since we want to allow capacitors that goes beyond layers. We can forget about the in between constraints for series caps for now. But we have to enforce constraints on hw series caps can be applied. 1. if both ends are of the same type e.g. connecting to same layer. Safe. port regularly be put in the same layer for emx 2. If one of them is lay 10 and one is lay 9. The port must be placed on different layers in emx

Visualization change: Since we now have layer metal options/advanced pixel options to visualize. We can make the advanced pixel type an add on marker to not block the original visualization (so that both the original pixel metal type and the advanced pixel type can be displayed)

optimization: Let's start with creating this configuration but only with the metal pixels: /disk/mcao53/RFIC_RL/virtuoso_optimization_GA/configs/differential/scratch_21x27.yaml. (not any advanced pixel to just create some baseline)