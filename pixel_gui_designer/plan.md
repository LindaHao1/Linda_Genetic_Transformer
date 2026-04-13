This is a gui for the desining pixel design based on virtuoso engine. This gui operates on pixel design.yaml. Modifying and saving the pixel design.yaml.

It will first load one design, display the design on the left, similar to how we plot it right now.

There are two modes, selection mode and brush mode.

In selection mode, we will select a pixel, and several configurations of the pixel will appear in the bottom. Depending on the pixel type.
For instance, if it is an inductor pixel, it will display its inductor value, if it is a switch it will display its switch capacitance. If it is a cap pair it will display the cap pair value.
we can change the single pixel type in the selection mode. If we change one of the pixel in a cap pair, the whole cap pair will gone.

In the brush mode, we can first select a pixel type, hold the mouse and where ever we brush the pixel is changed to that type. This only works with gnd and top layer as others are lesser and can be tuned individually.

On the right we can have a table of states for the switches and inductors. This is used to adjust the switching states.

If we click save, it will save in the origianl one. If we click save as, it will prompt a place to save the yaml.

This gui operates on the yaml files only, so you don't need to change the simulaton engine or other optimizations. But you can read them to understand what you need to do.

Please implement this with your favorite python tool.