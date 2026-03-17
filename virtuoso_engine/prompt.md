I have two virtuoso SKILL scripts written in the Single pixel demo folder. One is to generate a pixel layout, the other is to simulate the pixel layout using a diverse mix of methods. these two scripts are tested to be working in manual mode.

Now I want to write a python simulation engine to auto matically control the layout. First, we need to refactor the demo version code. Extract the common parameters. For instance the output folder, the parameters of the nmos in the layout and the spar scripts must match. (we can maybe have base scripts with placeholders that pyton can fill in according to configurations) Make all these into a configuration file. The python engine will read the configuration file and will generate the actual skill code according to the configuration. In all these parameters there will be several tunable parameters, those are the parameters that initialize the nmos and the capacitor. Those file should identify a unique cell name. The log folder is determined similarly, which different simulation results folder have different names, but have a common specified parent folder.

The interaction between python and virtuoso: The virtuoso should be running headlessly in the background. And the main script will inject new configurations to the engine so that it can run.  The virtuoso should be run in /disk/mcao53/Project65/. As this folder is already well configured.

Logging and saving: save the virtuoso command output and the configuration file with all the run artifacts.

plotting: you will notice the final s1p is written for different control voltages.  We can use some rf tool to read the s1p and plot S11 plots, and also S11 impedance in smith chart. Do this for both control voltages.

