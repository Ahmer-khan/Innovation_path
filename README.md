# Innovation_path
Implementation of Innovation path as per the publication "Innovation Path: Discovering an Ordered Set of Optimized Intermediate Solutions from an Existing to a Desired Solution"
Requires pymoo backend please make sure you have pymoo installed
Just need to run the main file for it to work
need to provide all the variable and a step constraint class which sould have an atrribute delta of the step used and should return 3 values, the bracketed sum of constraint voilations, the constraint violations itself and the right hand side of the constraint.
To get an idea of the constraint function see the constraint used in the paper in utilities.

#Citation
Khan, Ahmer, and Kalyanmoy Deb. "Innovation Path: Discovering an Ordered Set of Optimized Intermediate Solutions from an Existing to a Desired Solution." Proceedings of the Genetic and Evolutionary Computation Conference. 2024.
