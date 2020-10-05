This is a package to demonstrate drift flux prediction using neural network model in C++

The model and implementation is part of the supplentary material to the publication:

Yundi Jiang, Xiao Chen, Jari Kolehmainen, Ioannis G. Kevrekidis, Ali Ozel, Sankaran Sundaresan, "Development of Data-Driven Filtered Drag Model for Industrial-Scale Fluidized Beds", Chemical Engineering Science

The input variables are in `input.csv` file

`NN_prediction.C` has the main function that calls prediction subroutine and outputs a predicted drift flux


To compile and run the code:
   
   `g++-10 NN_prediction.C keras_model.C -o predicted.o`
   
   `./predicted.o`


The output should be printed as:

```
---input variables ---
Reynolds number: 1.13
solid volume fraction: 0.3
filter size: 0.015
grad_P: -5000
slip_velocity: 0.1
particle_diameter: 7.5e-05
scaled solid volume fraction: 0.46875
dimless filter size: 0.0195677
dimless grad P: -0.339789
dimless_slip_velocity: 0.47619
Fr: 59.9388
---prediction ---
prediction: -0.0111881 
```
