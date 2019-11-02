# Robust-Asynchrnous-Stochastic-Gradient-Push
Robust Asynchronous Stochastic Gradient-Push: Asymptotically Optimal and Network-Independent Performance for Strongly Convex Functions https://arxiv.org/abs/1811.03982

Here are the codes for the RASGP, the main algorithm proposed in the paper above. These codes are used to perform the main simulations presented in the paper and to verify the theoritical results.

Centralized_v4.py simulates the centralized SGD to compare with RASGP

rasgp.py simulates the decentralized SGD algorithm RASGP

rasgp_means.py collectes the average of outputs of each pool of simulations and collects and saves them, so that later, by taking the median of those means, you can have a better estimate of the true expected value of the Mean Squared Error of the algorithm, due to the very large variance of the output.

The functions used in the python codes above are collected in functions_v3.py

Data_n100x50.pkl is the synthetic dataset used for the simulations.
