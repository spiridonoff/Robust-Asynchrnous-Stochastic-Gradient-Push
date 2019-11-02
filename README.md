# Robust-Asynchrnous-Stochastic-Gradient-Push
A Decentralized SGD Algorithm: https://arxiv.org/abs/1811.03982

Centralized_v4.py simulates the centralized SGD

rasgp.py simulates the decentralized SGD algorithm RASGP proposed in the main paper. https://arxiv.org/abs/1811.03982

rasgp_means.py collectes the average of outputs of each pool of simulations and collects and saves them, so that later, by taking the median of those means, you can have a better estimate of the true expected value of the Mean Squared Error of the algorithm, due to the very large variance of the output.

The functions used in the python codes above are collected in functions_v3.py

Data_n100x50.pkl is the synthetic dataset used for the simulations in: https://arxiv.org/abs/1811.03982
