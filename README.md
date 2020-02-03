#MLMI4: Advanced Machine Learning
Implementation of *Doubly Stochastic Variational Inference in Deep Gaussian Processes*. The paper can be found [here](https://arxiv.org/abs/1705.08933).

## TODO:
* When using `full_rank=False`, `conditional_SD` can return negative variances, causing NaN values in `reparameterisation` ***FIXED***.
