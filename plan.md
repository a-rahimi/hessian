I want to implement the algorihtm described in hessian.tex in python. I'd like
to do this in a few steps:

1. implement a barebones multi-linear percepton with 20 layers.

2. represent the model as an nn.Sequential, where each layer is one item in the
nn.Sequential.

3. write a training loop for this model on imagenet.

4. write a help function that captures the mixed partial derivatives of each
each layer with respect to its parameters and inputs as a tensor.

5. implement the hessian-inverse-vector product described in hessian.tex.

6. In the training loop, the hessian-inverse-vector product to the gradient before doing the gradient update.