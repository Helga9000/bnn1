import bnn
import jax.numpy as jnp

x = jnp.array([[0.0], [0.01], [0.02], [0.04], [.06], [0.08], [0.1], [0.12], [0.15], [0.18]])
y = jnp.array([[0.0], [0.1], [0.04], [0.16], [.036], [0.064], [0.01], [0.0144], [0.225], [0.0324]])
trainingdata = []
for i in range(len(x)):
    a = jnp.array([[x[i][0]], [y[i][0]]])
    z = jnp.array([[x[i][0] + y[i][0]]])
    trainingdata.append((a, z))
net = bnn.bnn([2, 10, 1])
net.train(trainingdata, 1000, 5, 0.01)
for i in range(10):
    print(net.feedforward(net.parameters, jnp.array([[0.05], [0.05]])))
