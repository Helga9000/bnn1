import bnn
import jax.numpy as jnp
import jax.random as random

x = jnp.linspace(0, 0.8, 50)
y = jnp.linspace(0, 0.7, 50)

trainingdata = []
for i in range(len(x)):
    a = jnp.array([[x[i]], [y[i]]])
    z = jnp.array([[x[i] + y[i]]]) + random.uniform(random.PRNGKey(i), (1, 1)) * 0.6 - 0.3
    trainingdata.append((a, z))
net = bnn.bnn([2, 10, 1])
net.train(trainingdata, 300, 10, eta=0.001)
print("x=0.05, y=0.05")
for j in range(10):
    print(net.feedforward(net.parameters, jnp.array([[0.05], [0.05]])))
print("x=0.6, y=0.2")
for k in range(10):
    print(net.feedforward(net.parameters, jnp.array([[0.6], [0.2]])))