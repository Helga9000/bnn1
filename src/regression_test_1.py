import bnn
import jax.numpy as jnp
import jax.random as random

x = jnp.linspace(0.0, 0.8, 50)
y = jnp.linspace(0.0, 0.7, 50)
key = random.key(0)
y = random.permutation(key, y)

trainingdata = []
for i in range(len(x)):
    a = jnp.array([[x[i]], [y[i]]])
    z = jnp.array([[x[i] + y[i]]]) + random.uniform(random.key(i), (1, 1)) * 0.6 - 0.30
    trainingdata.append((a, z))
net = bnn.bnn([2, 10, 1])
net.train(trainingdata, 300, 10)
print("x=0.05, y=0.05")
for j in range(6):
    print(net.feedforward(net.parameters, jnp.array([[0.05], [0.05]])))
print("x=0.6, y=0.2")
for k in range(6):
    print(net.feedforward(net.parameters, jnp.array([[0.6], [0.2]])))
print("x=0.9, y=1.1")
for k in range(6):
    print(net.feedforward(net.parameters, jnp.array([[0.9], [1.1]])))
print("sigma_noise: ", jnp.exp(net.parameters['log_sigma_noise']))
print(net.parameters['rho_weights'][1])
