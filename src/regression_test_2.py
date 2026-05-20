import bnn_tuned_for_macrodata
import jax.numpy as jnp
import jax.random as random

N = 100
key = random.key(0)
k1, k2, k3 = random.split(key, 3)
x = random.normal(k1, (1, N))
y = random.normal(k2, (1, N))
true_mu = -2.0
true_phi = 0.9
true_sigma = 0.3
h = jnp.zeros(N)
h = h.at[0].set(true_mu + random.normal(k3, ())*true_sigma / jnp.sqrt(1 - true_phi**2))
for t in range(1, N):
    key, subkey = random.split(key)
    h.at[t].set(true_mu + true_phi * (h[t-1] - true_mu) + random.normal(subkey, ())*true_sigma)
true_variance = jnp.exp(h)
inputs = jnp.vstack([x, y])
noise = random.normal(key, (1, N))*jnp.sqrt(true_variance)
trainingdata = []

for i in range(N):
    a = jnp.vstack([x[:, i:i+1], y[:, i:i+1]])
    z = jnp.array([[x[0, i] + y[0, i]]]) + noise[:, i:i+1]
    trainingdata.append((a, z))
net = bnn_tuned_for_macrodata.bnn([2, 10, 1], len(trainingdata))
net.train(trainingdata, 100)
print("x=0.05, y=0.05")
for j in range(6):
    print(net.feedforward(net.parameters, jnp.array([[0.05], [0.05]])))
print("x=0.6, y=0.2")
for k in range(6):
    print(net.feedforward(net.parameters, jnp.array([[0.6], [0.2]])))
print("x=0.9, y=1.1")
for k in range(6):
    print(net.feedforward(net.parameters, jnp.array([[0.9], [1.1]])))

print("sv_sigma_raw:", net.parameters['sv_sigma_raw'])
print("sv_mu:", net.parameters['sv_mu'])
