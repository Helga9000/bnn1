import bnn_tuned_for_macrodata as bnn
import jax.numpy as jnp
import jax.random as random

N = 100
key = random.key(0)

# Generate synthetic dataset inputs and noise
key, k1, k2, k3 = random.split(key, 4)
x_raw = random.normal(k1, (1, N))
y_raw = random.normal(k2, (1, N))

true_mu = -2.0
true_phi = 0.9
true_sigma = 0.3

# Generate stochastic volatility trajectory h
h = jnp.zeros(N)
h = h.at[0].set(true_mu + random.normal(k3, ()) * true_sigma / jnp.sqrt(1 - true_phi**2))

for t in range(1, N):
    key, subkey = random.split(key)
    val = true_mu + true_phi * (h[t-1] - true_mu) + random.normal(subkey, ()) * true_sigma
    h = h.at[t].set(val)

true_variance = jnp.exp(h)

# Prepare inputs (shape: [batch_size, time_steps, features])
# Here: batch_size = 1, time_steps = N, features = 2 for inputs / 1 for target
inputs = jnp.vstack([x_raw, y_raw]).T  # Shape: (N, 2)
inputs = jnp.expand_dims(inputs, axis=0)  # Shape: (1, N, 2)

key, subkey = random.split(key)
noise = random.normal(subkey, (1, N)) * jnp.sqrt(true_variance)
targets = (x_raw + y_raw) + noise  # Shape: (1, N)
targets = jnp.expand_dims(targets, axis=-1)  # Shape: (1, N, 1)

dataset_size = inputs.shape[0]

# Initialize parameters and train
params = bnn.init_params(key, [2, 10, 1], dataset_size)
trained_params = bnn.train(params, inputs, targets, 100)

test_inputs = [
    ("x=0.05, y=0.05", jnp.array([[0.05, 0.05]])),
    ("x=0.6, y=0.2",   jnp.array([[0.6, 0.2]])),
    ("x=0.9, y=1.1",   jnp.array([[0.9, 1.1]]))
]

for label, test_x in test_inputs:
    print(label)
    for _ in range(6):
        key, subkey = random.split(key)
        pred = bnn.feedforward(subkey, trained_params, test_x)
        print(pred)

print("sv_sigma_raw:", trained_params['sv_sigma_raw'])
print("sv_mu:", trained_params['sv_mu'])
