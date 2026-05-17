import jax.numpy as jnp
from jax import random, vmap, grad, value_and_grad, jit
from jax.nn import relu, softplus
import optax
import random as rand

class bnn(object):
    
    def __init__(self, sizes):
        # Takes a list of layer sizes, e.g. [2, 10, 1] for a network with 2 inputs, one hidden layer with 10 neurons and one output neuron
        # Initializes the parameters of the network (weights and biases). Each parameter is represented by a mean (mu) and a rho value, which is transformed to a standard deviation using the softplus function.
        self.key = random.key(42)
        self.number_of_layers = len(sizes)
        self.number_of_weights = 2*sum((x*y for x, y in zip(sizes[:-1], sizes[1:])))
        self.sizes = sizes
        self.parameters = {
            'mu_biases': [jnp.zeros((y, 1)) for y in self.sizes[1:]],
            'mu_weights': [random.normal(self.key, (y, x))*0.1 for x, y in zip(self.sizes[:-1], self.sizes[1:])],
            'rho_biases': [jnp.zeros((y, 1))-2 for y in self.sizes[1:]],
            'rho_weights': [jnp.zeros((y, x))-2 for x, y in zip(self.sizes[:-1], self.sizes[1:])],
            # A single parameter for the log of the noise standard deviation. This parameter will be optimized during training to find the best noise level for the predictions.
            'log_sigma_noise': jnp.array(0.0)
            }
        # Stores the gradient function for the loss, which will be used during training to compute the gradients of the parameters with respect to the loss.
        self.grad_fun = value_and_grad(self.loss)

    def feedforward(self, params, a):
        # Calculates the output of the network for a given input 'a' and a set of parameters 'params'.
        self.key, subkey = random.split(self.key)
        for i in range(self.number_of_layers - 2):
            subkey, w_key, b_key = random.split(subkey, 3)
            mu_w = params['mu_weights'][i]
            mu_b = params['mu_biases'][i]
            # Samples weights and biases from a distribution given by mu and rho values.
            w = mu_w + random.normal(w_key, mu_w.shape) * softplus(params['rho_weights'][i])
            b = mu_b + random.normal(b_key, mu_b.shape) * softplus(params['rho_biases'][i])
            # Layer activation is calculated using the sampled parameters.
            a = relu(jnp.dot(w, a) + b)
        subkey, w_key, b_key = random.split(subkey, 3)
        mu_w = params['mu_weights'][self.number_of_layers - 2]
        mu_b = params['mu_biases'][self.number_of_layers - 2]
        w = mu_w + random.normal(w_key, mu_w.shape) * softplus(params['rho_weights'][self.number_of_layers-2])
        b = mu_b + random.normal(b_key, mu_b.shape) * softplus(params['rho_biases'][self.number_of_layers-2])
        # Final layer without activation function for refgression tasks.
        a = jnp.dot(w, a) + b
        return a

    def loss(self, params, x, y, dataset_size):
        # Loss function combines mean squared error and KL divergence between parameter distributions and a standard normal distribution.
        predictions = self.feedforward(params, x)
        # KL term is scaled by 1/(dataset_size * number_of_weights) to balance it's influence.
        kl_loss = self.kl_divergence(params['mu_biases'], params['mu_weights'], params['rho_biases'], params['rho_weights'])/(dataset_size * self.number_of_weights)
        return jnp.mean(optax.squared_error(predictions, y)) / (2.0*jnp.pow(jnp.exp(params['log_sigma_noise']), 2) + 1e-6) + params['log_sigma_noise'] + kl_loss
    
    def kl_divergence(self, mu_b_list, mu_w_list, rho_b_list, rho_w_list):
        total_kl = 0.0
        for mu_b, mu_w, rho_b, rho_w in zip(mu_b_list, mu_w_list, rho_b_list, rho_w_list):
            sigma_b = softplus(rho_b)
            sigma_w = softplus(rho_w)
            total_kl += 0.5 * (jnp.sum(jnp.square(sigma_b) + jnp.square(mu_b) - 1 - 2.0*jnp.log(sigma_b)) + 
                              jnp.sum(jnp.square(sigma_w) + jnp.square(mu_w) - 1 - 2.0*jnp.log(sigma_w)))
        return total_kl

    def train(self, training_data, epochs, batch_size):
        # Learning rate decays 10% every epoch until 0.001.
        eta = 0.1
        dataset_size = len(training_data)
        for i in range(epochs):
            rand.shuffle(training_data)
            batches = [training_data[j : j+batch_size] for j in range(0, dataset_size, batch_size)]
            for batch in batches:
                # Batch is concatenated into a matrix for efficiency.
                x = jnp.concatenate([e[0] for e in batch], axis=1)
                y = jnp.concatenate([e[1] for e in batch], axis=1)
                # Initializes accumulators for gradients of all parameters.
                delta_mu_b = [jnp.zeros_like(mu_b) for mu_b in self.parameters['mu_biases']]
                delta_mu_w = [jnp.zeros_like(mu_w) for mu_w in self.parameters['mu_weights']]
                delta_rho_b = [jnp.zeros_like(rho_b) for rho_b in self.parameters['rho_biases']]
                delta_rho_w = [jnp.zeros_like(rho_w) for rho_w in self.parameters['rho_weights']]
                delta_log_sigma_noise = jnp.zeros_like(self.parameters['log_sigma_noise'])
                # Gradient is calculated over 5 samples.
                for i in range(5):
                    loss_value, gradient = self.grad_fun(self.parameters, x, y, dataset_size)
                    delta_mu_b = [delta_mu_b + grad_mu_b for delta_mu_b, grad_mu_b in zip(delta_mu_b, gradient['mu_biases'])]
                    delta_mu_w = [delta_mu_w + grad_mu_w for delta_mu_w, grad_mu_w in zip(delta_mu_w, gradient['mu_weights'])]
                    delta_rho_b = [delta_rho_b + grad_rho_b for delta_rho_b, grad_rho_b in zip(delta_rho_b, gradient['rho_biases'])]
                    delta_rho_w = [delta_rho_w + grad_rho_w for delta_rho_w, grad_rho_w in zip(delta_rho_w, gradient['rho_weights'])]
                    delta_log_sigma_noise = delta_log_sigma_noise + gradient['log_sigma_noise']
                # Updates parameters
                self.parameters['mu_biases'] = [mu_b - eta*delta_mu_b for mu_b, delta_mu_b in zip(self.parameters['mu_biases'], delta_mu_b)]
                self.parameters['mu_weights'] = [mu_w - eta*delta_mu_w for mu_w, delta_mu_w in zip(self.parameters['mu_weights'], delta_mu_w)]
                self.parameters['rho_biases'] = [rho_b - 20.0*eta*delta_rho_b for rho_b, delta_rho_b in zip(self.parameters['rho_biases'], delta_rho_b)]
                self.parameters['rho_weights'] = [rho_w - 20.0*eta*delta_rho_w for rho_w, delta_rho_w in zip(self.parameters['rho_weights'], delta_rho_w)]
                self.parameters['log_sigma_noise'] = self.parameters['log_sigma_noise'] - eta*delta_log_sigma_noise
            if eta > 0.001:
                eta *= 0.90