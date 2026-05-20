import jax
import jax.numpy as jnp
from jax import random, vmap, grad, value_and_grad, jit
from jax.nn import relu, softplus
import optax

class bnn(object):
    
    def __init__(self, sizes, dataset_size):
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
            # Parameters for stochastic volatility
            'sv_mu': jnp.array(0.0),
            'sv_phi_raw': jnp.array(1.5), 
            'sv_sigma_raw': jnp.array(-1.0),
            # Local Volatility Shocks
            'mu_h_shocks': jnp.zeros((dataset_size,)),
            'rho_h_shocks': jnp.zeros((dataset_size,)) - 2.0
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
        predictions = self.feedforward(params, x)
        T = x.shape[1] # Number of tim steps in data
        sv_mu = params['sv_mu']
        sv_phi = jnp.tanh(params['sv_phi_raw'])
        sv_sigma = softplus(params['sv_sigma_raw'])

        # Sample volatility shocks and create volatility path
        self.key, subkey = random.split(self.key)
        eta = params['mu_h_shocks'][:T] + random.normal(subkey, (T,)) * softplus(params['rho_h_shocks'][:T])
        h_init = sv_mu + eta[0] * (sv_sigma / jnp.sqrt(1.0 - jnp.square(sv_phi) + 1e-6))
        def transition_fn(carry, eta_t): # Function for lax.scan
            h_prev = carry
            h_t = sv_mu + sv_phi * (h_prev - sv_mu) + sv_sigma * eta_t
            return h_t, h_t
        _, h_tail = jax.lax.scan(transition_fn, h_init, eta[1:])
        h_states = jnp.concatenate([jnp.array([h_init]), h_tail])
        h_states = jnp.expand_dims(h_states, axis=0)
        h_states = jnp.clip(h_states, min=-10.0, max=3.0)
        
        sq_error = optax.squared_error(predictions, y)
        variance = jnp.exp(h_states) + 1e-6
        likelihood_term = jnp.mean(sq_error / (2.0 * variance) + 0.5 * h_states)

        kl_weights = self.kl_divergence(params['mu_biases'], params['mu_weights'], params['rho_biases'], params['rho_weights'])
        sig_h = softplus(params['rho_h_shocks'][:T])
        kl_shocks = 0.5 * jnp.sum(jnp.square(sig_h) + jnp.square(params['mu_h_shocks'][:T]) - 1.0 - 2.0*jnp.log(sig_h + 1e-6))
        total_kl = ( kl_weights/(dataset_size * self.number_of_weights) + kl_shocks ) / T
        
        return likelihood_term + total_kl

    def kl_divergence(self, mu_b_list, mu_w_list, rho_b_list, rho_w_list):
        total_kl = 0.0
        for mu_b, mu_w, rho_b, rho_w in zip(mu_b_list, mu_w_list, rho_b_list, rho_w_list):
            sigma_b = softplus(rho_b)
            sigma_w = softplus(rho_w)
            total_kl += 0.5 * (jnp.sum(jnp.square(sigma_b) + jnp.square(mu_b) - 1 - 2.0*jnp.log(sigma_b)) + jnp.sum(jnp.square(sigma_w) + jnp.square(mu_w) - 1 - 2.0*jnp.log(sigma_w)))
        return total_kl

    def train(self, training_data, epochs):
        # Learning rate decays 10% every epoch until 0.001.
        eta = 0.1
        dataset_size = len(training_data)
        x = jnp.concatenate([e[0] for e in training_data], axis=1)
        y = jnp.concatenate([e[1] for e in training_data], axis=1)

        for i in range(epochs):
            # Initializes accumulators for gradients of all parameters.
            delta_mu_b = [jnp.zeros_like(mu_b) for mu_b in self.parameters['mu_biases']]
            delta_mu_w = [jnp.zeros_like(mu_w) for mu_w in self.parameters['mu_weights']]
            delta_rho_b = [jnp.zeros_like(rho_b) for rho_b in self.parameters['rho_biases']]
            delta_rho_w = [jnp.zeros_like(rho_w) for rho_w in self.parameters['rho_weights']]
            delta_sv_mu = jnp.zeros_like(self.parameters['sv_mu'])
            delta_sv_phi_raw = jnp.zeros_like(self.parameters['sv_phi_raw'])
            delta_sv_sigma_raw = jnp.zeros_like(self.parameters['sv_sigma_raw'])
            delta_mu_h_shocks = jnp.zeros_like(self.parameters['mu_h_shocks'])
            delta_rho_h_shocks = jnp.zeros_like(self.parameters['rho_h_shocks'])

            # Gradient is calculated over 5 samples.
            for i in range(5):
                loss_value, gradient = self.grad_fun(self.parameters, x, y, dataset_size)
                delta_mu_b = [delta_mu_b + grad_mu_b for delta_mu_b, grad_mu_b in zip(delta_mu_b, gradient['mu_biases'])]
                delta_mu_w = [delta_mu_w + grad_mu_w for delta_mu_w, grad_mu_w in zip(delta_mu_w, gradient['mu_weights'])]
                delta_rho_b = [delta_rho_b + grad_rho_b for delta_rho_b, grad_rho_b in zip(delta_rho_b, gradient['rho_biases'])]
                delta_rho_w = [delta_rho_w + grad_rho_w for delta_rho_w, grad_rho_w in zip(delta_rho_w, gradient['rho_weights'])]
                delta_sv_mu += gradient['sv_mu']
                delta_sv_phi_raw += gradient['sv_phi_raw']
                delta_sv_sigma_raw += gradient['sv_sigma_raw']
                delta_mu_h_shocks += gradient['mu_h_shocks']
                delta_rho_h_shocks += gradient['rho_h_shocks']
            
            # Updates parameters
            self.parameters['mu_biases'] = [mu_b - eta*delta_mu_b for mu_b, delta_mu_b in zip(self.parameters['mu_biases'], delta_mu_b)]
            self.parameters['mu_weights'] = [mu_w - eta*delta_mu_w for mu_w, delta_mu_w in zip(self.parameters['mu_weights'], delta_mu_w)]
            self.parameters['rho_biases'] = [rho_b - 20.0*eta*delta_rho_b for rho_b, delta_rho_b in zip(self.parameters['rho_biases'], delta_rho_b)]                
            self.parameters['rho_weights'] = [rho_w - 20.0*eta*delta_rho_w for rho_w, delta_rho_w in zip(self.parameters['rho_weights'], delta_rho_w)]
            self.parameters['sv_mu'] -= eta * delta_sv_mu
            self.parameters['sv_phi_raw'] -= eta * delta_sv_phi_raw
            self.parameters['sv_sigma_raw'] -= eta * delta_sv_sigma_raw
            self.parameters['mu_h_shocks'] -= eta * delta_mu_h_shocks
            self.parameters['rho_h_shocks'] -= 20.0 * eta * delta_rho_h_shocks

        if eta >= 0.001:
            eta *= 0.90