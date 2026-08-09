
def init_params(key, sizes, dataset_size):
    key, *weight_keys = random.split(key, len(sizes))
    # Takes a list of layer sizes, e.g. [2, 10, 1] for a network with 2 inputs, one hidden layer with 10 neurons and one output neuron
    # Initializes the parameters of the network (weights and biases). Each parameter is represented by a mean (mu) and a rho value, which is transformed to a standard deviation using the softplus function.
    parameters = {
        'mu_biases': [jnp.zeros((y,)) for y in sizes[1:]],
        'mu_weights': [random.normal(k, (x, y))*jnp.sqrt(2.0 / x) for k, x, y in zip(weight_keys, sizes[:-1], sizes[1:])],
        'rho_biases': [jnp.full((y,), -2.0) for y in sizes[1:]],
        'rho_weights': [jnp.full((x, y), -2.0) for x, y in zip(sizes[:-1], sizes[1:])],
        # Parameters for stochastic volatility
        'sv_mu': jnp.array(0.0),
        'sv_phi_raw': jnp.array(1.5), 
        'sv_sigma_raw': jnp.array(-1.0),
        # Local Volatility Shocks
        'mu_h_shocks': jnp.zeros((dataset_size,)),
        'rho_h_shocks': jnp.full((dataset_size,), -2.0)
        }
    
    return parameters

def feedforward(key, params, a):
    # Calculates the output of the network for a given input 'a' and a set of parameters 'params'.
    num_layers = len(params['mu_weights'])
    keys = random.split(key, 2 * num_layers)
    w_keys = keys[:num_layers]
    b_keys = keys[num_layers:]
    
    for i in range(num_layers):
        mu_w = params['mu_weights'][i]
        mu_b = params['mu_biases'][i]
        # Samples weights and biases from a distribution given by mu and rho values.
        w = mu_w + random.normal(w_keys[i], mu_w.shape) * softplus(params['rho_weights'][i])
        b = mu_b + random.normal(b_keys[i], mu_b.shape) * softplus(params['rho_biases'][i])
        # Layer activation is calculated using the sampled parameters.
        a = jnp.matmul(a, w) + b
        # Final layer without activation function for regression tasks.
        if i < num_layers - 1:
            a = relu(a)
    
    return a

def loss(params, key, x, y, dataset_size):
    ff_key, sv_key = random.split(key)

    net_params = {
        'w': params['mu_weights'],
        'b': params['mu_biases']
    }
    number_of_weights = 2*jax.tree.reduce(
        lambda acc, x: acc + x.size, 
        net_params, 
        initializer=0
    )       

    predictions = feedforward(ff_key, params, x)
    T = x.shape[1] # Number of time steps in data
    sv_mu = params['sv_mu']
    sv_phi = 0.75 + jnp.tanh(params['sv_phi_raw']) / 4.0
    sv_sigma = 0.1*jax.nn.sigmoid(params['sv_sigma_raw'])

    # Sample volatility shocks and create volatility path
    eta = params['mu_h_shocks'][:T] + random.normal(sv_key, (T,)) * softplus(params['rho_h_shocks'][:T])
    h_init = sv_mu + eta[0] * (sv_sigma / jnp.sqrt(1.0 - jnp.square(sv_phi) + 1e-6))
    def transition_fn(carry, eta_t): # Function for lax.scan
        h_prev = carry
        h_t = sv_mu + sv_phi * (h_prev - sv_mu) + sv_sigma * eta_t
        return h_t, h_t
    _, h_tail = jax.lax.scan(transition_fn, h_init, eta[1:])
    h_states = jnp.concatenate([jnp.array([h_init]), h_tail])
    h_states = jnp.clip(h_states, min=-10.0, max=3.0)
    
    sq_error = optax.squared_error(predictions, y)
    variance = jnp.exp(h_states) + 1e-6
    likelihood_term = jnp.mean(sq_error / (2.0 * variance) + 0.5 * h_states)

    kl_weights = kl_divergence(params['mu_biases'], params['mu_weights'], params['rho_biases'], params['rho_weights'])
    sig_h = softplus(params['rho_h_shocks'][:T])
    kl_shocks = 0.5 * jnp.sum(jnp.square(sig_h) + jnp.square(params['mu_h_shocks'][:T]) - 1.0 - 2.0*jnp.log(sig_h + 1e-6))
    total_kl = ( kl_weights/(dataset_size * number_of_weights) + kl_shocks ) / T

    sv_regulizer = 200.0*jnp.square(sv_sigma)
    
    return likelihood_term + total_kl + sv_regulizer

def kl_divergence(mu_b_list, mu_w_list, rho_b_list, rho_w_list):
    total_kl = 0.0
    for mu_b, mu_w, rho_b, rho_w in zip(mu_b_list, mu_w_list, rho_b_list, rho_w_list):
        sigma_b = softplus(rho_b)
        sigma_w = softplus(rho_w)
        total_kl += 0.5 * (jnp.sum(jnp.square(sigma_b) + jnp.square(mu_b) - 1 - 2.0*jnp.log(sigma_b)) + jnp.sum(jnp.square(sigma_w) + jnp.square(mu_w) - 1 - 2.0*jnp.log(sigma_w)))
    return total_kl

def train(params, x, y, epochs):
    grad_fun = value_and_grad(loss, argnums=0)
    vmapped_grad_fun = jax.vmap(grad_fun, in_axes=(None, 0, None, None, None))
    key = random.key(42)

    # Learning rate decays 10% every epoch until 0.001.
    eta = 0.1
    dataset_size = x.shape[0]

    @jax.jit
    def train_step(params, key, x, y, dataset_size, eta):
        # Step A: Split 5 keys at once
        key, subkey = random.split(key)
        sample_keys = random.split(subkey, 5)

        # Step B: Compute 5 Monte Carlo gradients in parallel
        losses, grads = vmapped_grad_fun(params, sample_keys, x, y, dataset_size)

        # Step C: Average gradients across the 5 samples (axis 0)
        mean_loss = jnp.mean(losses)
        avg_grads = jax.tree.map(lambda g: jnp.mean(g, axis=0), grads)

        # Step D: Apply parameter updates with 20*eta for 'rho'
        def update_fn(path, p, g):
            lr = 20.0 * eta if 'rho' in str(path) else eta
            return p - lr * g

        new_params = tree_map_with_path(update_fn, params, avg_grads)

        return new_params, key, mean_loss

    for epoch in range(epochs):
        params, key, loss_val = train_step(params, key, x, y, dataset_size, eta)

        if eta >= 0.001:
            eta *= 0.90

        if epoch % 50 == 0:
            print(f"Epoch {epoch} | Loss: {loss_val:.4f}")

    return params
