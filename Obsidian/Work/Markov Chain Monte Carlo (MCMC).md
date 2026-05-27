
So MCMC = **construct a Markov chain whose stationary distribution is the target $p(x)$, then run it to generate samples**.

The “Monte Carlo” part: we then use those samples to approximate expectations, probabilities, etc.

---

# Famous MCMC methods

There are many ways to design the chain. The most common are:

### a) **Metropolis–Hastings**

- Propose a move from $x$ to $x′$ (e.g. add some noise).
- Accept the move with probability:

$\alpha = \min\left(1, \frac{p(x') q(x|x')}{p(x) q(x'|x)}\right)$,

where $q$ is the proposal distribution.

- Otherwise, stay at $x$.  
    This guarantees detailed balance → stationary distribution is $p$.

---

### b) **Gibbs sampling**

- Special case of Metropolis where you sample one variable at a time from its conditional distribution:
$x_i \sim p(x_i \mid x_{-i})$.

- Works well when these conditionals are easy to sample.

---

### c) **Langevin dynamics (MALA / ULA)**

- Uses gradient information (the score $\nabla \log p(x)$).
- Update rule looks like:

$x_{t+1} = x_t + \frac{\epsilon}{2}\nabla \log p(x_t) + \sqrt{\epsilon}\,\xi_t, \quad \xi_t \sim \mathcal N(0,I)$.

- This is basically noisy gradient ascent on $\log p(x)$.
- With a Metropolis correction, this is **Metropolis-Adjusted Langevin Algorithm (MALA)**.
- Without correction, it’s **Unadjusted Langevin Algorithm (ULA)**.

---

# What makes MCMC powerful

- It works even when $p(x)$ is very complicated (multi-modal, high-dimensional).
- You only need to evaluate $p(x)$ up to a constant (e.g., no need to compute normalization).
- Widely used in Bayesian inference, physics simulations, deep generative models.

---

# The limitations

- Correlated samples (need to thin / burn-in).
- Slow mixing in high dimensions.
- Requires careful design of proposal/step size.

---

✅ **Summary in words**:  
MCMC is a family of methods where you build a random walk (Markov chain) that, in the long run, spends time in regions of space proportional to $p(x)$. That means the chain produces samples from $p(x)$, which you can then use for Monte Carlo estimation.