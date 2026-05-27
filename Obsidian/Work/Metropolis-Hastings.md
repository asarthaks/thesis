Metropolis–Hastings (MH) is one of the **core algorithms in [[Markov Chain Monte Carlo (MCMC)]]**, and it’s the ancestor of many modern methods (like Gibbs sampling, [[Langevin Dynamics]], HMC). Let’s carefully unpack it.

---

# 1. The problem

We want to sample from a complicated distribution $p(x)$ (say, a Bayesian posterior).  
But often:

- $p(x)$ is only known **up to a constant** (e.g. $p(x) \propto \exp(-E(x)$),
- and direct sampling is impossible.

So how can we get samples?

---

# 2. Key idea of MH

Build a **Markov chain** that has $p(x)$ as its stationary distribution.  
That means: after running long enough, the states of the chain look like samples from $p(x)$.

---

# 3. The algorithm

At each iteration, starting from current state $x_t$:

1. **Propose a move**:  
    Sample a candidate $x′$ from a proposal distribution $q(x' \mid x_t)$.  
    (e.g. add Gaussian noise: $x' = x_t + \epsilon \cdot \mathcal{N}(0,1)$).
    
2. **Compute acceptance ratio**:
    
    $r = \frac{p(x') \, q(x_t \mid x')}{p(x_t) \, q(x' \mid x_t)}$​.
3. **Accept/reject**:
    
    - With probability $\alpha = \min(1, r)$, set $x_{t+1} = x'$.
        
    - Otherwise, reject and keep $x_{t+1} = x_t$.
        

Repeat this many times.

---

# 4. Why this works

- If $q$ is symmetric (e.g. Gaussian), then the ratio simplifies to:
    
    $r = \frac{p(x')}{p(x_t)}$.
- This means moves toward regions of **higher probability** are likely accepted.
- Moves toward **lower probability** are sometimes accepted (ensuring exploration).

This **accept/reject rule** enforces _detailed balance_, which guarantees the chain’s stationary distribution is $p(x)$.

---

# 5. Intuition

- Imagine exploring a landscape where $p(x)$ is like a “height function.”
- MH says: propose random steps, but don’t always accept them.
- You linger longer in valleys of high probability, but you still occasionally climb out, so you don’t get stuck.

Over time, the fraction of visits to each region matches $p(x)$.

---

# 6. Example

Suppose we want to sample from:

$p(x) \propto \exp(-x^2/2)$ ,a standard normal.

- Proposal: $q(x' \mid x) = \mathcal{N}(x, \sigma^2)$.
- Acceptance ratio:
    $r = \frac{e^{-{x'}^2/2}}{e^{-x^2/2}} = e^{(x^2 - {x'}^2)/2}$.
- Accept with prob $min⁡(1,r)\min(1, r)min(1,r)$.

After many steps, the histogram of samples matches the Gaussian.

---

# 7. Why it’s powerful

- Works for any distribution, even if we only know $p(x)$ up to a constant.
- Flexible choice of proposal distributions.
- Foundation of many advanced MCMC algorithms.

---

✅ **In one line:**  
Metropolis–Hastings is an MCMC algorithm where you propose a move and then accept/reject it with a probability that guarantees your chain eventually samples from the target distribution $p(x)$.