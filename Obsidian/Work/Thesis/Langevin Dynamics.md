
## 1. Physics origin

Langevin dynamics comes from physics — describing the motion of a tiny particle (like a pollen grain in water).

- The particle is pulled by deterministic **forces** (e.g. friction, potential energy).
    
- At the same time, it’s buffeted randomly by water molecules = **noise**.
    

The Langevin equation (in simplest 1D form) is:

$\frac{dx}{dt} = -\nabla U(x) + \text{random noise},$

where:

- U(x) is a potential energy function (like a landscape).
    
- The particle tends to move downhill (towards lower energy).
    
- Noise keeps it wandering around instead of getting stuck.
    

---

## 2. Probabilistic interpretation

Now suppose we want to sample from a probability distribution:

$p(x) \propto e^{-U(x)}.$

That is: high probability where energy U(x) is low.

It turns out that if a particle follows Langevin dynamics in the potential U(x), then in the long run its position xx is distributed according to p(x).

So **Langevin dynamics gives us a way to sample from complex distributions**.

---

## 3. Discretized update (the algorithm)

We can’t solve differential equations exactly on a computer, so we simulate them with small steps (Euler–Maruyama scheme).

The **Unadjusted Langevin Algorithm (ULA)** update is:

$x_{t+1} = x_t + \frac{\epsilon}{2} \nabla \log p(x_t) + \sqrt{\epsilon}\,\xi_t,$

where:

- ϵ = step size (small).
    
- $\nabla \log p(x_t)$ = “score function,” gradient of log probability (directs the particle toward high probability regions).
    
- $\xi_t \sim N(0,1)$ = Gaussian noise (keeps exploration going).
    

---

## 4. Intuition

Think of it as:

- Gradient ascent on $\log p(x)$: move uphill towards more likely regions.
    
- **Plus Gaussian noise**: prevents you from collapsing to a single mode and ensures exploration.
    

Over many iterations, the trajectory of $x_t$ produces samples distributed like $p(x)$.

---

## 5. Why it matters

- In **Bayesian inference**, $p(x)$ might be a posterior distribution — Langevin lets you sample from it.
    
- In **machine learning**, this idea led to **Stochastic Gradient Langevin Dynamics (SGLD)**, which is like SGD with added noise, useful for approximate Bayesian learning.
    
- In **physics**, it models diffusion, Brownian motion, and thermal equilibrium.
    

---

## 6. Toy picture

Imagine a ball rolling on a hilly landscape:

- Gradient pulls it downhill into valleys (high probability regions).
    
- Random kicks from noise keep it exploring instead of sitting at the very bottom.
    
- Over time, the ball spends more time in valleys that correspond to high-probability regions under $p(x)$.
    

That’s Langevin dynamics!

---

✅ **In one line:**  
Langevin dynamics is a method to generate samples from a probability distribution by combining **gradient ascent on log-probability** with **random noise**, inspired by the physics of particles in a fluid.
