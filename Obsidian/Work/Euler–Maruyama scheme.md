The **Euler–Maruyama scheme** is the stochastic cousin of the **Euler method** you may know from ordinary differential equations (ODEs). Let’s build the idea from the ground up.

---

## 1. Reminder: Euler method for ODEs

Suppose you have a deterministic system:

$dxdt=f(x),x(0)=x0.\frac{dx}{dt} = f(x), \quad x(0) = x_0.dtdx​=f(x),x(0)=x0​.$

The **Euler method** approximates the trajectory with discrete steps:

$xt+1=xt+Δt⋅f(xt).x_{t+1} = x_t + \Delta t \cdot f(x_t).xt+1​=xt​+Δt⋅f(xt​).$

This is just “current position + slope × step size.”

---

## 2. Now: Stochastic differential equations (SDEs)

In Langevin dynamics (and many other models), we don’t just have deterministic drift — we also have **random noise**.  
A simple SDE is:

$dxt=f(xt) dt+g(xt) dWt,dx_t = f(x_t)\,dt + g(x_t)\,dW_t,dxt​=f(xt​)dt+g(xt​)dWt​,$

where:

- $f(xt) dtf(x_t)\,dtf(xt​)dt$ = deterministic drift term (like gradient descent step).
    
- $g(xt) dWtg(x_t)\,dW_tg(xt​)dWt​$ = stochastic term, where $dWtdW_tdWt$​ is an infinitesimal increment of Brownian motion (random Gaussian noise).
    

This is no longer an ODE, but an **SDE**.

---

## 3. Euler–Maruyama scheme

To simulate an SDE numerically, Euler–Maruyama extends Euler’s method by adding a noise term:

$xt+1=xt+f(xt) Δt+g(xt) ΔWt,x_{t+1} = x_t + f(x_t)\,\Delta t + g(x_t)\,\Delta W_t,xt+1​=xt​+f(xt​)Δt+g(xt​)ΔWt​,$

where:

- $ΔWt∼N(0,Δt)\Delta W_t \sim \mathcal{N}(0, \Delta t)ΔWt​∼N(0,Δt)$ (Gaussian noise with variance proportional to the step size).
    

So it’s “deterministic step + random Gaussian kick.”

---

## 4. Example: Langevin dynamics

The Langevin SDE is:

$dxt=12∇log⁡p(xt) dt+dWt.dx_t = \tfrac{1}{2} \nabla \log p(x_t)\,dt + dW_t.dxt​=21​∇logp(xt​)dt+dWt​.$

Using Euler–Maruyama with step size ϵ\epsilonϵ:

$xt+1=xt+ϵ2∇log⁡p(xt)+ϵ ξt,ξt∼N(0,1).x_{t+1} = x_t + \tfrac{\epsilon}{2} \nabla \log p(x_t) + \sqrt{\epsilon}\,\xi_t, \quad \xi_t \sim \mathcal{N}(0,1).xt+1​=xt​+2ϵ​∇logp(xt​)+ϵ​ξt​,ξt​∼N(0,1).$

This is exactly the update rule you’ve seen for the **Unadjusted Langevin Algorithm (ULA)**.

---

## 5. Intuition

- **Euler method**: follow the slope deterministically.
    
- **Euler–Maruyama**: follow the slope + add a little Gaussian jiggle proportional to the step size.
    

This lets us simulate random processes like Brownian motion, Langevin dynamics, etc.

---

## 6. Limitations

- It’s a **first-order method**: errors shrink linearly with step size.
    
- If step size $ϵ\epsilonϵ$ is too large, simulation becomes inaccurate or unstable.
    
- For more precision, higher-order schemes exist (like Milstein’s method).
    

---

✅ **In one line:**  
The Euler–Maruyama scheme is the numerical method for simulating stochastic differential equations — it’s like Euler’s method for ODEs, but with an added Gaussian noise term to approximate Brownian motion.