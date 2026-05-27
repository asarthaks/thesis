## 1. Continuous vs. discrete evolution

Imagine we have a **continuous process** (an SDE or ODE). For example:

dxdt=f(x).\frac{dx}{dt} = f(x).dtdx​=f(x).

That means: at every infinitesimal moment in time, xxx changes smoothly according to f(x)f(x)f(x).  
If we could solve it exactly, we’d know the _true_ trajectory x(t)x(t)x(t).

---

## 2. Discretization: approximating with steps

On a computer we **can’t handle infinitesimal time**.  
So we approximate using small steps of size ϵ\epsilonϵ:

xt+1≈xt+ϵf(xt).x_{t+1} \approx x_t + \epsilon f(x_t).xt+1​≈xt​+ϵf(xt​).

This is the **Euler method**. It’s an approximation to the continuous dynamics.

For stochastic processes (like Langevin), we do the same with the **Euler–Maruyama scheme**:

xt+1=xt+ϵ2∇log⁡p(xt)+ϵ ξt,ξt∼N(0,1).x_{t+1} = x_t + \frac{\epsilon}{2}\nabla \log p(x_t) + \sqrt{\epsilon}\,\xi_t, \quad \xi_t \sim N(0,1).xt+1​=xt​+2ϵ​∇logp(xt​)+ϵ​ξt​,ξt​∼N(0,1).

---

## 3. Why does this introduce error?

Two main reasons:

### a) **Local truncation error**

- In the true dynamics, the path evolves continuously.
    
- In the discrete version, we “jump” in chunks of size ϵ\epsilonϵ.
    
- The jump is only a **first-order approximation**. Higher-order curvature in the dynamics is ignored.
    

Analogy: tracing a smooth curve with straight-line segments. The bigger the step, the more jagged (and inaccurate) the path.

---

### b) **Stationary distribution mismatch**

- The continuous Langevin SDE has the **exact posterior** p(x)p(x)p(x) as its stationary distribution.
    
- The discretized version (ULA) has a _different_ stationary distribution, shifted because of the discretization error.
    
- Only as ϵ→0\epsilon \to 0ϵ→0 does the discrete chain’s stationary distribution converge to the true posterior.
    

This is why ULA samples are _biased_ unless the step size is tiny.

---

## 4. How Metropolis–Hastings fixes it

- ULA step = approximate move.
    
- MH accept/reject = corrects for the approximation, ensuring the chain has exactly the right stationary distribution p(x)p(x)p(x).
    
- This is the **Metropolis-Adjusted Langevin Algorithm (MALA)**.
    

So MH acts like a “correction mechanism” that removes discretization error.

---

## 5. Simple analogy

Imagine you want to walk along a curved road:

- **Exact dynamics**: follow the curve perfectly.
    
- **Discretization**: walk in straight lines that approximate the curve. If your steps are big, you drift off-road.
    
- **Error**: the path you trace is not the true curve.
    
- **MH correction**: after each step, check whether you’re still “on the road.” If not, reject the move and stay put.
    

---

✅ **In short:**  
Discretization means replacing smooth continuous dynamics with finite steps. This makes simulation possible, but introduces error because the steps only approximate the true evolution. In Langevin dynamics, that error shows up as the sampler converging to the _wrong distribution_ — unless step size is very small, or we add an MH correction.