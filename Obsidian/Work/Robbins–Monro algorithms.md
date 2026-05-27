Robin–Monro algorithms (often called **Robbins–Monro algorithms**) are a cornerstone in **stochastic approximation**, which is one of the mathematical roots of modern optimization and sampling. Let’s break it down step by step.

---

## 1. The problem Robbins & Monro wanted to solve

Suppose you want to solve an equation of the form:

$h(\theta) = \alpha$

but you **can’t evaluate $h(θ)$ exactly**.  
Instead, you can only observe **noisy samples** of it.

For example:

- $h(\theta) = \mathbb{E}[X \mid \theta]$
- You can simulate random $X$, but not compute the expectation exactly.

This happens all the time in statistics and ML.

---

## 2. The Robbins–Monro idea (1951)

Instead of solving the equation directly, **iteratively update your guess** for $\theta$ using noisy observations.

Update rule:

$\theta_{n+1} = \theta_n - a_n \big( Y_n - \alpha \big)$

where:

- $Y_n​$ is a noisy observation of $h(\theta_n)$.
- $a_n​$ is a step size (learning rate), usually decreasing like $1/n$.

This is a **stochastic iterative algorithm** that converges to the true root under mild conditions.

---

## 3. Why this is important

This is one of the **first algorithms for stochastic optimization**:

- If you set $h(\theta) = \nabla_\theta L(\theta)$, you recover **stochastic gradient descent (SGD)**.
- Robbins–Monro is the ancestor of all SGD-based methods used in ML today.

---

## 4. Robbins–Monro in Langevin & MCMC

- When we run **stochastic gradient Langevin dynamics (SGLD)**, we are essentially combining **Robbins–Monro–type updates** (noisy gradient descent) with Langevin noise.
- The step size schedule ana_nan​ (decreasing slowly to 0) is critical for convergence in both Robbins–Monro and SGLD.

---

## 5. A concrete example

Suppose we want to find the root of:

$h(\theta) = \mathbb{E}[X] - 5 = 0$

but we only see noisy samples of $X \sim \text{Poisson}(\theta)$.

We can’t solve for θ\thetaθ exactly because we only get random draws.

Using Robbins–Monro:

1. Start with a guess $\theta_0​.$
    
2. At each step:
    
    - Sample $X_n \sim \text{Poisson}(\theta_n)$.
        
    - Update:
        $\theta_{n+1} = \theta_n - a_n (X_n - 5)$
    - with $a_n = 1/n$.
        

Over time, $\theta_n$ converges to the true root: $\theta=5$. 🎉

---

## 6. Key takeaways

- **Robbins–Monro algorithm** is the _first stochastic approximation method_.
- It solves root-finding problems when you only get noisy evaluations.
- It’s the theoretical foundation behind:
    - **Stochastic gradient descent (SGD)**.
    - **Stochastic optimization in ML**.
    - **Stochastic gradient Langevin dynamics (SGLD)**.

---

✅ **In one line:**  
Robbins–Monro algorithms are early iterative methods for solving equations with noisy observations. They are the mathematical grandparents of SGD and modern stochastic optimization/sampling methods.