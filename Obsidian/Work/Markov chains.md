A **Markov chain** is a random process that evolves step by step:

$x_{t+1} \sim T(x_t, \cdot),$

where $T$ is a transition rule (probability of going from $x_t​$ to a new state).

- **Memoryless**: the next state depends only on the current state.
    
- If designed well, after many steps the chain “forgets” where it started and **settles into a stationary distribution**.
    
- If we design T so the stationary distribution is $p(x)$, then running the chain gives us samples _from $p$. 🎉