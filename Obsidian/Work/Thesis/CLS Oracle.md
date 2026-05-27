Here is how we should handle the Oracle in the Continuous Langevin Sampler (CLS), followed by the complete, restructured cls.py file.

### How the Oracle works in Continuous Space (CLS)

In your Discrete Sampler (DLS), the Oracle tests different step sizes ($α$) to find the one that maximizes the categorical probability of jumping to the Ground Truth (GT) token.

In the Continuous Sampler (CLS), your proposal is not a categorical probability, but a **Gaussian Distribution**. Specifically, the next continuous state $s_u$ is drawn from:  $s_u∼N(m_s(α),α⋅noise$_$scale⋅I)$

Where $m_s(α)$ is your heuristic mean: $0.5⋅(interim+interim$_proj)$

**The Oracle Suggestion:**  
To build an Oracle for CLS, we should evaluate the **Gaussian log-density** of the Ground Truth continuous embedding ($e_{gt}$) for every α in your grid. The Oracle will pick the α that makes $e_{gt}$ the most likely under that specific Gaussian curve.