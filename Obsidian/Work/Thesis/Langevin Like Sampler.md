The paper "A Langevin-like Sampler for Discrete Distributions" by Zhang et al 2022



Proposal Distribution (logprob)

$$
q(\theta' \mid \theta)
=
\frac{
\exp\!\left(
-\frac{1}{2\alpha}
\left\|
\theta' - \theta - \frac{\alpha}{2}\nabla U(\theta)
\right\|_2^2
\right)
}{
Z_{\Theta}(\theta)
}
\tag{1}
$$

$$
Z_{\Theta}(\theta)
=
\sum_{\theta' \in \Theta}
\exp\!\left(
-\frac{1}{2\alpha}
\left\|
\theta' - \theta - \frac{\alpha}{2}\nabla U(\theta)
\right\|_2^2
\right)
$$


Coordinate wise factorization
$$
q(\theta' \mid \theta)
=
\prod_{i=1}^{d}
q_i(\theta'_i \mid \theta),
$$

$$
q_i(\theta'_i \mid \theta)
=
\text{Categorical}\!\left(
\text{Softmax}\!\left(
\frac{1}{2}\nabla U(\theta)_i(\theta'_i - \theta_i)
-
\frac{(\theta'_i - \theta_i)^2}{2\alpha}
\right)
\right)
\tag{2}
$$
