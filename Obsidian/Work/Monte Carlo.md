

“Monte Carlo” just means: use **random sampling** to approximate things.  
For example, to estimate $E[f(X)]\mathbb{E}[f(X)]E[f(X)]:$

$E[f(X)]≈1N∑i=1Nf(xi),xi∼p(x).\mathbb{E}[f(X)] \approx \frac{1}{N}\sum_{i=1}^N f(x_i), \quad x_i \sim p(x).E[f(X)]≈N1​i=1∑N​f(xi​),xi​∼p(x).$

But this only works if we _can_ draw samples $xi∼p(x)x_i \sim p(x)xi​∼p(x).$