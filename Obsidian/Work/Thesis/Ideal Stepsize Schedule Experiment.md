The original experiment was to find a ideal schedule for the alpha/stepsize. It should ideally be going down.
The experiment setup was as follows: 
We have a number of steps.
We define a grid of alpha values.

For each Optimization Step:

For each value of alpha, we calculate the logprob based on equation 1 in the paper [[Langevin Like Sampler]]. We use the value of alpha, that gives us the max value of logprob for the ground truth token, and add that alpha to the alpha schedule.
basically:

        for alpha in alpha_grid:
            lp = compute_q_logprob(emb_gt, theta.detach(), grad, alpha)
            gt_logprobs.append(lp.item())

        assert len(gt_logprobs) == len(alpha_grid)
        
        # Normalize each step's logprob row BEFORE adding to matrix
        row = np.array(gt_logprobs.copy())
        row = (row - row.min()) / (row.max() - row.min() + 1e-12)
        all_gt_logprobs.append(row)

        gt_logprobs = np.array(gt_logprobs)
        best_idx = np.argmax(gt_logprobs)
        alpha_star = alpha_grid[best_idx]
        alpha_schedule.append(alpha_star)


Then, we will use the alpha* (the best alpha we got) and we compute the value for equation 2 from the paper [[Langevin Like Sampler]] to choose the next token.

We are calling this Oracle.
We perform this till we are done with all the optimization steps.
Ideally, this should be a big step size for the first optimization step and then zero for the rest of the steps because we are picking the best value of alpha, but this doesn't happen in the results (need to think why this happens, I think I knew the reason but kind of forgot)

On top of this experiments, we added other modifications like [[Gradient Information Experiment]], [[Temperature Sampling Modification]], [[Entropy Plotting]]

For the Continuous Langevin Sampler, the oracle will work like this: [[CLS Oracle]]

