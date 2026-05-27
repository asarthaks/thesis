Another thing we wanted to try was to see how important is the direction of gradient for this process of finding the correct/better token. For that, we tried an experiment. We used a random vector with the same norm as the gradient vector  instead of the actual gradient vector. the result was something that we didn't expect. we were still getting the correct token even when we were using the random vector. So, this essentially means that the gradient vector is not important at all.

This was performed with the [[Ideal Stepsize Schedule Experiment]]

There was a data leak I found in the random gradient experiment. for some reason, the initialization was the GT embedding plus some noise, idk how it got there. So, the method was choosing small values of alpha because the two terms in equation two are like direction - distance^2, so it was keeping small alpha to keep it near the GT embedding. 

05 Feb 2026 Meeting Points:
We want to setup this experiment for a dataset now. Choose a dataset of your choice, which is publicly available on huggingface and then setup the experiment for it.
The overall flow will be like this: 
Take first N samples, flip a random token, decode, report results.

Now, for the details: 
We want to report majorly these two metrices:
- L2/Squared distance (We are still not sure how accurate this is)
- KL divergence of the logprob from the Autoregressive Model

For the methods we want to cover, Majorly we will be covering three methods/policies.

1) DLP Policy: Our Langevin Like Sampler Method
2) Noisy with Grad Norm: We will replace the direction with random noise but preserve the norm of the gradient
3) Noise: We will just replace the gradient completely with noise. To Implement this, Lukas gave this hint/explanation that It will be from a gaussian distribution $N(x_k, \alpha)$, for a reference, the DLP policy will follow this gaussian distribution $N(x_k + \alpha\nabla(U(x_{k-1};\theta)), \alpha)$. So, I need to understand it first and then implement it.

And for all these three methods, I also have to keep the option for [[Metropolis-Hastings]] step after them.

Then, the loss is defined as Expectation of the loss observed for all the samples in the dataset, depending on the metrics we are choosing.

Plots:
Finally, we will plot the expected loss of these different methods vs the number of optimization steps.

This experiment will be used to answer the following research questions:
	Q1 : How much information is contained in the gradient following policy?
	Q2: How does it scale with respect to number of masked tokens?

Essentially, for Q2, we want to increase the number of masked token and see how these methods perform. For this, I'll need to make changes to the methods first and make them more unified and understand the math and flow of information so that I am 100% sure that the method is correct before running the experiments for multiple masked tokens.

Even right now, we will be using the Oracle (basically, we will "cheat", we will try to maximize the output probability of the GT token)


17th Feb

Got initial results, It seems like Euclidean Distance is not reliable in this space as the results for Euclidean Distance are completely opposite as compared to KL Divergence.

The results with 70 Samples experiment is showing good results that gradient signal does show improvement. Waiting for step wise result for 100 samples and then will update here and also discuss with Lukas

18th-19th Feb

Basically, the results are the same for Single Token Recovery and Multi Token recovery.

For single token recovery, the results are as follows: 
![[Pasted image 20260310135054.png]]

![[Pasted image 20260310135247.png]]

### 10th March 2026

Was not able to work last few weeks because of the layoff stuff going on  but now right on track

Got results for the multi token recovery using DLS as well and they are pretty similar to the single token recovery using DLS


![[Pasted image 20260310133411.png]]

![[Pasted image 20260310133420.png]]

In the plots I could notice two most prominent things: 1) For avg l2 distance the methods with no metropolis hastings are performing better than the one with metropois hastings (which shouldnt happen I think). I guess this could be due to the fact that l2 distance might not be a reliable metrics in this case because its so high dimensional. but this brings us to this fact that this method dls primarily uses eucledean distance for calculation of proposal tokens (check original paper of discrete langevin sampler), does this mean that this method is not useful at all, or do we need to change something in this? 2) In KL divergence plot we can see that random noise is performing better than the policy and grad norm noisy. Why is this happening?

^ This was the prompt given to gemini to understand why this could be happening and here was the speculation, which I kind of agree with for now:


#### 1. Why does "No MH" beat "With MH" on L2 Distance?

This happens because of the **"MH Gatekeeper Effect"** colliding with your **Oracle**.

- **What No-MH is doing:** In your code, you have oracle=True, which searches an α grid to find the step size that points directly toward the Ground Truth token. Without Metropolis-Hastings (MH), the algorithm blindly takes this step. It completely ignores whether the intermediate word makes grammatical sense. It just moves closer in L2 space.
- **What MH is doing:** MH acts as a strict grammar police. It checks the Joint Probability of the whole sentence. In LLMs, the loss landscape is extremely jagged. To get to the true word, you often have to pass through a "valley" of low-probability words. MH sees this temporary drop in sentence probability and **rejects the step** (if you look at your logs, mh_rejected is True almost constantly). Because it rejects the steps, it gets stuck at the corrupted token, resulting in a worse L2 distance.
    

**Is Euclidean (L2) distance flawed here?**  
Yes, partially. LLM embedding spaces are highly **anisotropic** (all vectors are clustered in a narrow cone, and distance doesn't perfectly map to semantic meaning). Moving 0.5 units in one direction might give you a synonym, but 0.5 units in another direction lands you on a <|reserved_special_token|>.

### 2. Why is Random Noise beating the Policy in KL Divergence?

This is the most revealing part of your data. To understand why, look at what the Policy without MH predicted in your sample output:

> ...underside,ustomer are lacking in { . gammarus.

It predicted garbage tokens like ustomer and {. Why? **Gradient Overshooting.**

- **The Policy:** The raw gradient from an LLM (true_grad) can be massive. When you take a step (s + alpha * grad), the massive gradient acts like a rocket thruster, shooting the embedding state far outside the "safe manifold" of normal vocabulary words. It lands in the outer edges of the space where special tokens and garbage fragments live. This completely destroys the context, causing massive KL Divergence.
    
- **The Random Method:** In your code, the random method uses torch.randn_like(grad_s). This generates numbers from a standard normal distribution (mean 0, std 1). These are **much smaller and safer** steps than the raw LLM gradient. It acts as a gentle random walk. It might not find the perfect word, but it stays near normal words, resulting in a more stable KL Divergence.



After these results and speculations, we made changes to the gradient that we are using for the DLS methods.

We normalized and rescaled the gradient according to this code: 



`log_joint = joint_log_prob_from_inputs_embeds(model, inputs_embeds, target_ids)

raw_grad_s = torch.autograd.grad(log_joint, s, retain_graph=False)[0]

  
if method == "policy":

    if grad_normalization:

        # Normalize along the embedding dimension (dim=1)

        grad_norm = raw_grad_s.norm(dim=1, keepdim=True) + 1e-12

        grad_s = raw_grad_s / grad_norm

    else:

        # Use raw gradient (original behavior)

        grad_s = raw_grad_s

elif method == "grad_norm_preserved_random_dir":

    # Generate a unit-length random direction vector

    rand_dir = torch.randn_like(raw_grad_s)

    rand_dir_unit = rand_dir / (rand_dir.norm(dim=1, keepdim=True) + 1e-12)

    if grad_normalization:

        # Scale to 1.0 (because the gradient is "normalized")

        grad_s = rand_dir_unit

    else:

        # Scale to match the massive norm of the raw gradient

        grad_norms = raw_grad_s.norm(dim=1, keepdim=True)

        grad_s = rand_dir_unit * grad_norms


elif method == "random":

    # Pure random walk

    rand_noise = torch.randn_like(raw_grad_s)

    if grad_normalization:

        # Force the random noise to also have a norm of 1.0 for a fair comparison!

        grad_s = rand_noise / (rand_noise.norm(dim=1, keepdim=True) + 1e-12)

    else:

        # Standard gaussian noise

        grad_s = rand_noise

else:

    raise ValueError(f"Unknown method: {method}")`

Right now waiting for the results, to see if there are any improvements, otherwise I will discuss with Lukas on why this is happening and what would be the next steps.



11th March 2026

After meeting with Lukas yesterday, in total 7 experiments were run.
They are as follows (All the experiments are with gradient normaliation and scaling on now):

1) DLS Multi Token Experiment (With Oracle)
	Basically, all experiments are now run with normalization and scaling of gradient as suggested by gemini.
	Let's look into the results:![[Pasted image 20260311185737.png]]
	![[Pasted image 20260311185741.png]]
	I would say that this makes a lot more sense now. Even though the results are not that good, atleast they make sense. The methods with metropolis hastings perform better and Policy has a better KL divergence score as it should be
	Result Filename: experiment_results_multi_sentences_samples_1000_date_2026-03-10.csv
	Code to visualize: PlottingMultiTokenRecoveryResults.ipynb
	
2) CLS Single Token Experiment (With Oracle)
	![[Pasted image 20260311190747.png]]
	![[Pasted image 20260311190822.png]]
	Gemini Says because of normalization, this is happening. Will run without normalization.
	Result Filename: experiment_results_cls_single_samples_100_date_2026-03-10.csv
	Code to visualize: PlottingEvaluateResults.ipynb
	
3) CLS Multi Token Experiment (With Oracle)
	For some reason, this did not end up finishing and is stuck. Reran the script on another server. Let's see.![[Pasted image 20260312152602.png]]
	![[Pasted image 20260312152607.png]]
	![[Pasted image 20260312152618.png]]
	
	Result Filename: experiment_results_cls_multi_samples_1000_date_2026-03-10.csv
	Code to visualize: PlottingMultiTokenRecoveryResults.ipynb
4) DLS Single Token Experiment (Without Oracle)
	![[Pasted image 20260311190949.png]]
	![[Pasted image 20260311191058.png]]
	In both single token and multi token experiments the methods are performing very close. This could be due to normalization as the step sizes are very small. I think I need to run all the without oracle experiments without normalization.
	Result Filename: experiment_results_dls_each_step_oracle_False_samples_100.csv
	Code to visualize: PlottingEvaluateResults.ipynb
5) DLS Multi Token Experiment (Without Oracle)
	![[Pasted image 20260311190501.png]]
	![[Pasted image 20260311190527.png]]
	This is similar to results without oracle, also, one peculiar thing is that after 40 steps, there is a huge crash in the metrics, Does that mean we need to run for more steps? I will run it for more steps to see how it performs.
	Result Filename: experiment_results_dls_multi_sentences_oracle_False_samples_1000_date_2026-03-10.csv
	Code to visualize: PlottingMultiTokenRecoveryResults.ipynb
6) CLS Single Token Experiment (Without Oracle)
	![[Pasted image 20260311191957.png]]
	![[Pasted image 20260311192003.png]]
	Same results, need to run without normalization
	Result Filename: experiment_results_cls_single_oracle_False_samples_100_date_2026-03-10.csv
	Code to visualize: PlottingEvaluateResults.ipynb
7) CLS Multi Token Experiment (Without Oracle)
	
	![[Pasted image 20260311192052.png]]
	![[Pasted image 20260311192058.png]]
	These are even weirder! Let's run them again without normalization
	Result Filename: experiment_results_cls_multi_oracle_False_samples_1000_date_2026-03-10.csv
	Code to visualize: PlottingMultiTokenRecoveryResults.ipynb

So, we will be running the following experiments without normalization:

1. CLS Single Token Experiment (With Oracle) - Ran
2. CLS Multi Token Experiment (With Oracle) - Ran
3. DLS Single Token Experiment (Without Oracle) - Ran
4. DLS Multi Token Experiment (Without Oracle) - Ran
5. CLS Single Token Experiment (Without Oracle) - Ran
6. CLS Multi Token Experiment (Without Oracle) - Ran

These are the jobs:
![[Pasted image 20260311193650.png]]

Will see the experiment results and discuss with Lukas tomorrow.

### 12th March 2026


1. CLS Single Token Experiment (With Oracle) 
	This was the first to finish executing because I ran it with only 100 samples.
	![[Pasted image 20260312153624.png|661]]
	![[Pasted image 20260313141719.png]]
	Without normalization, it's behaving really randomly I would say. The metropolis hastings methods are not performing well. Idk how that is related to the normalization step.
	Result File: experiment_results_cls_single_oracle_True_samples_100_date_2026-03-11.csv
	Code to visualize: PlottingEvaluateResults.ipynb
2. CLS Multi Token Experiment (With Oracle) 
	![[Pasted image 20260313140631.png]]
	![[Pasted image 20260313140635.png]]
	Idk why, without normalization, it's still very close together, this shouldn't happen.
	Result Filename: experiment_results_cls_multi_oracle_True_samples_1000_date_2026-03-11.csv
	Code to visualize: PlottingMultiTokenRecoveryResults.ipynb
3. DLS Single Token Experiment (Without Oracle)
	![[Pasted image 20260313141203.png]]
	![[Pasted image 20260313141211.png]]
	
4. DLS Multi Token Experiment (Without Oracle) 
	![[Pasted image 20260312154022.png]]
	![[Pasted image 20260312154041.png]]
	Result Filename: experiment_results_dls_multi_sentences_gn_False_oracle_False_samples_1000_date_2026-03-11.csv
	Code to visualize: PlottingMultiTokenRecoveryResults.ipynb
5. CLS Single Token Experiment (Without Oracle) - Running
6. CLS Multi Token Experiment (Without Oracle) 
	![[Pasted image 20260312154257.png]]
	![[Pasted image 20260312154303.png]]
	Result Filename: experiment_results_cls_multi_gn_False_oracle_False_samples_1000_date_2026-03-11.csv
	Code to visualize: PlottingMultiTokenRecoveryResults.ipynb

Need to run the experiment for more number of steps, especially number 4 to see what's happening exactly.

Need to debug CLS as well, I will do that tomorrow

One thing to note is that CLS doesn't work well with normalization whereas DLS works well with it. Works well in the sense, the MH combination methods in DLS performs better than the ones without it (Which is theoretically correct). In case of CLS, the normalization doesn't affect the difference between MH with/wo methods. They all perform almost the same, except for the fact that the policy one without MH diverge sometimes. This could be something I can look into.
Specifically CLS experiments without normalization seems to be having this divergent characteristic of the policy method without MH sampling. With normalization, they all seems to be performing the same.

This means, I need to debug more into CLS. I ran 8 new set of experiments with a proper MH setup (I reviewed the Metropolis hastings setup using gemini, gemini said it was incorrect as i suspected). I hope after this set of experiment, the policy method with metro polis hastings correct should also diverge from the noisy methods. 
Also, minor changes in the experimentation, I notices the above experiments were of 30 steps for CLS single token recovery and 50 steps for CLS multitoken recovery, I fixed it to 50 for both and I changed number of samples to 250 for all the methods.

The 8 experiments I ran are as follows:
![[Pasted image 20260313144739.png]]



1. CLS Single Token Experiment (With Oracle With Normalization) - Running
2. CLS Single Token Experiment (Without Oracle With Normalization) - Running
3. CLS Single Token Experiment (With Oracle Without Normalization) - Running
4. CLS Single Token Experiment (Without Oracle Without Normalization) - Running
5. CLS Multi Token Experiment (With Oracle With Normalization) - Running
6. CLS Multi Token Experiment (Without Oracle With Normalization) - Running
7. CLS Multi Token Experiment (With Oracle Without Normalization) - Running
8. CLS Multi Token Experiment (Without Oracle Without Normalization) - Running

One more set of experiments I needed to run was checking by increasing the number of steps for DLS, as there was some drop going on. Let me run that experiment as well.
This is the updated experiment run ids 
![[Pasted image 20260313150600.png]]
dls_of_m and dls_of_s are added
dls oracle off multi token and dls oracle off single token

One more experiment I need to do is to debug the output of the CLS single token recovery. Because, in visual evaluation on the test string, it was working well, but it doesn't seem to be working on the dataset. (Maybe in the evening today after kickboxing class)

One more to do, make code more streamlined, one single evaluate module which logs everything and can just pick the correct method by arguments and also the functions for single token recovery and multi token recovery should merge. Would like to do that preferably when I get some good results. Then I can cross check the results as well.



1. CLS Single Token Experiment (With Oracle With Normalization) - Done
	![[Pasted image 20260317132752.png]]	![[Pasted image 20260317132832.png]]
	Same as expected, with normalization, they seem to be performing the same
	Result Filename: experiment_results_cls_single_gn_True_oracle_True_samples_250_date_2026-03-13.csv
	Code to visualize: PlottingEvaluateResults.ipynb
2. CLS Single Token Experiment (Without Oracle With Normalization) - Done
	![[Pasted image 20260317133032.png]]
	![[Pasted image 20260317133057.png]]
	Same result as above
	Result Filename: experiment_results_cls_single_gn_True_oracle_False_samples_250_date_2026-03-13.csv
	Code to visualize: PlottingEvaluateResults.ipynb
3. CLS Single Token Experiment (With Oracle Without Normalization) - Done
	![[Pasted image 20260317133214.png]]
	![[Pasted image 20260317133236.png]]
	This is similar to our results from before, but, unfortunately, even after fixing the metropolis hastings correction step, the DLP policy method with MH correction doesn't seem to be diverging, the one without it seems to be performing the same as before. Okay, maybe this kind of makes sense, with oracle on, The KL divergence is not good because we are trying to move into the direction of the GT token, without taking into consideration the log likelyhood of the sentence.
	Result Filename: experiment_results_cls_single_gn_False_oracle_True_samples_250_date_2026-03-13.csv
	Code to visualize: PlottingEvaluateResults.ipynb
4. CLS Single Token Experiment (Without Oracle Without Normalization) - Done
	![[Pasted image 20260317133417.png]]
	![[Pasted image 20260317133535.png]]
	Okay, yeah, this is what I was talking about. This KL also gets low when the oracle is false. This is good.
	Result Filename: experiment_results_cls_single_gn_False_oracle_False_samples_250_date_2026-03-13.csv
	Code to visualize: PlottingEvaluateResults.ipynb
5. CLS Multi Token Experiment (With Oracle With Normalization) - Done
	![[Pasted image 20260317133902.png]]
	![[Pasted image 20260317133906.png]]
	With normalization, they perform the same. No need to look
	Result Filename: experiment_results_cls_multi_gn_True_oracle_True_samples_250_date_2026-03-13.csv
	Code to visualize: PlottingMultiTokenRecoveryResults.ipynb
6. CLS Multi Token Experiment (Without Oracle With Normalization) - Done
	![[Pasted image 20260317134210.png]]
	![[Pasted image 20260317134215.png]]
	Same as above
	Result Filename: experiment_results_cls_multi_gn_True_oracle_False_samples_250_date_2026-03-13.csv
	Code to visualize: PlottingMultiTokenRecoveryResults.ipynb
7. CLS Multi Token Experiment (With Oracle Without Normalization) - Done
	![[Pasted image 20260317134403.png]]
	![[Pasted image 20260317134409.png]]
	Similar to single token results
	Result Filename: experiment_results_cls_multi_gn_False_oracle_True_samples_250_date_2026-03-13.csv
	Code to visualize: PlottingMultiTokenRecoveryResults.ipynb
8. CLS Multi Token Experiment (Without Oracle Without Normalization) - Done
	![[Pasted image 20260317134502.png]]
	![[Pasted image 20260317134506.png]]
	This is good, but for some reason the MH sampling one still  is not performing well.
	Result Filename: experiment_results_cls_multi_gn_False_oracle_False_samples_250_date_2026-03-13.csv
	Code to visualize: PlottingMultiTokenRecoveryResults.ipynb


To sum up these results. The CLS method does work well. but for some reason, the MH sampling correction with it doesn't work well. I can understand why it doesn't work well with oracle, but even withjout oracle, iot doens';t seem to be working well.

Need to discuss it with lukas.

Lets see the 2nd set of experiments:

dls_of_m and dls_of_s are added

1) DLS Single token Normalization True Oracle False 100 steps
	![[Pasted image 20260317140424.png]]![[Pasted image 20260317140436.png]]
	Nothing worth noting I would say
	Result Filename: experiment_results_dls_steps_100_gn_True_oracle_False_samples_250_date_2026-03-13.csv
	Code to visualize: PlottingEvaluateResults.ipynb
2) DLS Multi Token Normalization True Oracle False 100 Steps
	![[Pasted image 20260317140538.png]]
	![[Pasted image 20260317140542.png]]
	Similar Results
	Result Filename: experiment_results_dls_multi_sentences_steps_100_gn_True_oracle_False_samples_250_date_2026-03-13.csv
	Code to visualize: PlottingEvaluateResults.ipynb


Read paper about GFlownets by Moksh Jain




More experiments on gpt2 sft (on rocstories) and then peft for gflow subtf tuning
the plotting notebooks have been merged and contains all config now:

![[Pasted image 20260416162119.png]]
![[Pasted image 20260416162129.png]]
![[Pasted image 20260416162138.png]]
![[Pasted image 20260416162142.png]]
![[Pasted image 20260416162146.png]]
![[Pasted image 20260416162151.png]]
![[Pasted image 20260416162155.png]]
![[Pasted image 20260416162159.png]]


![[Pasted image 20260421144014.png]]

![[Pasted image 20260428093838.png]]
![[Pasted image 20260428093842.png]]
![[Pasted image 20260428093848.png]]
![[Pasted image 20260428093854.png]]
![[Pasted image 20260428093900.png]]
![[Pasted image 20260428093904.png]]
![[Pasted image 20260428093910.png]]
![[Pasted image 20260428093914.png]]


4 may
![[Pasted image 20260504144918.png]]

![[Pasted image 20260505140417.png]]
![[Pasted image 20260505140422.png]]
![[Pasted image 20260505140427.png]]
![[Pasted image 20260505140434.png]]
![[Pasted image 20260505140439.png]]
![[Pasted image 20260505140445.png]]
![[Pasted image 20260505140450.png]]
![[Pasted image 20260505140457.png]]
