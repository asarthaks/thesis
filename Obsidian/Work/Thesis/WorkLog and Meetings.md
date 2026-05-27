
## Prior Work
We were working with [[Langevin Dynamics]] and [[Langevin Like Sampler]]

We coded the [[Langevin Dynamics]] Method for continuous space and [[Langevin Like Sampler]] for discrete space

Plotted a heatmap for the the schedule, worked on dimensionality reduction to plot the embedding space and trajectory.

As far as I remember during the last couple of meetings, we were discussing how the gradient landscape looked for the embeddings. 

 We discussed the results of [[Gradient Information Experiment]] and [[Temperature Sampling Modification]]

### 5th Feb 2026 Meeting

Going to have a meeting with Lukas, before this I was working on the proposal.
Submitted the proposal to Professor Dr. Thang Vu, I might need to change the template, I sent the proposal in the wrong template. 

My thoughts:
I think the results of [[Gradient Information Experiment]] are like that because, in the [[Ideal Stepsize Schedule Experiment]], we are taking the value of alpha, which will give maximize the logprob for the GT token, which obviously means we will get the correct token even if we are using the wrong direction. But this says something about the embedding space that, even if we are moving in the wrong direction, by just choosing the correct step size, we will get the correct token, which essentially means that direction is just not important in this high dimensional space. Which I am not sure if it is correct or not. I checked the code, the code seems correct. 

Need to decide next plan of action with Lukas


Found the issue, updated in [[Gradient Information Experiment]]

Meeting notes:
Lukas asked to manage everything in GitLab repo properly, I'll start doing that
with issues and file names which contains which code noted down in markdown
I already have the repo, i cloned it. I'll update everything tomorrow and also code the next set of experiments. Also, he suggested to make scripts for the code that has been finalized, so that I can run that in slurm easily.

Next set of Tasks:


Defined the [[Gradient Information Experiment]] in more detail and with more research questions.

## 06 Feb 2026

Started setting up GitLab
elaborated about the next set of experiments in obsidian and GitLab both
updated the code for discrete langevin sampler:
added argumments for oracle and different methods that we want to compare



## 9th Feb

Updated the code for discrete langevin sampler. (updated the calculate for mh sampling (temp scaled logits))
was trying out with different hyperparameters to make the sampler work, didnt work as in wasnt able to get it to decode the proper token, there was another issue with the score becoming zero for some tokens. Was debugging that

## 10th Feb
Will continue to debug the zero score issue 

Even while trying with different hyperparameters, Still not able to decode the correct token, even with the MH sampling. In the continuous one, we were able to sample the correct token even without using MH sampling.

I kind of understand why the score is zero. But still I need to discuss it with Lukas if its correct
I tried with temp sampling >1 and the results are kind of fine i would say, for now. Now I'll check the oracle  part

Oracle works well too, specially with metropolis Hastings sampling. 
Now, I'll setup the experiment.

## 17th Feb

Wrote the script for whole evaluation and fixed issues with imports (kind of used a hack for now)

Also, need to optimize the method for the calculation of KL divergence. (done)

The script is ready and ran the script for evaluating the method on the dataset WikiText-2

current latest code for evaluation is in the file evaluate_dls_dataset.py and the code for plotting is in the notebook PlottingEvaluateResults.ipynb 

I ran a sbatch script for running the evaluate for 100 samples and shown the current results for 70 samples (only final metric results, not each step) and he asked to wait for the final results.
I did ran the experiment with logging of metrics for each step of optimization for 2 samples, the MH sampling was not showing good results with policy there which was weird,. But it could be because of the sample as well. I will wait for the 100 sample experiment result.

Next step I have to do is to make it run for multiple masked tokens and see how does it scale, that's the next major step.

If there no major issue (like the MH sampling not being better or showing some other issue), this means that gradients are definitely important for a better output. But maybe they are not that good.

Then we can move to the next phase that is trying out newer methods to improve the current method.
More specifically, we want to train a smaller network that can give us better proposals.
For that, we will try to train a smaller network (like smaller diffusion model or [[GFlowNets]]). For this, Lukas suggested me some resources to look into myself first, then he might give an overview as well.

Resources: 
Gflownets.org (It'll give me an overview)
Foundation Paper (Too much math according to Lukas)
AwesomeGFlowNets Git Repository (It contains important papers related to it)

What we would ideally want is a online training setup with a reinforcement learning setup

There is another paper that is very closely linked to what we want to do, its a must read. I'll go through it.: "AMORTIZING INTRACTABLE INFERENCE IN LARGE LANGUAGE MODELS ([2310.04363](https://arxiv.org/pdf/2310.04363))"


## 18th Feb
 So, there was some issue with the code I asked Gemini to create for the multi token recovery, I am trying to fix that. Also waiting on Lukas's response on Yesterday's result on the dataset evaluation. The results were not up to the mark, more specifically the MH was not as good as the non MH ones, which is weird and also the plots were kind of weird. Need to debug them I guess.

## 19th Feb

Today's work
created the script for multi masked token experiment and now trying to run the script
will discuss results with Lukas tomorrow if this works right now.


### 3rd March

had a meeting with Lukas, regarding thesis registration and Another Thing I brought up in front of Lukas  was that out continuous Langevin sampler was working good. So, we wanted to evaluate the method on the same dataset so we can compare the two methods as well. 

### 9th March
Need to get results for CLS eval
and multi-token results


### 10th March
Interpreted the results for the multi-token recovery eval.

The interpretations and next steps are written in [[Gradient Information Experiment]]
Basically, I made some changes in gradient normalization and ran the experiments again.

Also, setup the code for CLS script that can be imported and setup a script for its evaluations and I also setup an oracle for CLS, described in [[CLS Oracle]].

The code is running and most probably by tomorrow we will have results for all three experiments.

Need to run DLS and CLS Eval without oracle


### 11th March

After discussion with Lukas on 10th march, 7 experiments were run in total. They are mentioned in [[Gradient Information Experiment]]
Looked at the results and analyzed and ran 6 (one experiment is still stuck) of the experiments again with gradient normalization and scaling as false on suggestion from Gemini. Let's see how the results are. 
