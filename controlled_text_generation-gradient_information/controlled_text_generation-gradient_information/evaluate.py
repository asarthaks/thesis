import os
import random
import torch
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
from datasets import load_dataset
from core.prep import load_tokenizer_and_model, load_tokenizer_and_model_peft
from core.dls import DiscreteLangevinSampler
from core.cls import ContinuousLangevinSampler

def main():
    parser = argparse.ArgumentParser(description="Unified Langevin Dynamics Evaluation")
    parser.add_argument("--sampler", type=str, choices=["dls", "cls"], required=True)
    parser.add_argument("--num_masks", type=int, default=1)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--n_samples", type=int, default=250)
    parser.add_argument("--temperature", type=float, default=5.0)
    parser.add_argument("--method", type=str, default="policy", choices=["policy", "grad_norm_preserved_random_dir", "random"])
    
    # Booleans (use --flag to set to True)
    parser.add_argument("--oracle", action="store_true")
    parser.add_argument("--mh_sampling", action="store_true")
    parser.add_argument("--grad_norm", action="store_true")
    parser.add_argument("--use_peft", action="store_true")
    
    # Paths
    parser.add_argument("--model_path", type=str, default="//speech/dbwork/mul/spielwiese3/students/desinghs/models/gpt2_sft_output")
    parser.add_argument("--peft_path", type=str, default="/home/desinghs/projects/gfn-lm-tuning/infill_subj_arithmetic/outputs/2026-04-10/12-14-30/dummy-0wsnpfjl_1990.pt")
    
    args = parser.parse_args()

    # --- Load Model ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {args.sampler.upper()} on model {args.model_path}...")
    
    if args.use_peft:
        tokenizer, model = load_tokenizer_and_model_peft(args.model_path, args.peft_path, device=device)
    else:
        tokenizer, model = load_tokenizer_and_model(args.model_path, device=device)

    # --- Setup Sampler ---
    alpha_grid = np.logspace(-2, 2, 50) if args.sampler == "dls" else np.logspace(-4, -1, 50)
    SamplerClass = DiscreteLangevinSampler if args.sampler == "dls" else ContinuousLangevinSampler
    
    sampler = SamplerClass(
        model=model, tokenizer=tokenizer, steps=args.steps, temperature=args.temperature,
        oracle=args.oracle, alpha_grid=alpha_grid, method=args.method, 
        mh_sampling=args.mh_sampling, grad_normalization=args.grad_norm
    )

    # --- Load Dataset ---
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    filtered_data = [x['text'].strip() for x in dataset if 10 < len(x['text'].strip().split()) < 40]
    
    results_data =[]
    
    for i in tqdm(range(min(args.n_samples, len(filtered_data))), desc="Processing Samples"):
        text = filtered_data[i]
        input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
        seq_len = input_ids.shape[1]

        if seq_len < args.num_masks + 5: continue
        
        valid_indices = list(range(1, seq_len - 1))
        mask_indices = sorted(random.sample(valid_indices, args.num_masks))
        
        orig_ids = input_ids.clone().detach()
        original_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
        
        corrupted_ids = input_ids.clone()
        for idx in mask_indices:
            orig_id = input_ids[0, idx].item()
            rand_id = np.random.randint(0, tokenizer.vocab_size)
            while rand_id == orig_id: rand_id = np.random.randint(0, tokenizer.vocab_size)
            corrupted_ids[0, idx] = rand_id

        # --- Run Optimizer ---
        try:
            s_ids_history, metrics_history = sampler.optimize(corrupted_ids, mask_indices, orig_ids)
            
            final_pred_ids = s_ids_history[-1].tolist()
            original_token_ids = orig_ids[0, mask_indices].tolist()
            matches = sum([1 for p, t in zip(final_pred_ids, original_token_ids) if p == t])
            accuracy_pct = (matches / args.num_masks) * 100.0
            
            pred_seq = corrupted_ids.clone()
            pred_seq[0, mask_indices] = s_ids_history[-1]
            pred_text = tokenizer.decode(pred_seq[0], skip_special_tokens=True)

            results_data.append({
                "sample_idx": i,
                "accuracy_pct": accuracy_pct,
                "original_text": original_text,
                "predicted_text": pred_text,
                "avg_l2_dist": metrics_history[-1]["avg_l2_distance"],
                "avg_kl_div": metrics_history[-1]["avg_kl_divergence"],
                "trajectory": metrics_history
            })
            
        except Exception as e:
            print(f"Error on sample {i}: {e}")

    # --- Save CSV ---
    df = pd.DataFrame(results_data)
    os.makedirs("results", exist_ok=True)
    filename = f"results/exp_{args.sampler}_masks_{args.num_masks}_mh_{args.mh_sampling}_oracle_{args.oracle}_peft_{args.use_peft}_gn_{args.grad_norm}_{args.method}.csv"
    df.to_csv(filename, index=False)
    print(f"Saved to {filename}")

if __name__ == "__main__":
    main()