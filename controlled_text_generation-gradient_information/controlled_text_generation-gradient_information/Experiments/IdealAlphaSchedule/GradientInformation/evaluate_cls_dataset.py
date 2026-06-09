import sys
import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
import datetime

# -----------------------------------------------------------------------------
# 1. Path Setup
# -----------------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../.."))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

# -----------------------------------------------------------------------------
# 2. Imports
# -----------------------------------------------------------------------------
# Importing from CLS instead of DLS
from Methods.Scripts.cls import langevin_infilling_single_position
from Methods.Utils.prep import load_tokenizer_and_model, load_tokenizer_and_model_peft

# -----------------------------------------------------------------------------
# 3. Experiment Config
# -----------------------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_NAME = "gpt2_sft"
# MODEL_NAME = "llama3_8b"

# MODEL_PATH = "/speech/dbwork/mul/spielwiese3/llm_project/llm_models/pre-trained/Meta-Llama-3-8b/"
MODEL_PATH = "//speech/dbwork/mul/spielwiese3/students/desinghs/models/gpt2_sft_output"
# FALLBACK_ID = "meta-llama/Meta-Llama-3-8B"
PEFT_CHECKPOINT_PATH = "/home/desinghs/projects/gfn-lm-tuning/infill_subj_arithmetic/outputs/2026-04-10/12-14-30/dummy-0wsnpfjl_1990.pt"
USE_PEFT = True

N_SAMPLES = 250     # How many dataset samples to process
STEPS = 50
TEMPERATURE = 5
ALPHA_GRID = np.logspace(-4, -1, 50)

# CLS-Specific Parameters
GRAD_NORMALIZATION = False
NOISE_SCALE = 0.01
MH_BURN_IN = 0
DATE = str(datetime.date.today())
METHODS =[
    "policy",
    "grad_norm_preserved_random_dir", 
    "random"
]
ORACLE_PARAM = False
MH_VARIATIONS = [True, False] 



def run_experiment():
    print(f"Loading model from {MODEL_PATH}...")
    
    if USE_PEFT:
        tokenizer, model = load_tokenizer_and_model_peft(MODEL_PATH, PEFT_CHECKPOINT_PATH, device=device)
    else:
        tokenizer, model = load_tokenizer_and_model(MODEL_PATH, device=device)
    

    print("Loading Wikitext-2 dataset...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    
    filtered_data = [x['text'] for x in dataset if len(x['text']) > 100]
    print(f"Found {len(filtered_data)} valid samples. Processing the first {N_SAMPLES}...")

    results_data =[]

    for i in tqdm(range(min(N_SAMPLES, len(filtered_data))), desc="Samples"):
        text_sample = filtered_data[i]
        
        toks = tokenizer(text_sample, return_tensors="pt")
        input_ids = toks.input_ids.to(device)
        seq_len = input_ids.shape[1]

        if seq_len < 5: 
            continue

        target_idx = np.random.randint(1, seq_len - 1)
        position = (target_idx, target_idx + 1)

        original_token_id = input_ids[0, target_idx].item()
        original_token_str = tokenizer.decode([original_token_id])

        clean_ref_ids = input_ids.clone().detach()
        
        corrupted_input_ids = input_ids.clone()
        vocab_size = tokenizer.vocab_size
        
        random_noise_token = np.random.randint(0, vocab_size)
        while random_noise_token == original_token_id:
            random_noise_token = np.random.randint(0, vocab_size)
            
        corrupted_input_ids[0, target_idx] = random_noise_token
        corrupted_token_str = tokenizer.decode([random_noise_token])

        for method_name in METHODS:
            for use_mh in MH_VARIATIONS:
                
                current_input = corrupted_input_ids.clone()
                
                try:
                    # Run the Continuous Langevin Dynamics
                    token_ids_trace, metrics_history = langevin_infilling_single_position(
                        model=model,
                        input_ids=current_input,
                        position=position,
                        tokenizer=tokenizer,
                        steps=STEPS,
                        temperature=TEMPERATURE,
                        oracle=ORACLE_PARAM,
                        orig_ids=clean_ref_ids, 
                        alpha_grid=ALPHA_GRID,
                        method=method_name,
                        mh_sampling=use_mh,
                        grad_normalization=GRAD_NORMALIZATION,
                        debug=False,
                        mh_burn_in=MH_BURN_IN,
                        noise_scale=NOISE_SCALE
                    )

                    final_pred_id = token_ids_trace[-1]
                    if isinstance(final_pred_id, torch.Tensor):
                        final_pred_id = final_pred_id.item()
                    
                    final_pred_str = tokenizer.decode([final_pred_id])
                    success = (final_pred_id == original_token_id)
                    final_metrics = metrics_history[-1] 


                    results_data.append({
                        "sample_idx": i,
                        "method": method_name,
                        "mh_enabled": use_mh,
                        "original_token": original_token_str,
                        "corrupted_token": corrupted_token_str,
                        "predicted_token": final_pred_str,
                        "success": success,
                        "l2_dist": final_metrics["l2_distance"],
                        "kl_div": final_metrics["kl_divergence"],
                        "trajectory": metrics_history 
                    })

                except Exception as e:
                    print(f"Error on Sample {i} | Method {method_name} | MH {use_mh}: {e}")

    df = pd.DataFrame(results_data)
    
    print("\n========== RESULTS SUMMARY ==========")
    print(df.head(10))

    print("\n========== ACCURACY BY CONFIGURATION ==========")
    summary = df.groupby(["method", "mh_enabled"])["success"].mean()
    print(summary)
    
    # Save as _cls_ to avoid overwriting dls
    output_file = os.path.join(current_dir, f"results/experiment_results_cls_single_steps_{STEPS}_gn_{GRAD_NORMALIZATION}_oracle_{ORACLE_PARAM}_samples_{N_SAMPLES}_date_{DATE}_peft_{USE_PEFT}_model_name_{MODEL_NAME}.csv")
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    run_experiment()