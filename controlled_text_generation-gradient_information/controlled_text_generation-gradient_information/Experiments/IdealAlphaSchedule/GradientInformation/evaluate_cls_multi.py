import sys
import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
import random
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
from Methods.Scripts.cls import langevin_infilling_multiple_positions
from Methods.Utils.prep import load_tokenizer_and_model, load_tokenizer_and_model_peft

# -----------------------------------------------------------------------------
# 3. Experiment Configuration
# -----------------------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_NAME = "gpt2_sft"
# MODEL_NAME = "llama3_8b"

# MODEL_PATH = "/speech/dbwork/mul/spielwiese3/llm_project/llm_models/pre-trained/Meta-Llama-3-8b/"
MODEL_PATH = "//speech/dbwork/mul/spielwiese3/students/desinghs/models/gpt2_sft_output"
# FALLBACK_ID = "meta-llama/Meta-Llama-3-8B"
PEFT_CHECKPOINT_PATH = "/home/desinghs/projects/gfn-lm-tuning/infill_subj_arithmetic/outputs/2026-04-10/12-14-30/dummy-0wsnpfjl_1990.pt"
USE_PEFT = True

# Settings
N_SAMPLES = 250       
NUM_MASKS = 2          
STEPS = 50
TEMPERATURE = 5
ALPHA_GRID = np.logspace(-4, -1, 50)
DATE = str(datetime.date.today())

# CLS-Specific Parameters
GRAD_NORMALIZATION = False
NOISE_SCALE = 0.01
MH_BURN_IN = 0

METHODS =[
    "policy",
    "grad_norm_preserved_random_dir", 
    "random"
]
ORACLE_PARAM = False
MH_VARIATIONS = [True, False]

def run_multi_mask_experiment():
    print(f"Loading model from {MODEL_PATH}...")
    
    if USE_PEFT:
        tokenizer, model = load_tokenizer_and_model_peft(MODEL_PATH, PEFT_CHECKPOINT_PATH, device=device)
    else:
        tokenizer, model = load_tokenizer_and_model(MODEL_PATH, device=device)
    

    print("Loading Wikitext-2 dataset...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    
    filtered_data = []
    for x in dataset:
        text = x['text'].strip()
        word_count = len(text.split())
        if 10 < word_count < 40:
            filtered_data.append(text)
            
    print(f"Found {len(filtered_data)} 'normal length' samples. Processing first {N_SAMPLES}...")

    results_data =[]

    for i in tqdm(range(min(N_SAMPLES, len(filtered_data))), desc="Processing"):
        text_sample = filtered_data[i]
        
        toks = tokenizer(text_sample, return_tensors="pt")
        input_ids = toks.input_ids.to(device)
        seq_len = input_ids.shape[1]

        if seq_len < (NUM_MASKS + 5): 
            continue

        valid_indices = list(range(1, seq_len - 1))
        if len(valid_indices) < NUM_MASKS: continue
        mask_indices = sorted(random.sample(valid_indices, NUM_MASKS))
        
        original_token_ids = input_ids[0, mask_indices].tolist()
        original_full_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)

        corrupted_input_ids = input_ids.clone()
        vocab_size = tokenizer.vocab_size
        
        for idx in mask_indices:
            orig_id = input_ids[0, idx].item()
            rand_id = np.random.randint(0, vocab_size)
            while rand_id == orig_id: rand_id = np.random.randint(0, vocab_size)
            corrupted_input_ids[0, idx] = rand_id

        corrupted_full_text = tokenizer.decode(corrupted_input_ids[0], skip_special_tokens=True)
        
        clean_ref_ids = input_ids.clone().detach()

        for method_name in METHODS:
            for use_mh in MH_VARIATIONS:
                
                current_input = corrupted_input_ids.clone()
                
                try:
                    s_ids_history, metrics_history = langevin_infilling_multiple_positions(
                        model=model,
                        input_ids=current_input,
                        mask_indices=mask_indices,
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

                    final_pred_ids = s_ids_history[-1].tolist()
                    
                    final_sequence_tensor = current_input.clone()
                    final_sequence_tensor[0, mask_indices] = s_ids_history[-1]
                    predicted_full_text = tokenizer.decode(final_sequence_tensor[0], skip_special_tokens=True)

                    matches = sum([1 for pred, true in zip(final_pred_ids, original_token_ids) if pred == true])
                    accuracy_pct = (matches / NUM_MASKS) * 100.0
                    final_metrics = metrics_history[-1]

                    print(f"\n--- Sample {i} | Method: {method_name} | MH: {use_mh} ---")
                    print(f"Original:  {original_full_text}")
                    print(f"Corrupted: {corrupted_full_text}")
                    print(f"Predicted: {predicted_full_text}")
                    print(f"Acc: {accuracy_pct:.1f}% | KL: {final_metrics['avg_kl_divergence']:.4f}")

                    results_data.append({
                        "sample_idx": i,
                        "method": method_name,
                        "mh_enabled": use_mh,
                        "original_text": original_full_text,
                        "corrupted_text": corrupted_full_text,
                        "predicted_text": predicted_full_text,
                        "accuracy_pct": accuracy_pct,
                        "avg_l2_dist": final_metrics["avg_l2_distance"],
                        "avg_kl_div": final_metrics["avg_kl_divergence"],
                        "trajectory": metrics_history 
                    })

                except Exception as e:
                    print(f"Error on Sample {i}: {e}")
                    import traceback
                    traceback.print_exc()

    df = pd.DataFrame(results_data)
    
    cols =["sample_idx", "method", "mh_enabled", "accuracy_pct", "avg_kl_div", "original_text", "predicted_text"]
    print("\n========== SUMMARY HEAD ==========")
    print(df[cols].head())

    # Save as _cls_ to avoid overwriting dls
    output_file = os.path.join(current_dir, f"results/experiment_results_cls_multi_steps_{STEPS}_gn_{GRAD_NORMALIZATION}_oracle_{ORACLE_PARAM}_samples_{N_SAMPLES}_date_{DATE}_peft_{USE_PEFT}_model_name_{MODEL_NAME}.csv")
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    run_multi_mask_experiment()