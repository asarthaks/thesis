import torch
import numpy as np

# 1. Import your Legacy Code (Adjust paths as needed)
from Methods.Scripts.dls import langevin_infilling_multiple_positions as legacy_dls_multi

# 2. Import your New Refactored Code
from core.dls import DiscreteLangevinSampler
from core.prep import load_tokenizer_and_model

# --- SPY MECHANISM: Intercepting Gradients ---
original_grad = torch.autograd.grad
legacy_grads =[]
new_grads =[]
current_grad_list = None 

def hooked_grad(*args, **kwargs):
    """Intercepts torch.autograd.grad to steal a copy of the gradient"""
    grad_tuple = original_grad(*args, **kwargs)
    if current_grad_list is not None:
        # Save a detached CPU copy of the calculated gradient
        current_grad_list.append(grad_tuple[0].detach().cpu().clone())
    return grad_tuple

# Apply the patch
torch.autograd.grad = hooked_grad


def verify_equivalence():
    global current_grad_list
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("Loading model for verification...")
    # Use the base model to keep it simple
    tokenizer, model = load_tokenizer_and_model("/mount/studenten-temp1/users/singhsk/thesis/thesis/gfn-lm-tuning/infill_subj_arithmetic/gpt2_large_sft_output", device=device)

    # --- Freeze the Inputs ---
    text = "The quick brown fox jumps over the lazy dog."
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
    
    # Hardcode mask indices
    mask_indices =[2] # Masking "brown" and "jumps"
    orig_ids = input_ids.clone()
    
    # Hardcode the corrupted tokens (e.g., token ID 1000 and 2000)
    corrupted_ids = input_ids.clone()
    corrupted_ids[0, mask_indices[0]] = 1000
    # corrupted_ids[0, mask_indices[1]] = 2000

    # Shared hyperparameters
    steps = 10
    temperature = 5.0
    method = "policy"
    mh_sampling = True
    grad_norm = True

    print("\n--- Running Legacy Code ---")
    current_grad_list = legacy_grads  # Start spying for legacy
    
    # STRICT SEEDING: Right before the loop
    torch.manual_seed(42)
    np.random.seed(42)
    
    legacy_s_ids, legacy_metrics = legacy_dls_multi(
        model=model,
        input_ids=corrupted_ids.clone(),
        mask_indices=mask_indices,
        tokenizer=tokenizer,
        steps=steps,
        temperature=temperature,
        oracle=False,
        orig_ids=orig_ids.clone(),
        method=method,
        grad_normalization=grad_norm,
        mh_sampling=mh_sampling,
        debug=False
    )

    print("\n--- Running New OO Code ---")
    current_grad_list = new_grads  # Start spying for new code
    
    # STRICT SEEDING: Reset the exact same seed right before the new loop
    torch.manual_seed(42)
    np.random.seed(42)
    
    sampler = DiscreteLangevinSampler(
        model=model, tokenizer=tokenizer, steps=steps, temperature=temperature,
        oracle=False, method=method, mh_sampling=mh_sampling, grad_normalization=grad_norm
    )
    
    new_s_ids, new_metrics = sampler.optimize(corrupted_ids.clone(), mask_indices, orig_ids.clone())

    # --- Restore original PyTorch behavior ---
    torch.autograd.grad = original_grad

    print("\n========== VERIFICATION RESULTS ==========")
    passed = True
    
    # 1. Check Gradient Calls Count
    # (If MH is True, there should be 2 gradients per step: Forward and Backward)
    if len(legacy_grads) != len(new_grads):
        print(f"[FAIL] Gradient count mismatch! Legacy calculated {len(legacy_grads)} grads, New calculated {len(new_grads)}.")
        passed = False
    else:
        # 2. Check Exact Gradient Values
        for i in range(len(legacy_grads)):
            leg_g = legacy_grads[i]
            new_g = new_grads[i]
            print(leg_g, new_g)
            
            # Use torch.allclose to check if tensors are mathematically identical
            if not torch.allclose(leg_g, new_g, atol=1e-6):
                print(f"[FAIL] Gradient tensor mismatch at gradient calculation #{i}!")
                passed = False
                break
        else:
            print(f"[PASS] All {len(legacy_grads)} raw gradient tensors matched perfectly.")

    # 3. Compare Trajectories and Metrics
    for step in range(steps):
        leg_m = legacy_metrics[step]
        new_m = new_metrics[step]

        if leg_m["token_ids"] != new_m["token_ids"]:
            print(f"[FAIL] Step {step} Token IDs mismatch! Legacy: {leg_m['token_ids']} | New: {new_m['token_ids']}")
            passed = False

        if not np.isclose(leg_m["avg_kl_divergence"], new_m["avg_kl_divergence"], atol=1e-5):
            print(f"[FAIL] Step {step} KL Div mismatch! Legacy: {leg_m['avg_kl_divergence']} | New: {new_m['avg_kl_divergence']}")
            passed = False

        if not np.isclose(leg_m["avg_l2_distance"], new_m["avg_l2_distance"], atol=1e-5):
            print(f"[FAIL] Step {step} L2 Dist mismatch! Legacy: {leg_m['avg_l2_distance']} | New: {new_m['avg_l2_distance']}")
            passed = False

    if passed:
        print("\n✅ SUCCESS! The refactored OO code is mathematically and physically identical to the legacy code.")
    else:
        print("\n❌ FAILED! There is a divergence in the math, gradients, or random number generation sequence.")

if __name__ == "__main__":
    verify_equivalence()