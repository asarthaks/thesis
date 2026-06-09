import sys
import os


# Get the path to the current file
current_dir = os.path.dirname(os.path.abspath(__file__))

# Go up 3 levels to reach the project root 
# (GradientInformation -> IdealAlphaSchedule -> Experiments -> Root)
project_root = os.path.abspath(os.path.join(current_dir, "../../.."))

# Add the root to sys.path
if project_root not in sys.path:
    sys.path.insert(0, project_root)



import torch
import torch.nn.functional as F
import numpy as np


device = "cuda" if torch.cuda.is_available() else "cpu"

from Methods.Scripts.dls import langevin_infilling_single_position
from Methods.Utils.prep import load_tokenizer_and_model, get_embedding_matrix

# ---------- Example usage ----------
if __name__ == "__main__":
    model_id = "/speech/dbwork/mul/spielwiese3/llm_project/llm_models/pre-trained/Meta-Llama-3-8b/"
    tokenizer, model = load_tokenizer_and_model(model_id, device=device)


    prompt = "Dogs are considered to be our best friends."
    toks = tokenizer(prompt, return_tensors="pt")
    input_ids = toks.input_ids  # (1, L)
    orig_ids = input_ids.clone().detach()
    print("Initial tokens:", tokenizer.decode(input_ids[0]))
    
    # flip some of the tokens
    position = (8, 9)  # last token (0-based)
    input_ids[0, position[0]:position[1]] = 12345 #torch.randint_like(input_ids[0, position[0]:position[1]], 0, 30000)
    print("Distorted tokens:", tokenizer.decode(input_ids[0]))


    token_ids, metrics = langevin_infilling_single_position(model,
        input_ids,
        position,
        tokenizer,
        steps=30,
        temperature=5,
        oracle=True,
        orig_ids=orig_ids,
        alpha_grid=np.logspace(-4, -1, 50),
        method="policy", #options: "policy", "grad_norm_preserved_random_dir", "random",
        mh_sampling=True,
        debug=False)

    # Make a copy of the tensor
    infilled_sequence = toks.input_ids.clone()   # creates an independent tensor
    
    # Replace the token at `position` for the first (and only) sequence in batch
    infilled_sequence[0, position[0]:position[1]] = token_ids[-1]

    # Decode the full infilled sequence to text
    infilled_text = tokenizer.decode(infilled_sequence[0], skip_special_tokens=True)
    

    print("Token id:", token_ids[-1])
    print("Token string:", tokenizer.decode(token_ids[-1]))
    print("Infilled sequence token IDs:", infilled_sequence.tolist())
    print("Infilled text:", infilled_text)
    print(f"Metrics: {metrics}")

    