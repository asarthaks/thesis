import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


device = "cuda" if torch.cuda.is_available() else "cpu"

def load_tokenizer_and_model(model_id: str, device=device, dtype=None):
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model_kwargs = {}
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype
    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    model.to(device)
    model.eval()
    return tokenizer, model

def load_tokenizer_and_model_peft(model_id: str, checkpoint: str, device=device, dtype=None):
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model_kwargs = {}
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype
    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    # 2. Load the GFlowNet LoRA adapter on top of it
    peft_model = PeftModel.from_pretrained(model, checkpoint)
    model = peft_model.merge_and_unload()
    model.to(device)
    model.eval()
    return tokenizer, model

def get_embedding_matrix(model):
    # returns tensor (vocab_size, emb_dim)
    emb_layer = model.get_input_embeddings()
    return emb_layer.weight.detach()  # (V, D)

def project_to_vocab_by_l2(s: torch.FloatTensor, emb_matrix: torch.FloatTensor):
    """
    Projects a continuous embedding or batch of embeddings to the nearest token in the
    embedding matrix using Euclidean (L2) distance.

    Args:
        s: (..., D) continuous embedding(s) — can be (D,), (B, D), or (B, T, D)
        emb_matrix: (V, D) vocabulary embedding matrix

    Returns:
        top_idx: tensor of nearest token indices with shape matching s[..., 0]
    """
    # Ensure last dimension is embedding dim
    D = emb_matrix.size(1)
    assert s.size(-1) == D, f"Embedding dim mismatch: s has {s.size(-1)}, emb_matrix has {D}"

    original_shape = s.shape[:-1]  # e.g. (1, 2)
    s_flat = s.view(-1, D)         # Flatten to (N, D)

    # Compute squared Euclidean distance to all vocab embeddings
    diff = s_flat[:, None, :] - emb_matrix[None, :, :]  # (N, V, D)
    dist2 = torch.sum(diff ** 2, dim=-1)                # (N, V)

    # Find nearest vocab index
    top_idx = dist2.argmin(dim=-1)                      # (N,)

    # Reshape back to match original shape (excluding embedding dim)
    top_idx = top_idx.view(*original_shape)             # (B, T) or whatever s was

    return top_idx
